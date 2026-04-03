use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::path::PathBuf;

const GROW_SLOTS: usize = 16384;

/// Memory-mapped slab for quantized vectors or raw float vectors.
///
/// # Windows safety
/// On Windows, a file cannot be renamed or overwritten while any handle (mmap
/// *or* the underlying `File`) is open against it.  Call [`release_handles`]
/// before any `fs::rename` / overwrite that targets this file; the next call
/// to any mutating method will reopen the handle automatically.
pub struct LiveCodesFile {
    path: PathBuf,
    file: Option<File>,
    mmap: Option<MmapMut>,
    stride: usize,
    capacity: usize,
    len: usize,
}

impl LiveCodesFile {
    pub fn open(
        path: PathBuf,
        stride: usize,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;

        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;
        let capacity = if stride > 0 { file_size / stride } else { 0 };
        let len = capacity;

        let mut live_codes = Self {
            path,
            file: Some(file),
            mmap: None,
            stride,
            capacity,
            len,
        };

        if file_size > 0 {
            live_codes.remap()?;
        }

        Ok(live_codes)
    }

    /// Ensure the file handle is open, reopening from the stored path if needed.
    fn ensure_open(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.file.is_none() {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(&self.path)?;
            self.file = Some(file);
        }
        Ok(())
    }

    fn remap(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.mmap = None;
        self.ensure_open()?;
        if self.capacity > 0 {
            let mmap = unsafe {
                MmapOptions::new().map_mut(self.file.as_ref().expect("file handle open"))?
            };
            self.mmap = Some(mmap);
        }
        Ok(())
    }

    pub fn get_slot(&self, slot: usize) -> &[u8] {
        let start = slot * self.stride;
        let end = start + self.stride;
        &self.mmap.as_ref().expect("mmap not initialized")[start..end]
    }

    pub fn get_slot_mut(&mut self, slot: usize) -> &mut [u8] {
        let start = slot * self.stride;
        let end = start + self.stride;
        &mut self.mmap.as_mut().expect("mmap not initialized")[start..end]
    }

    pub fn alloc_slot(&mut self) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        // Re-establish mmap if handles were released (e.g. after release_handles())
        // and we still have capacity — growth path already calls ensure_open + remap.
        if self.mmap.is_none() && self.len < self.capacity {
            self.ensure_open()?;
            self.remap()?;
        }
        if self.len >= self.capacity {
            let new_capacity = self.capacity + GROW_SLOTS;
            self.mmap = None; // mmap must be dropped before set_len on Windows
            self.ensure_open()?;
            self.file
                .as_ref()
                .unwrap()
                .set_len((new_capacity * self.stride) as u64)?;
            self.capacity = new_capacity;
            self.remap()?;
        }
        let slot = self.len;
        self.len += 1;
        Ok(slot)
    }

    pub fn truncate_to(
        &mut self,
        new_len: usize,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.mmap = None; // mmap must be dropped before set_len on Windows
        self.ensure_open()?;
        self.file
            .as_ref()
            .unwrap()
            .set_len((new_len * self.stride) as u64)?;
        self.capacity = new_len;
        self.len = new_len;
        self.remap()?;
        Ok(())
    }

    pub fn flush(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(mmap) = &self.mmap {
            mmap.flush()?;
        }
        Ok(())
    }

    /// Release both the mmap and the OS file handle.
    ///
    /// Required on Windows before any `fs::rename` or overwrite targeting
    /// this file.  The handle is reopened automatically on the next access.
    pub fn release_handles(&mut self) {
        self.mmap = None;
        self.file = None;
    }

    /// Release the mmap only (kept for callers that don't need full handle release).
    pub fn release_mmap(&mut self) {
        self.mmap = None;
    }

    /// Hint the OS that this mmap will be accessed randomly (not sequentially).
    /// Disables read-ahead on Linux, reducing page-cache pressure for sparse ANN lookups.
    /// No-op on platforms that don't support madvise.
    #[allow(unused_variables)]
    pub fn advise_random(&self) {
        #[cfg(unix)]
        if let Some(mmap) = &self.mmap {
            let _ = mmap.advise(memmap2::Advice::Random);
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Override `len` without modifying the file.
    ///
    /// Used by the engine after loading the IdPool on reopen: the file may be
    /// pre-allocated to `capacity > actual_len` slots, so `len` must be
    /// corrected to match the number of populated slots tracked by the IdPool.
    pub fn set_len(&mut self, len: usize) {
        debug_assert!(
            len <= self.capacity,
            "set_len({len}) exceeds capacity({})",
            self.capacity
        );
        self.len = len;
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn byte_len(&self) -> usize {
        self.len * self.stride
    }

    /// Returns a read-only view of the entire mmap, safe to share across threads.
    pub fn as_bytes(&self) -> &[u8] {
        match &self.mmap {
            Some(m) => &m[..],
            None => &[],
        }
    }

    pub fn clear(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.mmap = None; // mmap must be dropped before set_len on Windows
        self.ensure_open()?;
        self.file.as_ref().unwrap().set_len(0)?;
        self.len = 0;
        self.capacity = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    const STRIDE: usize = 16;

    fn make_lc() -> (tempfile::TempDir, LiveCodesFile) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("live_codes.bin");
        let lc = LiveCodesFile::open(path, STRIDE).unwrap();
        (dir, lc)
    }

    // -----------------------------------------------------------------------
    // open / basic state
    // -----------------------------------------------------------------------

    #[test]
    fn open_empty_file_has_zero_len() {
        let (_dir, lc) = make_lc();
        assert_eq!(lc.len(), 0);
        assert_eq!(lc.stride(), STRIDE);
        assert_eq!(lc.byte_len(), 0);
    }

    #[test]
    fn as_bytes_empty_returns_empty_slice() {
        let (_dir, lc) = make_lc();
        assert_eq!(lc.as_bytes().len(), 0);
    }

    // -----------------------------------------------------------------------
    // alloc_slot
    // -----------------------------------------------------------------------

    #[test]
    fn alloc_first_slot_returns_zero() {
        let (_dir, mut lc) = make_lc();
        let slot = lc.alloc_slot().unwrap();
        assert_eq!(slot, 0);
        assert_eq!(lc.len(), 1);
    }

    #[test]
    fn alloc_multiple_slots_are_sequential() {
        let (_dir, mut lc) = make_lc();
        for i in 0..5usize {
            let slot = lc.alloc_slot().unwrap();
            assert_eq!(slot, i);
        }
        assert_eq!(lc.len(), 5);
    }

    #[test]
    fn alloc_grows_capacity_in_chunks() {
        let (_dir, mut lc) = make_lc();
        // Allocate one slot — forces a GROW_SLOTS (16384) expansion
        lc.alloc_slot().unwrap();
        assert!(lc.capacity >= GROW_SLOTS);
    }

    // -----------------------------------------------------------------------
    // get_slot / get_slot_mut
    // -----------------------------------------------------------------------

    #[test]
    fn write_and_read_slot_data() {
        let (_dir, mut lc) = make_lc();
        lc.alloc_slot().unwrap();
        let data = [42u8; STRIDE];
        lc.get_slot_mut(0).copy_from_slice(&data);
        assert_eq!(lc.get_slot(0), &data);
    }

    #[test]
    fn two_slots_do_not_overlap() {
        let (_dir, mut lc) = make_lc();
        lc.alloc_slot().unwrap();
        lc.alloc_slot().unwrap();

        let data0 = [1u8; STRIDE];
        let data1 = [2u8; STRIDE];
        lc.get_slot_mut(0).copy_from_slice(&data0);
        lc.get_slot_mut(1).copy_from_slice(&data1);

        assert_eq!(lc.get_slot(0), &data0, "slot 0 corrupted by slot 1 write");
        assert_eq!(lc.get_slot(1), &data1, "slot 1 data incorrect");
    }

    // -----------------------------------------------------------------------
    // truncate_to
    // -----------------------------------------------------------------------

    #[test]
    fn truncate_to_shrinks_len() {
        let (_dir, mut lc) = make_lc();
        for _ in 0..5 {
            lc.alloc_slot().unwrap();
        }
        lc.truncate_to(2).unwrap();
        assert_eq!(lc.len(), 2);
        assert_eq!(lc.capacity, 2);
    }

    #[test]
    fn truncate_to_zero_empties() {
        let (_dir, mut lc) = make_lc();
        lc.alloc_slot().unwrap();
        lc.truncate_to(0).unwrap();
        assert_eq!(lc.len(), 0);
        assert_eq!(lc.byte_len(), 0);
    }

    // -----------------------------------------------------------------------
    // flush
    // -----------------------------------------------------------------------

    #[test]
    fn flush_with_data_does_not_panic() {
        let (_dir, mut lc) = make_lc();
        lc.alloc_slot().unwrap();
        lc.flush().unwrap();
    }

    #[test]
    fn flush_empty_does_not_panic() {
        let (_dir, lc) = make_lc();
        lc.flush().unwrap();
    }

    // -----------------------------------------------------------------------
    // release_handles / release_mmap
    // -----------------------------------------------------------------------

    #[test]
    fn release_handles_then_alloc_reopens() {
        let (_dir, mut lc) = make_lc();
        lc.alloc_slot().unwrap();
        let data = [77u8; STRIDE];
        lc.get_slot_mut(0).copy_from_slice(&data);
        lc.flush().unwrap();

        lc.release_handles();
        // After release, alloc_slot must reopen the file
        let slot = lc.alloc_slot().unwrap();
        assert_eq!(slot, 1, "len should resume at 1 after reopen");
    }

    #[test]
    fn release_mmap_then_alloc_continues() {
        let (_dir, mut lc) = make_lc();
        lc.alloc_slot().unwrap();
        lc.release_mmap();
        let slot = lc.alloc_slot().unwrap();
        assert_eq!(slot, 1);
    }

    // -----------------------------------------------------------------------
    // advise_random (no-op on non-Unix, no panic anywhere)
    // -----------------------------------------------------------------------

    #[test]
    fn advise_random_does_not_panic() {
        let (_dir, mut lc) = make_lc();
        lc.alloc_slot().unwrap();
        lc.advise_random();
    }

    // -----------------------------------------------------------------------
    // as_bytes
    // -----------------------------------------------------------------------

    #[test]
    fn as_bytes_length_matches_capacity_times_stride() {
        let (_dir, mut lc) = make_lc();
        lc.alloc_slot().unwrap();
        // After first alloc the file grows by GROW_SLOTS
        assert_eq!(lc.as_bytes().len(), lc.capacity * STRIDE);
    }

    // -----------------------------------------------------------------------
    // clear
    // -----------------------------------------------------------------------

    #[test]
    fn clear_resets_to_empty() {
        let (_dir, mut lc) = make_lc();
        lc.alloc_slot().unwrap();
        lc.alloc_slot().unwrap();
        lc.clear().unwrap();
        assert_eq!(lc.len(), 0);
        assert_eq!(lc.byte_len(), 0);
        assert_eq!(lc.capacity, 0);
    }

    // -----------------------------------------------------------------------
    // byte_len helper
    // -----------------------------------------------------------------------

    #[test]
    fn byte_len_equals_len_times_stride() {
        let (_dir, mut lc) = make_lc();
        for _ in 0..3 {
            lc.alloc_slot().unwrap();
        }
        assert_eq!(lc.byte_len(), lc.len() * STRIDE);
    }

    // ── ensure_open() after release_handles() (lines 58-63) ─────────────────
    // release_handles() sets file=None; next call to truncate_to() calls
    // ensure_open() which reopens the file, covering lines 58-63.

    #[test]
    fn truncate_to_after_release_handles_reopens_file() {
        let (_dir, mut lc) = make_lc();
        lc.alloc_slot().unwrap();
        assert_eq!(lc.len(), 1);
        // release_handles() sets both file=None and mmap=None
        lc.release_handles();
        // truncate_to calls ensure_open() → file.is_none() → opens file → lines 58-63
        lc.truncate_to(0).unwrap();
        assert_eq!(lc.len(), 0);
    }
}
