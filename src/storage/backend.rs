use std::fs;
use std::path::{Path, PathBuf};
use url::Url;

/// Trait inspired by object_store but simplified and synchronous for Phase 4.
pub trait StorageProvider: Send + Sync {
    /// Read a file into memory.
    fn read(&self, path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>>;

    /// Write a file from memory.
    fn write(
        &self,
        path: &str,
        data: &[u8],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// List files matching a prefix.
    fn list(&self, prefix: &str) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>>;

    /// Delete a file.
    fn delete(&self, path: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Check if a file exists.
    fn exists(&self, path: &str) -> bool;

    /// Get file size.
    fn size(&self, path: &str) -> Result<u64, Box<dyn std::error::Error + Send + Sync>>;
}

/// Local filesystem storage provider.
pub struct LocalProvider {
    root: PathBuf,
}

impl LocalProvider {
    pub fn new<P: AsRef<Path>>(root: P) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }
}

impl StorageProvider for LocalProvider {
    fn read(&self, path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        let full = self.root.join(path);
        Ok(fs::read(full)?)
    }

    fn write(
        &self,
        path: &str,
        data: &[u8],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let full = self.root.join(path);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(full, data)?;
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let mut results = Vec::new();
        let search_path = self.root.join(prefix);

        // Determine the directory to enumerate and the file-name prefix to filter on.
        // If `prefix` contains path separators (e.g. "subdir/seg-"), split at the last
        // separator so we enumerate the correct subdirectory and filter by filename only.
        let (target_dir, name_prefix): (PathBuf, &str) = if search_path.is_dir() {
            (search_path, "")
        } else {
            let parent = search_path.parent().unwrap_or(&self.root).to_path_buf();
            let name_part = Path::new(prefix)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(prefix);
            (parent, name_part)
        };

        if target_dir.exists() {
            let dir_prefix = target_dir
                .strip_prefix(&self.root)
                .ok()
                .and_then(|p| p.to_str())
                .map(|s| {
                    if s.is_empty() {
                        String::new()
                    } else {
                        format!("{}/", s)
                    }
                })
                .unwrap_or_default();

            for entry in fs::read_dir(&target_dir)? {
                let entry = entry?;
                let name = entry.file_name().to_string_lossy().into_owned();
                if name_prefix.is_empty() || name.starts_with(name_prefix) {
                    // Return paths relative to root so callers can use them with other backend ops
                    results.push(format!("{}{}", dir_prefix, name));
                }
            }
        }
        Ok(results)
    }

    fn delete(&self, path: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let full = self.root.join(path);
        if full.exists() {
            fs::remove_file(full)?;
        }
        Ok(())
    }

    fn exists(&self, path: &str) -> bool {
        self.root.join(path).exists()
    }

    fn size(&self, path: &str) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let full = self.root.join(path);
        Ok(fs::metadata(full)?.len())
    }
}

/// A unified storage backend wrapper.
pub struct StorageBackend {
    pub provider: Box<dyn StorageProvider>,
    pub base_uri: String,
}

impl StorageBackend {
    pub fn from_uri(uri: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if uri.starts_with("s3://") || uri.starts_with("gs://") {
            return Err("Cloud storage requires the 'cloud' feature. Phase 4 provides local-first stability.".into());
        }

        let path = if uri.starts_with("file://") {
            let url = Url::parse(uri)?;
            url.to_file_path().map_err(|_| "Invalid file URI path")?
        } else {
            PathBuf::from(uri)
        };

        fs::create_dir_all(&path)?;
        Ok(Self {
            provider: Box::new(LocalProvider::new(path)),
            base_uri: uri.to_string(),
        })
    }

    pub fn read(&self, path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        self.provider.read(path)
    }

    pub fn write(
        &self,
        path: &str,
        data: &[u8],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.provider.write(path, data)
    }

    pub fn list(
        &self,
        prefix: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        self.provider.list(prefix)
    }

    pub fn delete(&self, path: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.provider.delete(path)
    }

    pub fn exists(&self, path: &str) -> bool {
        self.provider.exists(path)
    }

    pub fn size(&self, path: &str) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        self.provider.size(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn make_backend() -> (tempfile::TempDir, StorageBackend) {
        let dir = tempdir().unwrap();
        let backend = StorageBackend::from_uri(dir.path().to_str().unwrap()).unwrap();
        (dir, backend)
    }

    #[test]
    fn from_uri_local_path_creates_dir() {
        let dir = tempdir().unwrap();
        let new_dir = dir.path().join("mydb");
        let _backend = StorageBackend::from_uri(new_dir.to_str().unwrap()).unwrap();
        assert!(new_dir.is_dir());
    }

    #[test]
    fn from_uri_file_scheme() {
        let dir = tempdir().unwrap();
        // Build a valid file:// URI (forward slashes, no drive-letter colon issue on Windows)
        let path_str = dir.path().to_str().unwrap().replace('\\', "/");
        let uri = if path_str.starts_with('/') {
            format!("file://{}", path_str)
        } else {
            // Windows: "C:/..." → "file:///C:/..."
            format!("file:///{}", path_str)
        };
        let result = StorageBackend::from_uri(&uri);
        assert!(
            result.is_ok(),
            "file:// URI should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn from_uri_cloud_rejected() {
        let result = StorageBackend::from_uri("s3://my-bucket/path");
        assert!(result.is_err(), "s3:// should be rejected");
        let result2 = StorageBackend::from_uri("gs://my-bucket/path");
        assert!(result2.is_err(), "gs:// should be rejected");
    }

    #[test]
    fn write_and_read_roundtrip() {
        let (_dir, backend) = make_backend();
        let data = b"hello world";
        backend.write("test.bin", data).unwrap();
        let read = backend.read("test.bin").unwrap();
        assert_eq!(read.as_slice(), data);
    }

    #[test]
    fn exists_returns_correct_values() {
        let (_dir, backend) = make_backend();
        assert!(!backend.exists("nonexistent.bin"));
        backend.write("file.bin", b"data").unwrap();
        assert!(backend.exists("file.bin"));
    }

    #[test]
    fn delete_removes_file() {
        let (_dir, backend) = make_backend();
        backend.write("to_delete.bin", b"bye").unwrap();
        assert!(backend.exists("to_delete.bin"));
        backend.delete("to_delete.bin").unwrap();
        assert!(!backend.exists("to_delete.bin"));
    }

    #[test]
    fn delete_nonexistent_is_ok() {
        let (_dir, backend) = make_backend();
        // Should not error on missing file
        let result = backend.delete("does_not_exist.bin");
        assert!(result.is_ok());
    }

    #[test]
    fn size_returns_correct_value() {
        let (_dir, backend) = make_backend();
        let data = b"12345";
        backend.write("sized.bin", data).unwrap();
        let size = backend.size("sized.bin").unwrap();
        assert_eq!(size, 5);
    }

    #[test]
    fn list_empty_prefix_returns_all_files() {
        let (_dir, backend) = make_backend();
        backend.write("a.bin", b"a").unwrap();
        backend.write("b.bin", b"b").unwrap();
        backend.write("c.txt", b"c").unwrap();
        let mut files = backend.list("").unwrap();
        files.sort();
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn list_with_name_prefix_filters() {
        let (_dir, backend) = make_backend();
        backend.write("seg-00000000.bin", b"s0").unwrap();
        backend.write("seg-00000001.bin", b"s1").unwrap();
        backend.write("manifest.json", b"{}").unwrap();
        let files = backend.list("seg-").unwrap();
        assert_eq!(files.len(), 2);
        assert!(files.iter().all(|f| f.contains("seg-")));
    }

    #[test]
    fn list_with_subdir_prefix() {
        let (dir, backend) = make_backend();
        // Create a subdirectory with files via the backend
        fs::create_dir_all(dir.path().join("subdir")).unwrap();
        backend.write("subdir/file1.bin", b"f1").unwrap();
        backend.write("subdir/file2.bin", b"f2").unwrap();
        backend.write("root.bin", b"r").unwrap();

        // List all files in root — should include root.bin and subdir entry
        let all = backend.list("").unwrap();
        assert!(all.iter().any(|f| f.contains("root")));

        // List with subdir prefix — should list files in that directory
        let sub = backend.list("subdir").unwrap();
        assert_eq!(sub.len(), 2);
        assert!(sub.iter().all(|f| f.contains("subdir")));
    }

    #[test]
    fn list_nonexistent_subdir_returns_empty() {
        let (_dir, backend) = make_backend();
        let files = backend.list("nonexistent_subdir/").unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn write_creates_intermediate_directories() {
        let (_dir, backend) = make_backend();
        backend.write("deep/nested/file.bin", b"data").unwrap();
        assert!(backend.exists("deep/nested/file.bin"));
    }

    #[test]
    fn overwrite_replaces_content() {
        let (_dir, backend) = make_backend();
        backend.write("file.bin", b"old").unwrap();
        backend.write("file.bin", b"new content").unwrap();
        let data = backend.read("file.bin").unwrap();
        assert_eq!(data.as_slice(), b"new content");
    }

    #[test]
    fn base_uri_stored() {
        let dir = tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let backend = StorageBackend::from_uri(uri).unwrap();
        assert_eq!(backend.base_uri, uri);
    }
}
