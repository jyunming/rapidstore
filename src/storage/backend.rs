use std::fs;
use std::path::{Path, PathBuf};
use url::Url;

#[cfg(feature = "cloud")]
use futures::TryStreamExt;
#[cfg(feature = "cloud")]
use object_store::{ObjectStore, path::Path as OsPath};
#[cfg(feature = "cloud")]
use std::sync::Arc;

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

// ---------------------------------------------------------------------------
// S3Provider — write-through local cache backed by Amazon S3
// ---------------------------------------------------------------------------

/// S3-backed storage provider with a local write-through cache.
///
/// All files are written locally first, then uploaded to S3.  Reads are served
/// from the local cache; on a cache miss the object is fetched from S3 and
/// cached.  The local cache mirrors the S3 key hierarchy so that mmap-based
/// reads (live_codes.bin, graph.bin) always hit local paths.
///
/// **Thread safety:** each `S3Provider` owns a dedicated single-thread tokio
/// runtime used for all async S3 calls.  This is safe to call from both sync
/// embedded code and from inside a tokio `spawn_blocking` task.
#[cfg(feature = "cloud")]
pub struct S3Provider {
    store: Arc<dyn ObjectStore>,
    prefix: String,
    local_cache: PathBuf,
    rt: tokio::runtime::Runtime,
}

#[cfg(feature = "cloud")]
impl S3Provider {
    /// Construct a new S3Provider.
    ///
    /// AWS credentials are read from the standard environment variables
    /// (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, etc.) or
    /// from the instance metadata service.
    pub fn new(
        bucket: &str,
        prefix: &str,
        local_cache: &Path,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let store = object_store::aws::AmazonS3Builder::from_env()
            .with_bucket_name(bucket)
            .build()?;
        fs::create_dir_all(local_cache)?;
        // Detect accidental use from inside an existing Tokio runtime.
        // S3Provider is designed for synchronous callers (embedded engine or
        // spawn_blocking); calling block_on() inside an async context panics.
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err("S3Provider::new() called from within a Tokio runtime; \
                 use spawn_blocking() to invoke S3 operations from async code"
                .into());
        }
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        Ok(Self {
            store: Arc::new(store),
            prefix: prefix.trim_matches('/').to_string(),
            local_cache: local_cache.to_path_buf(),
            rt,
        })
    }

    fn s3_key(&self, key: &str) -> OsPath {
        if self.prefix.is_empty() {
            OsPath::from(key)
        } else {
            OsPath::from(format!("{}/{}", self.prefix, key))
        }
    }

    fn cached(&self, key: &str) -> PathBuf {
        self.local_cache.join(key)
    }
}

#[cfg(feature = "cloud")]
impl StorageProvider for S3Provider {
    fn read(&self, path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        let cached = self.cached(path);
        if cached.exists() {
            return Ok(fs::read(&cached)?);
        }
        // Cache miss: fetch from S3 and populate local cache.
        let key = self.s3_key(path);
        let bytes = self.rt.block_on(async {
            let result = self.store.get(&key).await?;
            result.bytes().await
        })?;
        if let Some(parent) = cached.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&cached, &bytes)?;
        Ok(bytes.to_vec())
    }

    fn write(
        &self,
        path: &str,
        data: &[u8],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Write to local cache first.
        let cached = self.cached(path);
        if let Some(parent) = cached.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&cached, data)?;
        // Upload to S3.
        let key = self.s3_key(path);
        let payload = object_store::PutPayload::from(data.to_vec());
        self.rt.block_on(self.store.put(&key, payload))?;
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        // List from S3 (authoritative); strip the provider prefix.
        let key_prefix = self.s3_key(prefix);
        let objects: Vec<object_store::ObjectMeta> = self.rt.block_on(async {
            self.store
                .list(Some(&key_prefix))
                .try_collect::<Vec<_>>()
                .await
        })?;
        let provider_prefix = if self.prefix.is_empty() {
            None
        } else {
            Some(format!("{}/", self.prefix))
        };
        Ok(objects
            .into_iter()
            .map(|o| {
                let loc = o.location.as_ref();
                provider_prefix
                    .as_deref()
                    .and_then(|p| loc.strip_prefix(p))
                    .unwrap_or(loc)
                    .to_string()
            })
            .collect())
    }

    fn delete(&self, path: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let cached = self.cached(path);
        if cached.exists() {
            fs::remove_file(&cached)?;
        }
        let key = self.s3_key(path);
        self.rt.block_on(self.store.delete(&key))?;
        Ok(())
    }

    fn exists(&self, path: &str) -> bool {
        if self.cached(path).exists() {
            return true;
        }
        let key = self.s3_key(path);
        self.rt.block_on(self.store.head(&key)).is_ok()
    }

    fn size(&self, path: &str) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let cached = self.cached(path);
        if cached.exists() {
            return Ok(fs::metadata(&cached)?.len());
        }
        let key = self.s3_key(path);
        let meta = self.rt.block_on(self.store.head(&key))?;
        Ok(meta.size as u64)
    }
}

/// A unified storage backend wrapper.
pub struct StorageBackend {
    pub provider: Box<dyn StorageProvider>,
    pub base_uri: String,
}

impl StorageBackend {
    pub fn from_uri(uri: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        #[cfg(feature = "cloud")]
        if uri.starts_with("s3://") {
            return Self::from_s3_uri(uri);
        }

        if uri.starts_with("gs://") || uri.starts_with("az://") {
            return Err(
                "gs:// and az:// cloud storage are not yet supported. Use s3:// (requires the `cloud` feature) or a local path.".into(),
            );
        }
        if !cfg!(feature = "cloud") && uri.starts_with("s3://") {
            return Err(
                "s3:// requires the `cloud` Cargo feature. Rebuild with: cargo build --features cloud".into(),
            );
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

    #[cfg(feature = "cloud")]
    fn from_s3_uri(uri: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Parse s3://{bucket}/{prefix}
        let without_scheme = uri.strip_prefix("s3://").unwrap_or(uri);
        let (bucket, prefix) = without_scheme
            .split_once('/')
            .map(|(b, p)| (b, p))
            .unwrap_or((without_scheme, ""));

        // Local cache lives next to the process working directory under .tqdb_s3_cache/{bucket}
        let cache_dir = std::env::var("TQDB_S3_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(".tqdb_s3_cache"))
            .join(bucket)
            .join(prefix.replace('/', "_"));

        let provider = S3Provider::new(bucket, prefix, &cache_dir)?;
        Ok(Self {
            provider: Box::new(provider),
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
    fn from_uri_gs_az_rejected() {
        let result2 = StorageBackend::from_uri("gs://my-bucket/path");
        assert!(result2.is_err(), "gs:// should be rejected");
    }

    #[cfg(not(feature = "cloud"))]
    #[test]
    fn from_uri_s3_rejected_without_cloud_feature() {
        let result = StorageBackend::from_uri("s3://my-bucket/path");
        assert!(
            result.is_err(),
            "s3:// should be rejected without cloud feature"
        );
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
