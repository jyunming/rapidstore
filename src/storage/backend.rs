use std::path::PathBuf;

/// Storage backend configuration.
/// Phase 1-3 supports local filesystem only.
/// Phase 5 will add cloud backends (S3, GCS, Azure) via the `cloud` feature flag.
#[derive(Clone, Debug)]
pub struct StorageBackend {
    pub local_path: PathBuf,
    pub uri: String,
}

impl StorageBackend {
    /// Parse a URI into a StorageBackend.
    /// Currently only local paths are supported.
    /// Cloud URIs (s3://, gs://, az://) are stubbed for Phase 5.
    pub fn from_uri(uri: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if uri.starts_with("s3://") || uri.starts_with("gs://") || uri.starts_with("az://") {
            return Err(format!(
                "Cloud storage backend '{}' requires the `cloud` feature. \
                 Compile with `cargo build --features cloud`. \
                 Tracking: Phase 5 implementation.",
                uri
            ).into());
        }
        // Local filesystem (file:// or bare path)
        let local_path = uri.strip_prefix("file://").unwrap_or(uri);
        std::fs::create_dir_all(local_path)?;
        Ok(Self {
            local_path: PathBuf::from(local_path),
            uri: uri.to_string(),
        })
    }

    pub fn child_path(&self, name: &str) -> PathBuf {
        self.local_path.join(name)
    }
}
