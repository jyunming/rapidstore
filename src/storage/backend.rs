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
        data: Vec<u8>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// List files matching a prefix.
    fn list(&self, prefix: &str) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>>;

    /// Delete a file.
    fn delete(&self, path: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Check if a file exists.
    fn exists(&self, path: &str) -> bool;
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
        data: Vec<u8>,
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

        // Debug
        // println!("Listing root: {:?}, prefix: {}, search_path: {:?}", self.root, prefix, search_path);

        let target_dir = if search_path.is_dir() {
            search_path
        } else {
            search_path.parent().unwrap_or(&self.root).to_path_buf()
        };

        if target_dir.exists() {
            for entry in fs::read_dir(target_dir)? {
                let entry = entry?;
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.starts_with(prefix) || prefix.is_empty() {
                    results.push(name);
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
        data: Vec<u8>,
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
}
