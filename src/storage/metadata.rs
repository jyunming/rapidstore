use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use redb::{Database as RedbDatabase, TableDefinition, ReadableTable};

const METADATA_TABLE: TableDefinition<&str, &str> = TableDefinition::new("metadata");

/// Arbitrary metadata stored alongside a vector.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct VectorMetadata {
    pub properties: HashMap<String, serde_json::Value>,
}

/// Embedded B-Tree metadata store backed by redb.
/// Stores JSON-serialized VectorMetadata keyed by vector ID string.
pub struct MetadataStore {
    db: RedbDatabase,
}

impl MetadataStore {
    /// Open or create the metadata store at the given path.
    pub fn open(path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let db = RedbDatabase::create(path)?;
        // Ensure table exists
        let write_txn = db.begin_write()?;
        { let _table = write_txn.open_table(METADATA_TABLE)?; }
        write_txn.commit()?;
        Ok(Self { db })
    }

    /// Insert or update metadata for a vector ID.
    pub fn put(&self, id: &str, meta: &VectorMetadata) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let json = serde_json::to_string(meta)?;
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(METADATA_TABLE)?;
            table.insert(id, json.as_str())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Retrieve metadata for a vector ID.
    pub fn get(&self, id: &str) -> Result<Option<VectorMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(METADATA_TABLE)?;
        if let Some(value) = table.get(id)? {
            let meta: VectorMetadata = serde_json::from_str(value.value())?;
            Ok(Some(meta))
        } else {
            Ok(None)
        }
    }

    /// Delete metadata for a vector ID.
    pub fn delete(&self, id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(METADATA_TABLE)?;
            table.remove(id)?;
        }
        write_txn.commit()?;
        Ok(())
    }
}
