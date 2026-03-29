use redb::{Database as RedbDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const METADATA_TABLE: TableDefinition<&str, &str> = TableDefinition::new("metadata");

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct VectorMetadata {
    pub properties: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub document: Option<String>,
}

pub struct MetadataStore {
    db: RedbDatabase,
}

impl MetadataStore {
    pub fn open(path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let db = RedbDatabase::create(path)?;
        let write_txn = db.begin_write()?;
        {
            let _table = write_txn.open_table(METADATA_TABLE)?;
        }
        write_txn.commit()?;
        Ok(Self { db })
    }

    pub fn put(
        &self,
        id: &str,
        meta: &VectorMetadata,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let json = serde_json::to_string(meta)?;
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(METADATA_TABLE)?;
            table.insert(id, json.as_str())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn put_many(
        &self,
        entries: &[(String, VectorMetadata)],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if entries.is_empty() {
            return Ok(());
        }
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(METADATA_TABLE)?;
            for (id, meta) in entries {
                let json = serde_json::to_string(meta)?;
                table.insert(id.as_str(), json.as_str())?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn get(
        &self,
        id: &str,
    ) -> Result<Option<VectorMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(METADATA_TABLE)?;
        if let Some(value) = table.get(id)? {
            let meta: VectorMetadata = serde_json::from_str(value.value())?;
            Ok(Some(meta))
        } else {
            Ok(None)
        }
    }

    pub fn get_many(
        &self,
        ids: &[String],
    ) -> Result<HashMap<String, VectorMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        let mut out = HashMap::with_capacity(ids.len());
        if ids.is_empty() {
            return Ok(out);
        }
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(METADATA_TABLE)?;
        for id in ids {
            if let Some(value) = table.get(id.as_str())? {
                let meta: VectorMetadata = serde_json::from_str(value.value())?;
                out.insert(id.clone(), meta);
            }
        }
        Ok(out)
    }

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
