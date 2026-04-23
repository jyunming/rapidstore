//! Storage layer for TurboQuantDB.
//!
//! | Module | Responsibility |
//! |--------|---------------|
//! | [`engine`] | Main orchestrator: insert, search, delete, flush, index |
//! | [`wal`] | Write-ahead log for crash recovery (V3 packed binary format) |
//! | [`segment`] | Immutable, append-only segment files flushed from the WAL |
//! | [`live_codes`] | Memory-mapped hot vector cache (`live_codes.bin`, `live_vectors.bin`) |
//! | [`graph`] | HNSW graph index for approximate nearest-neighbour search |
//! | [`id_pool`] | Bidirectional ID ↔ slot mapping (FNV-1a hash table) |
//! | [`metadata`] | Per-vector metadata (JSON) and document string storage |
//! | [`compaction`] | Segment merging to reclaim space from deleted vectors |
//! | [`backend`] | `StorageProvider` trait: local filesystem, extensible to cloud |

pub mod backend;
pub mod compaction;
pub mod engine;
pub mod graph;
pub mod id_pool;
pub mod ivf;
pub mod live_codes;
pub mod metadata;
pub mod segment;
pub mod wal;
