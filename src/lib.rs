//! # TurboQuantDB
//!
//! A high-performance, embedded vector database written in Rust with Python bindings.
//! Implements the **TurboQuant** algorithm ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)):
//! zero-training-time vector quantization with provably near-optimal distortion and
//! unbiased inner product estimation via Quantized Johnson-Lindenstrauss (QJL) transforms.
//!
//! ## Architecture
//!
//! ```text
//! Python API (PyO3)
//!   └── python::Database  ← thread-safe Arc<RwLock<TurboQuantEngine>>
//!         ├── storage::engine   ← orchestrates insert / search / persist
//!         │     ├── storage::wal        ← crash-safe write-ahead log
//!         │     ├── storage::segment    ← immutable append-only segments
//!         │     ├── storage::live_codes ← mmap'd hot vector cache
//!         │     ├── storage::graph      ← HNSW index (optional)
//!         │     ├── storage::id_pool    ← ID ↔ slot hash table
//!         │     └── storage::metadata   ← per-vector metadata + documents
//!         └── quantizer::ProdQuantizer  ← two-stage MSE + QJL quantizer
//!               ├── quantizer::mse  ← SRHT rotation + Lloyd-Max codebook (exact: QR)
//!               └── quantizer::qjl  ← SRHT projection, 1-bit sign (exact: dense Gaussian)
//! ```
//!
//! ## Data Flow
//!
//! **Write:** `insert_batch()` → quantize (SRHT → MSE centroids + QJL bits) → WAL
//! entry → `live_codes.bin` → periodic flush to immutable segment.
//!
//! **Search (brute-force):** query → precompute MSE lookup table + QJL scale
//! → score all live vectors → top-k.
//!
//! **Search (ANN):** query → HNSW beam search → optional raw-vector rerank → top-k.
//!
//! ## Deployment
//!
//! - **Embedded** (`tqdb` Python package) — runs in-process, no server needed.
//! - **Server** (`server/` workspace) — Axum HTTP service with multi-tenancy, RBAC, quotas.

use pyo3::prelude::*;

pub mod linalg;
pub mod python;
pub mod quantizer;
pub mod storage;

#[pymodule]
fn tqdb(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(py, m)?;
    Ok(())
}
