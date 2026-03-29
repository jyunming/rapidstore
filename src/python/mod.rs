#![allow(unsafe_op_in_unsafe_fn)]
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::storage::engine::{BatchWriteItem, DistanceMetric, TurboQuantEngine};

/// Thread-safe Python-accessible database handle.
/// RwLock allows concurrent reads (search) with exclusive writes (insert).
/// GIL is released during heavy Rust computation via py.allow_threads().
#[pyclass]
pub struct Database {
    engine: Arc<RwLock<TurboQuantEngine>>,
}

#[pyclass]
pub struct Client {
    uri: String,
    local_dir: String,
    dimension: usize,
    bits: usize,
    seed: u64,
    metric: DistanceMetric,
}

#[pymethods]
impl Database {
    /// Open or create a RapidStore at the given path.
    /// `uri` can be a local directory path: "/data/my_db" or cloud URI
    /// `local_dir` is used for caching remote data and storing local artifacts.
    #[staticmethod]
    #[pyo3(signature = (uri, dimension, bits, seed=42, local_dir=None, metric="ip"))]
    fn open(
        uri: &str,
        dimension: usize,
        bits: usize,
        seed: u64,
        local_dir: Option<&str>,
        metric: &str,
    ) -> PyResult<Self> {
        let actual_local = local_dir.unwrap_or(uri);
        let metric = parse_metric(metric)?;
        let engine =
            TurboQuantEngine::open_with_metric(uri, actual_local, dimension, bits, seed, metric)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            engine: Arc::new(RwLock::new(engine)),
        })
    }

    /// Insert a single vector with optional metadata and document.
    #[pyo3(signature = (id, vector, metadata=None, document=None))]
    fn insert(
        &self,
        py: Python<'_>,
        id: String,
        vector: PyReadonlyArray1<f64>,
        metadata: Option<&Bound<'_, PyDict>>,
        document: Option<String>,
    ) -> PyResult<()> {
        let vec = vector.as_array().to_owned();
        let props = parse_pydict(py, metadata)?;

        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine
                .insert_with_document(id, &vec, props, document)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Batch insert vectors. Lengths of ids/vectors/metadatas/documents must match.
    #[pyo3(signature = (ids, vectors, metadatas=None, documents=None))]
    fn insert_many(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        vectors: Vec<PyReadonlyArray1<f64>>,
        metadatas: Option<Vec<Bound<'_, PyDict>>>,
        documents: Option<Vec<Option<String>>>,
    ) -> PyResult<()> {
        let items = build_batch_items(py, ids, vectors, metadatas, documents)?;
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine
                .insert_many(items)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Batch insert with per-item failure reporting (continues on errors).
    #[pyo3(signature = (ids, vectors, metadatas=None, documents=None))]
    fn insert_many_report(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        vectors: Vec<PyReadonlyArray1<f64>>,
        metadatas: Option<Vec<Bound<'_, PyDict>>>,
        documents: Option<Vec<Option<String>>>,
    ) -> PyResult<PyObject> {
        let items = build_batch_items(py, ids, vectors, metadatas, documents)?;
        let report = py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine.insert_many_report(items)
        });
        batch_report_to_py(py, report)
    }

    /// Collection-first alias: add many embeddings/documents by IDs.
    #[pyo3(signature = (ids, embeddings, metadatas=None, documents=None))]
    fn add(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        embeddings: Vec<PyReadonlyArray1<f64>>,
        metadatas: Option<Vec<Bound<'_, PyDict>>>,
        documents: Option<Vec<Option<String>>>,
    ) -> PyResult<()> {
        let items = build_batch_items(py, ids, embeddings, metadatas, documents)?;
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine
                .insert_many(items)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Upsert a single vector with optional metadata and document.
    #[pyo3(signature = (id, vector, metadata=None, document=None))]
    fn upsert(
        &self,
        py: Python<'_>,
        id: String,
        vector: PyReadonlyArray1<f64>,
        metadata: Option<&Bound<'_, PyDict>>,
        document: Option<String>,
    ) -> PyResult<()> {
        let vec = vector.as_array().to_owned();
        let props = parse_pydict(py, metadata)?;

        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine
                .upsert_with_document(id, &vec, props, document)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Batch upsert vectors. Lengths of ids/vectors/metadatas/documents must match.
    #[pyo3(signature = (ids, vectors, metadatas=None, documents=None))]
    fn upsert_many(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        vectors: Vec<PyReadonlyArray1<f64>>,
        metadatas: Option<Vec<Bound<'_, PyDict>>>,
        documents: Option<Vec<Option<String>>>,
    ) -> PyResult<()> {
        let items = build_batch_items(py, ids, vectors, metadatas, documents)?;
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine
                .upsert_many(items)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Batch upsert with per-item failure reporting (continues on errors).
    #[pyo3(signature = (ids, vectors, metadatas=None, documents=None))]
    fn upsert_many_report(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        vectors: Vec<PyReadonlyArray1<f64>>,
        metadatas: Option<Vec<Bound<'_, PyDict>>>,
        documents: Option<Vec<Option<String>>>,
    ) -> PyResult<PyObject> {
        let items = build_batch_items(py, ids, vectors, metadatas, documents)?;
        let report = py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine.upsert_many_report(items)
        });
        batch_report_to_py(py, report)
    }

    /// Update an existing vector, metadata, and optional document.
    #[pyo3(signature = (id, vector, metadata=None, document=None))]
    fn update(
        &self,
        py: Python<'_>,
        id: String,
        vector: PyReadonlyArray1<f64>,
        metadata: Option<&Bound<'_, PyDict>>,
        document: Option<String>,
    ) -> PyResult<()> {
        let vec = vector.as_array().to_owned();
        let props = parse_pydict(py, metadata)?;

        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine
                .update_with_document(id, &vec, props, document)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Batch update vectors. Lengths of ids/vectors/metadatas/documents must match.
    #[pyo3(signature = (ids, vectors, metadatas=None, documents=None))]
    fn update_many(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        vectors: Vec<PyReadonlyArray1<f64>>,
        metadatas: Option<Vec<Bound<'_, PyDict>>>,
        documents: Option<Vec<Option<String>>>,
    ) -> PyResult<()> {
        let items = build_batch_items(py, ids, vectors, metadatas, documents)?;
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine
                .update_many(items)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Batch update with per-item failure reporting (continues on errors).
    #[pyo3(signature = (ids, vectors, metadatas=None, documents=None))]
    fn update_many_report(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        vectors: Vec<PyReadonlyArray1<f64>>,
        metadatas: Option<Vec<Bound<'_, PyDict>>>,
        documents: Option<Vec<Option<String>>>,
    ) -> PyResult<PyObject> {
        let items = build_batch_items(py, ids, vectors, metadatas, documents)?;
        let report = py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine.update_many_report(items)
        });
        batch_report_to_py(py, report)
    }

    /// Delete by ID, returning True when a live record was deleted.
    fn delete(&self, py: Python<'_>, id: String) -> PyResult<bool> {
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine
                .delete(id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Fetch one record by ID.
    /// `include` supports: ids, metadatas, documents.
    #[pyo3(signature = (id, include_document=true, include=None))]
    fn get(
        &self,
        py: Python<'_>,
        id: String,
        include_document: bool,
        include: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let result = py.allow_threads(|| {
            let engine = self.engine.read().unwrap();
            engine
                .get(&id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        let include_set = parse_include_set(include, &["ids", "metadatas", "documents"]);
        let include_documents = include_document && include_set.contains("documents");

        if let Some(r) = result {
            let dict = PyDict::new_bound(py);
            if include_set.contains("ids") {
                dict.set_item("id", r.id)?;
            }
            if include_set.contains("metadatas") {
                let meta_dict = PyDict::new_bound(py);
                for (k, v) in &r.metadata {
                    meta_dict.set_item(k, json_to_py(py, v)?)?;
                }
                dict.set_item("metadata", meta_dict)?;
            }
            if include_documents {
                if let Some(doc) = r.document {
                    dict.set_item("document", doc)?;
                } else {
                    dict.set_item("document", py.None())?;
                }
            }
            Ok(dict.into())
        } else {
            Ok(py.None())
        }
    }

    /// Fetch multiple IDs and return found records.
    /// Deterministic pagination is applied over the input ID order.
    /// `include` supports: ids, metadatas, documents.
    #[pyo3(signature = (ids, include_document=true, offset=0, limit=None, include=None))]
    fn get_many(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        include_document: bool,
        offset: usize,
        limit: Option<usize>,
        include: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let mut result = py.allow_threads(|| {
            let engine = self.engine.read().unwrap();
            engine
                .get_many(&ids)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        if offset > 0 {
            result = result.into_iter().skip(offset).collect();
        }
        if let Some(lim) = limit {
            result.truncate(lim);
        }
        let include_set = parse_include_set(include, &["ids", "metadatas", "documents"]);
        let include_documents = include_document && include_set.contains("documents");

        let py_list = PyList::empty_bound(py);
        for r in result {
            let dict = PyDict::new_bound(py);
            if include_set.contains("ids") {
                dict.set_item("id", r.id)?;
            }
            if include_set.contains("metadatas") {
                let meta_dict = PyDict::new_bound(py);
                for (k, v) in &r.metadata {
                    meta_dict.set_item(k, json_to_py(py, v)?)?;
                }
                dict.set_item("metadata", meta_dict)?;
            }
            if include_documents {
                if let Some(doc) = r.document {
                    dict.set_item("document", doc)?;
                } else {
                    dict.set_item("document", py.None())?;
                }
            }
            py_list.append(dict)?;
        }
        Ok(py_list.into())
    }

    /// Delete multiple IDs, returns number of deleted live records.
    fn delete_many(&self, py: Python<'_>, ids: Vec<String>) -> PyResult<usize> {
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine
                .delete_many(&ids)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Create a Vamana ANN index for fast retrieval.
    #[pyo3(signature = (max_degree=64, search_list_size=100, alpha=1.2))]
    fn create_index(
        &self,
        py: Python<'_>,
        max_degree: usize,
        search_list_size: usize,
        alpha: f64,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine
                .create_index_with_params(max_degree, search_list_size, alpha)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Search for the top_k most similar vectors. Returns a list of dicts.
    #[pyo3(signature = (query, top_k, where_filter=None, include_document=true, ann_search_list_size=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: PyReadonlyArray1<f64>,
        top_k: usize,
        where_filter: Option<&Bound<'_, PyDict>>,
        include_document: bool,
        ann_search_list_size: Option<usize>,
    ) -> PyResult<PyObject> {
        let q = query.as_array().to_owned();
        let where_props = parse_pydict(py, where_filter)?;

        let results = py.allow_threads(|| {
            let engine = self.engine.read().unwrap();
            let filter_ref = if where_props.is_empty() {
                None
            } else {
                Some(&where_props)
            };
            engine
                .search_with_filter_and_ann(&q, top_k, filter_ref, ann_search_list_size)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;

        let py_list = PyList::empty_bound(py);
        for r in results {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", &r.id)?;
            dict.set_item("score", r.score)?;
            let meta_dict = PyDict::new_bound(py);
            for (k, v) in &r.metadata {
                meta_dict.set_item(k, json_to_py(py, v)?)?;
            }
            dict.set_item("metadata", meta_dict)?;
            if include_document {
                if let Some(doc) = r.document {
                    dict.set_item("document", doc)?;
                } else {
                    dict.set_item("document", py.None())?;
                }
            }
            py_list.append(dict)?;
        }
        Ok(py_list.into())
    }

    /// Hybrid search with dense vector similarity + sparse BM25-like text relevance.
    #[pyo3(signature = (query, query_text, top_k, where_filter=None, dense_weight=0.7, sparse_weight=0.3, include_document=true))]
    fn search_hybrid(
        &self,
        py: Python<'_>,
        query: PyReadonlyArray1<f64>,
        query_text: String,
        top_k: usize,
        where_filter: Option<&Bound<'_, PyDict>>,
        dense_weight: f64,
        sparse_weight: f64,
        include_document: bool,
    ) -> PyResult<PyObject> {
        let q = query.as_array().to_owned();
        let where_props = parse_pydict(py, where_filter)?;

        let results = py.allow_threads(|| {
            let engine = self.engine.read().unwrap();
            let filter_ref = if where_props.is_empty() {
                None
            } else {
                Some(&where_props)
            };
            engine
                .search_hybrid_with_filter(
                    &q,
                    &query_text,
                    top_k,
                    filter_ref,
                    dense_weight,
                    sparse_weight,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;

        let py_list = PyList::empty_bound(py);
        for r in results {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", &r.id)?;
            dict.set_item("score", r.score)?;
            let meta_dict = PyDict::new_bound(py);
            for (k, v) in &r.metadata {
                meta_dict.set_item(k, json_to_py(py, v)?)?;
            }
            dict.set_item("metadata", meta_dict)?;
            if include_document {
                if let Some(doc) = r.document {
                    dict.set_item("document", doc)?;
                } else {
                    dict.set_item("document", py.None())?;
                }
            }
            py_list.append(dict)?;
        }
        Ok(py_list.into())
    }

    /// Dense search followed by user-provided reranker callback on top-N candidates.
    /// Callback receives a result dict and must return a numeric score.
    #[pyo3(signature = (query, top_k, reranker, rerank_top_n=None, where_filter=None, include_document=true, ann_search_list_size=None))]
    fn search_rerank(
        &self,
        py: Python<'_>,
        query: PyReadonlyArray1<f64>,
        top_k: usize,
        reranker: &Bound<'_, PyAny>,
        rerank_top_n: Option<usize>,
        where_filter: Option<&Bound<'_, PyDict>>,
        include_document: bool,
        ann_search_list_size: Option<usize>,
    ) -> PyResult<PyObject> {
        let q = query.as_array().to_owned();
        let where_props = parse_pydict(py, where_filter)?;
        let n = rerank_top_n.unwrap_or(top_k.max(10));

        let mut results = {
            let engine = self.engine.read().unwrap();
            let filter_ref = if where_props.is_empty() {
                None
            } else {
                Some(&where_props)
            };
            engine
                .search_with_filter_and_ann(&q, n, filter_ref, ann_search_list_size)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        };

        for r in &mut results {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", &r.id)?;
            dict.set_item("score", r.score)?;
            let meta_dict = PyDict::new_bound(py);
            for (k, v) in &r.metadata {
                meta_dict.set_item(k, json_to_py(py, v)?)?;
            }
            dict.set_item("metadata", meta_dict)?;
            if include_document {
                if let Some(doc) = &r.document {
                    dict.set_item("document", doc)?;
                } else {
                    dict.set_item("document", py.None())?;
                }
            }
            let out = reranker.call1((dict,))?;
            let new_score: f64 = out.extract()?;
            r.score = new_score;
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        results.truncate(top_k);

        let py_list = PyList::empty_bound(py);
        for r in results {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", &r.id)?;
            dict.set_item("score", r.score)?;
            let meta_dict = PyDict::new_bound(py);
            for (k, v) in &r.metadata {
                meta_dict.set_item(k, json_to_py(py, v)?)?;
            }
            dict.set_item("metadata", meta_dict)?;
            if include_document {
                if let Some(doc) = r.document {
                    dict.set_item("document", doc)?;
                } else {
                    dict.set_item("document", py.None())?;
                }
            }
            py_list.append(dict)?;
        }

        Ok(py_list.into())
    }

    /// Collection-first query API.
    /// Returns a dict with list-per-query fields, similar to common vector DB clients.
    #[pyo3(signature = (query_embeddings, n_results=10, where_filter=None, include=None, offset=0, ann_search_list_size=None))]
    fn query(
        &self,
        py: Python<'_>,
        query_embeddings: Vec<PyReadonlyArray1<f64>>,
        n_results: usize,
        where_filter: Option<&Bound<'_, PyDict>>,
        include: Option<Vec<String>>,
        offset: usize,
        ann_search_list_size: Option<usize>,
    ) -> PyResult<PyObject> {
        let queries: Vec<_> = query_embeddings
            .into_iter()
            .map(|q| q.as_array().to_owned())
            .collect();
        let where_props = parse_pydict(py, where_filter)?;
        let include_set = parse_include_set(include, &["ids", "scores", "metadatas", "documents"]);
        let candidate_n = n_results.saturating_add(offset);

        let results_per_query = py.allow_threads(|| {
            let engine = self.engine.read().unwrap();
            let mut out = Vec::with_capacity(queries.len());
            for q in &queries {
                let hits = if where_props.is_empty() {
                    engine.search_with_filter_and_ann(q, candidate_n, None, ann_search_list_size)
                } else {
                    engine.search_with_filter_and_ann(
                        q,
                        candidate_n,
                        Some(&where_props),
                        ann_search_list_size,
                    )
                }
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                let mut page: Vec<_> = hits.into_iter().skip(offset).collect();
                page.truncate(n_results);
                out.push(page);
            }
            Ok::<_, PyErr>(out)
        })?;

        let out = PyDict::new_bound(py);
        if include_set.contains("ids") {
            let ids_outer = PyList::empty_bound(py);
            for hits in &results_per_query {
                let ids = PyList::empty_bound(py);
                for h in hits {
                    ids.append(&h.id)?;
                }
                ids_outer.append(ids)?;
            }
            out.set_item("ids", ids_outer)?;
        }
        if include_set.contains("scores") {
            let scores_outer = PyList::empty_bound(py);
            for hits in &results_per_query {
                let scores = PyList::empty_bound(py);
                for h in hits {
                    scores.append(h.score)?;
                }
                scores_outer.append(scores)?;
            }
            out.set_item("scores", scores_outer)?;
        }
        if include_set.contains("metadatas") {
            let metas_outer = PyList::empty_bound(py);
            for hits in &results_per_query {
                let metas = PyList::empty_bound(py);
                for h in hits {
                    let d = PyDict::new_bound(py);
                    for (k, v) in &h.metadata {
                        d.set_item(k, json_to_py(py, v)?)?;
                    }
                    metas.append(d)?;
                }
                metas_outer.append(metas)?;
            }
            out.set_item("metadatas", metas_outer)?;
        }
        if include_set.contains("documents") {
            let docs_outer = PyList::empty_bound(py);
            for hits in &results_per_query {
                let docs = PyList::empty_bound(py);
                for h in hits {
                    if let Some(doc) = &h.document {
                        docs.append(doc)?;
                    } else {
                        docs.append(py.None())?;
                    }
                }
                docs_outer.append(docs)?;
            }
            out.set_item("documents", docs_outer)?;
        }

        Ok(out.into())
    }

    /// Flush all buffered WAL entries to disk segments.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine
                .flush_wal_to_segment()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Close the database, flushing remaining buffers.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine
                .close()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Consolidate segments for better performance.
    fn compact(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine
                .compact()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Return database statistics as a dict.
    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = {
            let engine = self.engine.read().unwrap();
            engine.stats()
        };
        let dict = PyDict::new_bound(py);
        dict.set_item("vector_count", stats.vector_count)?;
        dict.set_item("physical_record_count", stats.physical_record_count)?;
        dict.set_item("deleted_record_count", stats.deleted_record_count)?;
        dict.set_item("segment_count", stats.segment_count)?;
        dict.set_item("buffered_vectors", stats.buffered_vectors)?;
        dict.set_item("dimension", stats.d)?;
        dict.set_item("bits", stats.b)?;
        dict.set_item("total_disk_bytes", stats.total_disk_bytes)?;
        match stats.segment_min_bytes {
            Some(v) => dict.set_item("segment_min_bytes", v)?,
            None => dict.set_item("segment_min_bytes", py.None())?,
        }
        match stats.segment_max_bytes {
            Some(v) => dict.set_item("segment_max_bytes", v)?,
            None => dict.set_item("segment_max_bytes", py.None())?,
        }
        match stats.segment_avg_bytes {
            Some(v) => dict.set_item("segment_avg_bytes", v)?,
            None => dict.set_item("segment_avg_bytes", py.None())?,
        }
        match stats.segment_skew_ratio {
            Some(v) => dict.set_item("segment_skew_ratio", v)?,
            None => dict.set_item("segment_skew_ratio", py.None())?,
        }
        dict.set_item("has_index", stats.has_index)?;
        dict.set_item("index_nodes", stats.index_nodes)?;
        dict.set_item("compaction_runs", stats.compaction_runs)?;
        dict.set_item("compaction_recovery_runs", stats.compaction_recovery_runs)?;
        dict.set_item("last_reclaimed_segments", stats.last_reclaimed_segments)?;
        if let Some(v) = stats.index_search_list_size {
            dict.set_item("index_search_list_size", v)?;
        } else {
            dict.set_item("index_search_list_size", py.None())?;
        }
        if let Some(v) = stats.index_alpha {
            dict.set_item("index_alpha", v)?;
        } else {
            dict.set_item("index_alpha", py.None())?;
        }
        Ok(dict.into())
    }

    fn __len__(&self) -> PyResult<usize> {
        let count = self.engine.read().unwrap().vector_count();
        Ok(count as usize)
    }

    fn __repr__(&self) -> PyResult<String> {
        let engine = self.engine.read().unwrap();
        Ok(format!(
            "RapidStore(d={}, b={}, vectors={})",
            engine.d,
            engine.b,
            engine.vector_count()
        ))
    }
}

#[pymethods]
impl Client {
    #[new]
    #[pyo3(signature = (uri, dimension, bits, seed=42, local_dir=None, metric="ip"))]
    fn new(
        uri: String,
        dimension: usize,
        bits: usize,
        seed: u64,
        local_dir: Option<String>,
        metric: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            local_dir: local_dir.unwrap_or_else(|| uri.clone()),
            uri,
            dimension,
            bits,
            seed,
            metric: parse_metric(metric)?,
        })
    }

    fn create_collection(&self, name: String) -> PyResult<Database> {
        TurboQuantEngine::create_collection_with_uri(&self.uri, &self.local_dir, &name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        self.open_collection(name)
    }

    fn get_collection(&self, name: String) -> PyResult<Option<Database>> {
        let exists = TurboQuantEngine::get_collection_with_uri(&self.uri, &self.local_dir, &name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .is_some();
        if !exists {
            Ok(None)
        } else {
            self.open_collection(name).map(Some)
        }
    }

    fn get_or_create_collection(&self, name: String) -> PyResult<Database> {
        TurboQuantEngine::create_collection_with_uri(&self.uri, &self.local_dir, &name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        self.open_collection(name)
    }

    fn list_collections(&self) -> PyResult<Vec<String>> {
        TurboQuantEngine::list_collections(&self.local_dir)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn delete_collection(&self, name: String) -> PyResult<bool> {
        TurboQuantEngine::delete_collection_with_uri(&self.uri, &self.local_dir, &name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn snapshot_collection(&self, name: String, snapshot_dir: String) -> PyResult<()> {
        TurboQuantEngine::snapshot_collection(&self.local_dir, &name, &snapshot_dir)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn restore_collection(&self, name: String, snapshot_dir: String) -> PyResult<()> {
        TurboQuantEngine::restore_collection(&self.local_dir, &name, &snapshot_dir)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "RapidStoreClient(uri={}, d={}, b={})",
            self.uri, self.dimension, self.bits
        ))
    }
}

/// Parse an optional Python dict into HashMap<String, JsonValue>
fn parse_pydict(
    py: Python<'_>,
    dict: Option<&Bound<'_, PyDict>>,
) -> PyResult<HashMap<String, JsonValue>> {
    let mut props = HashMap::new();
    if let Some(d) = dict {
        for (k, v) in d.iter() {
            let key: String = k.extract()?;
            let val = py_to_json(py, &v)?;
            props.insert(key, val);
        }
    }
    Ok(props)
}

fn parse_include_set(include: Option<Vec<String>>, defaults: &[&str]) -> std::collections::HashSet<String> {
    include
        .unwrap_or_else(|| defaults.iter().map(|s| s.to_string()).collect())
        .into_iter()
        .map(|s| s.to_ascii_lowercase())
        .collect()
}

/// Convert a Python object to a serde_json Value (best-effort)
fn py_to_json(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    // Order matters: bool before i64 since Python bool is a subtype of int
    if let Ok(v) = obj.extract::<bool>() {
        return Ok(JsonValue::Bool(v));
    }
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(JsonValue::Number(v.into()));
    }
    if let Ok(v) = obj.extract::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(v) {
            return Ok(JsonValue::Number(n));
        }
    }
    if let Ok(v) = obj.extract::<String>() {
        return Ok(JsonValue::String(v));
    }
    if let Ok(v) = obj.downcast::<PyString>() {
        return Ok(JsonValue::String(v.to_string()));
    }
    if let Ok(v) = obj.downcast::<PyDict>() {
        let mut out = serde_json::Map::new();
        for (k, val) in v.iter() {
            let key: String = k.extract()?;
            out.insert(key, py_to_json(py, &val)?);
        }
        return Ok(JsonValue::Object(out));
    }
    if let Ok(v) = obj.downcast::<PyList>() {
        let mut out = Vec::with_capacity(v.len());
        for item in v.iter() {
            out.push(py_to_json(py, &item)?);
        }
        return Ok(JsonValue::Array(out));
    }
    // Fallback: store string representation
    Ok(JsonValue::String(obj.str()?.to_string()))
}

/// Convert a serde_json Value to a Python object (pyo3 0.21 compatible)
fn json_to_py(py: Python<'_>, val: &JsonValue) -> PyResult<PyObject> {
    Ok(match val {
        JsonValue::Null => py.None(),
        JsonValue::Bool(b) => (*b).to_object(py),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_object(py)
            } else if let Some(f) = n.as_f64() {
                f.to_object(py)
            } else {
                n.to_string().to_object(py)
            }
        }
        JsonValue::String(s) => s.to_object(py),
        JsonValue::Array(arr) => {
            let list = PyList::empty_bound(py);
            for v in arr {
                list.append(json_to_py(py, v)?)?;
            }
            list.into()
        }
        JsonValue::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            dict.into()
        }
    })
}

/// Register the Database class with the PyO3 module.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Database>()?;
    m.add_class::<Client>()?;
    Ok(())
}

impl Client {
    fn open_collection(&self, name: String) -> PyResult<Database> {
        let engine = TurboQuantEngine::open_collection(
            &self.uri,
            &self.local_dir,
            &name,
            self.dimension,
            self.bits,
            self.seed,
            self.metric.clone(),
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Database {
            engine: Arc::new(RwLock::new(engine)),
        })
    }
}

fn build_batch_items(
    py: Python<'_>,
    ids: Vec<String>,
    vectors: Vec<PyReadonlyArray1<f64>>,
    metadatas: Option<Vec<Bound<'_, PyDict>>>,
    documents: Option<Vec<Option<String>>>,
) -> PyResult<Vec<BatchWriteItem>> {
    if ids.len() != vectors.len() {
        return Err(PyValueError::new_err(
            "ids and vectors must have the same length",
        ));
    }
    let metas_len = metadatas.as_ref().map(|m| m.len()).unwrap_or(ids.len());
    if metas_len != ids.len() {
        return Err(PyValueError::new_err(
            "metadatas length must match ids length",
        ));
    }
    let docs_len = documents.as_ref().map(|d| d.len()).unwrap_or(ids.len());
    if docs_len != ids.len() {
        return Err(PyValueError::new_err(
            "documents length must match ids length",
        ));
    }

    let mut items = Vec::with_capacity(ids.len());
    for (i, (id, vector)) in ids.into_iter().zip(vectors.into_iter()).enumerate() {
        let metadata = if let Some(ref metas) = metadatas {
            parse_pydict(py, Some(&metas[i]))?
        } else {
            HashMap::new()
        };
        let document = if let Some(ref docs) = documents {
            docs[i].clone()
        } else {
            None
        };
        items.push(BatchWriteItem {
            id,
            vector: vector.as_array().to_owned(),
            metadata,
            document,
        });
    }
    Ok(items)
}

fn batch_report_to_py(
    py: Python<'_>,
    report: crate::storage::engine::BatchWriteReport,
) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    out.set_item("applied", report.applied)?;
    let failed = PyList::empty_bound(py);
    for f in report.failed {
        let d = PyDict::new_bound(py);
        d.set_item("index", f.index)?;
        d.set_item("id", f.id)?;
        d.set_item("error", f.error)?;
        failed.append(d)?;
    }
    out.set_item("failed", failed)?;
    Ok(out.into())
}

fn parse_metric(metric: &str) -> PyResult<DistanceMetric> {
    match metric.to_ascii_lowercase().as_str() {
        "ip" | "dot" | "inner_product" => Ok(DistanceMetric::Ip),
        "cosine" => Ok(DistanceMetric::Cosine),
        "l2" | "euclidean" => Ok(DistanceMetric::L2),
        _ => Err(PyValueError::new_err(
            "metric must be one of: ip, cosine, l2",
        )),
    }
}

