#![allow(unsafe_op_in_unsafe_fn)]
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::storage::engine::{
    BatchWriteItem, BatchWriteMode, DistanceMetric, GetResult, RerankPrecision, TurboQuantEngine,
};

/// Thread-safe handle to a TurboQuantDB database.
///
/// All operations are safe to call from multiple Python threads simultaneously.
/// Reads are concurrent; writes (insert/delete/index) are serialised by an internal
/// `RwLock`.
///
/// Open or create a database with :meth:`Database.open`.
#[pyclass]
pub struct Database {
    engine: Arc<RwLock<TurboQuantEngine>>,
}


#[pymethods]
impl Database {
    /// Open (or create) a TurboQuantDB database at the given directory path.
    ///
    /// Args:
    ///     path: Directory where database files are stored.
    ///     dimension: Vector dimensionality. Must match on reopen.
    ///     bits: Quantization bits per coordinate — ``4`` (8× compression, good recall)
    ///           or ``8`` (4× compression, better recall). Default ``4``.
    ///     seed: Random seed for the quantizer. Must match on reopen. Default ``42``.
    ///     metric: Distance metric — ``"ip"`` (inner product), ``"cosine"``,
    ///             or ``"l2"`` (Euclidean). Fixed at creation. Default ``"ip"``.
    ///     rerank: Enable reranking of HNSW candidates. Default ``True``.
    ///     fast_mode: Skip QJL residual quantization (~30 % faster ingest, slightly
    ///                lower recall). Default ``False``.
    ///     rerank_precision: Raw-vector reranking precision:
    ///         - ``None`` (default): dequantization reranking — no extra disk/RAM.
    ///         - ``"f16"``: store raw vectors as float16 (+n×d×2 bytes), exact reranking.
    ///         - ``"f32"``: store raw vectors as float32 (+n×d×4 bytes), maximum precision.
    ///
    /// Returns:
    ///     An open :class:`Database` instance.
    ///
    /// Example::
    ///
    ///     db = Database.open("mydb", dimension=1536, bits=4, metric="cosine")
    #[staticmethod]
    #[pyo3(signature = (path, dimension, bits=4, seed=42, metric="ip", rerank=true, fast_mode=false, rerank_precision=None))]
    fn open(
        path: String,
        dimension: usize,
        bits: usize,
        seed: u64,
        metric: &str,
        rerank: bool,
        fast_mode: bool,
        rerank_precision: Option<&str>,
    ) -> PyResult<Self> {
        let dist_metric = match metric.to_lowercase().as_str() {
            "ip" => DistanceMetric::Ip,
            "cosine" => DistanceMetric::Cosine,
            "l2" => DistanceMetric::L2,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid metric: {}",
                    metric
                )));
            }
        };

        let precision = match rerank_precision.map(|s| s.to_lowercase()) {
            None => RerankPrecision::Disabled,
            Some(ref s) if s == "none" || s == "disabled" || s == "dequant" => RerankPrecision::Disabled,
            Some(ref s) if s == "f16" || s == "half" => RerankPrecision::F16,
            Some(ref s) if s == "f32" || s == "float" || s == "full" => RerankPrecision::F32,
            Some(ref s) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid rerank_precision: '{}'. Use 'f16', 'f32', or None (dequant reranking).",
                    s
                )));
            }
        };

        let engine = TurboQuantEngine::open_with_options(
            &path,
            &path,
            dimension,
            bits,
            seed,
            dist_metric,
            rerank,
            fast_mode,
            precision,
        )
        .map_err(to_py_runtime)?;
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
        vector: PyObject,
        metadata: Option<&Bound<'_, PyDict>>,
        document: Option<String>,
    ) -> PyResult<()> {
        let vec = if let Ok(v) = vector.extract::<PyReadonlyArray1<f32>>(py) {
            v.as_array().mapv(|x| x as f64)
        } else {
            vector
                .extract::<PyReadonlyArray1<f64>>(py)?
                .as_array()
                .to_owned()
        };
        let props = parse_pydict(metadata)?;
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine
                .insert_with_document(id, &vec, props, document)
                .map_err(to_py_runtime)
        })
    }

    #[pyo3(signature = (ids, vectors, metadatas=None, documents=None, mode="insert"))]
    fn insert_batch(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        vectors: PyObject,
        metadatas: Option<&Bound<'_, PyAny>>,
        documents: Option<&Bound<'_, PyAny>>,
        mode: &str,
    ) -> PyResult<()> {
        let b_mode = match mode.to_lowercase().as_str() {
            "insert" => BatchWriteMode::Insert,
            "update" => BatchWriteMode::Update,
            "upsert" => BatchWriteMode::Upsert,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid mode")),
        };

        if let Ok(v32) = vectors.extract::<PyReadonlyArray2<f32>>(py) {
            let matrix = v32.as_array();
            if ids.len() != matrix.nrows() {
                return Err(pyo3::exceptions::PyValueError::new_err("mismatch"));
            }
            let metas = parse_metadata_rows(metadatas, ids.len())?;
            let docs = parse_document_rows(documents, ids.len())?;

            let chunk_size = 2000;
            for i in (0..ids.len()).step_by(chunk_size) {
                let end = (i + chunk_size).min(ids.len());
                let mut chunk_items = Vec::with_capacity(end - i);
                for j in i..end {
                    chunk_items.push(BatchWriteItem {
                        id: ids[j].clone(),
                        vector: matrix.row(j).to_vec(),
                        metadata: metas[j].clone(),
                        document: docs[j].clone(),
                    });
                }
                py.allow_threads(|| {
                    let mut engine = self.engine.write().unwrap();
                    engine
                        .insert_many_with_mode(chunk_items, b_mode)
                        .map_err(to_py_runtime)
                })?;
            }
        } else {
            let v64 = vectors.extract::<PyReadonlyArray2<f64>>(py)?;
            let matrix = v64.as_array();
            if ids.len() != matrix.nrows() {
                return Err(pyo3::exceptions::PyValueError::new_err("mismatch"));
            }
            let metas = parse_metadata_rows(metadatas, ids.len())?;
            let docs = parse_document_rows(documents, ids.len())?;

            let chunk_size = 2000;
            for i in (0..ids.len()).step_by(chunk_size) {
                let end = (i + chunk_size).min(ids.len());
                let mut chunk_items = Vec::with_capacity(end - i);
                for j in i..end {
                    chunk_items.push(BatchWriteItem {
                        id: ids[j].clone(),
                        vector: matrix.row(j).iter().map(|&x| x as f32).collect(),
                        metadata: metas[j].clone(),
                        document: docs[j].clone(),
                    });
                }
                py.allow_threads(|| {
                    let mut engine = self.engine.write().unwrap();
                    engine
                        .insert_many_with_mode(chunk_items, b_mode)
                        .map_err(to_py_runtime)
                })?;
            }
        }
        Ok(())
    }

    fn upsert(
        &self,
        py: Python<'_>,
        id: String,
        vector: PyObject,
        metadata: Option<&Bound<'_, PyDict>>,
        document: Option<String>,
    ) -> PyResult<()> {
        let vec = if let Ok(v) = vector.extract::<PyReadonlyArray1<f32>>(py) {
            v.as_array().mapv(|x| x as f64)
        } else {
            vector
                .extract::<PyReadonlyArray1<f64>>(py)?
                .as_array()
                .to_owned()
        };
        let props = parse_pydict(metadata)?;
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine
                .upsert_with_document(id, &vec, props, document)
                .map_err(to_py_runtime)
        })
    }

    fn update(
        &self,
        py: Python<'_>,
        id: String,
        vector: PyObject,
        metadata: Option<&Bound<'_, PyDict>>,
        document: Option<String>,
    ) -> PyResult<()> {
        let vec = if let Ok(v) = vector.extract::<PyReadonlyArray1<f32>>(py) {
            v.as_array().mapv(|x| x as f64)
        } else {
            vector
                .extract::<PyReadonlyArray1<f64>>(py)?
                .as_array()
                .to_owned()
        };
        let props = parse_pydict(metadata)?;
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine
                .update_with_document(id, &vec, props, document)
                .map_err(to_py_runtime)
        })
    }

    fn delete(&self, py: Python<'_>, id: String) -> PyResult<bool> {
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine.delete(id).map_err(to_py_runtime)
        })
    }

    fn get(&self, py: Python<'_>, id: String) -> PyResult<Option<PyObject>> {
        let got = py.allow_threads(|| {
            let engine = self.engine.read().unwrap();
            engine.get(&id).map_err(to_py_runtime)
        })?;
        Ok(match got {
            Some(g) => Some(get_result_to_py(py, &g)?),
            None => None,
        })
    }

    fn get_many(&self, py: Python<'_>, ids: Vec<String>) -> PyResult<PyObject> {
        let results = py.allow_threads(|| {
            let engine = self.engine.read().unwrap();
            engine.get_many(&ids).map_err(to_py_runtime)
        })?;
        let py_list = PyList::empty_bound(py);
        for r in results {
            match r {
                Some(g) => py_list.append(get_result_to_py(py, &g)?)?,
                None => py_list.append(py.None())?,
            }
        }
        Ok(py_list.into())
    }

    fn list_all(&self, _py: Python<'_>) -> PyResult<Vec<String>> {
        let engine = self.engine.read().unwrap();
        Ok(engine.list_all())
    }

    #[pyo3(signature = (query, top_k, filter=None, _use_ann=true, ann_search_list_size=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: PyObject,
        top_k: usize,
        filter: Option<&Bound<'_, PyDict>>,
        _use_ann: bool,
        ann_search_list_size: Option<usize>,
    ) -> PyResult<PyObject> {
        let q = if let Ok(v) = query.extract::<PyReadonlyArray1<f32>>(py) {
            v.as_array().mapv(|x| x as f64)
        } else {
            query
                .extract::<PyReadonlyArray1<f64>>(py)?
                .as_array()
                .to_owned()
        };
        let parsed_filter = parse_pydict(filter)?;
        let filter_ref = if parsed_filter.is_empty() {
            None
        } else {
            Some(&parsed_filter)
        };

        let results = py.allow_threads(|| {
            let engine = self.engine.read().unwrap();
            engine
                .search_with_filter_and_ann(&q, top_k, filter_ref, ann_search_list_size)
                .map_err(to_py_runtime)
        })?;

        let py_list = PyList::empty_bound(py);
        for r in results {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", r.id)?;
            dict.set_item("score", r.score)?;
            let meta_dict = PyDict::new_bound(py);
            for (k, v) in r.metadata {
                meta_dict.set_item(k, json_to_py(py, &v)?)?;
            }
            dict.set_item("metadata", meta_dict)?;
            if let Some(doc) = r.document {
                dict.set_item("document", doc)?;
            }
            py_list.append(dict)?;
        }
        Ok(py_list.into())
    }

    #[pyo3(signature = (max_degree=None, ef_construction=None, search_list_size=None, alpha=None, n_refinements=None))]
    fn create_index(
        &self,
        py: Python<'_>,
        max_degree: Option<usize>,
        ef_construction: Option<usize>,
        search_list_size: Option<usize>,
        alpha: Option<f64>,
        n_refinements: Option<usize>,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine
                .create_index_with_params(
                    max_degree.unwrap_or(32),
                    ef_construction.unwrap_or(200),
                    search_list_size.unwrap_or(128),
                    alpha.unwrap_or(1.2),
                    n_refinements.unwrap_or(5),
                )
                .map_err(to_py_runtime)
        })
    }

    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = {
            let engine = self.engine.read().unwrap();
            engine.stats()
        };
        let dict = PyDict::new_bound(py);
        dict.set_item("vector_count", stats.vector_count)?;
        dict.set_item("segment_count", stats.segment_count)?;
        dict.set_item("buffered_vectors", stats.buffered_vectors)?;
        dict.set_item("dimension", stats.d)?;
        dict.set_item("bits", stats.b)?;
        dict.set_item("total_disk_bytes", stats.total_disk_bytes)?;
        dict.set_item("has_index", stats.has_index)?;
        dict.set_item("index_nodes", stats.index_nodes)?;
        dict.set_item("live_codes_bytes", stats.live_codes_bytes)?;
        dict.set_item("live_slot_count", stats.live_slot_count)?;
        dict.set_item("live_id_count", stats.live_id_count)?;
        dict.set_item("live_vectors_count", stats.live_vectors_count)?;
        dict.set_item(
            "live_vectors_bytes_estimate",
            stats.live_vectors_bytes_estimate,
        )?;
        dict.set_item("metadata_entries", stats.metadata_entries)?;
        dict.set_item("metadata_bytes_estimate", stats.metadata_bytes_estimate)?;
        dict.set_item("ann_slot_count", stats.ann_slot_count)?;
        dict.set_item("graph_nodes", stats.graph_nodes)?;
        // Computed estimate: in-memory footprint across all major buffers.
        let ram_estimate_bytes = stats.live_codes_bytes
            + stats.live_vectors_bytes_estimate
            + stats.metadata_bytes_estimate;
        dict.set_item("ram_estimate_bytes", ram_estimate_bytes)?;
        Ok(dict.into())
    }

    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine.flush_wal_to_segment().map_err(to_py_runtime)
        })
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine.close().map_err(to_py_runtime)
        })
    }
}

fn get_result_to_py(py: Python<'_>, g: &GetResult) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", &g.id)?;
    let meta_dict = PyDict::new_bound(py);
    for (k, v) in &g.metadata {
        meta_dict.set_item(k, json_to_py(py, v)?)?;
    }
    dict.set_item("metadata", meta_dict)?;
    if let Some(doc) = &g.document {
        dict.set_item("document", doc)?;
    }
    Ok(dict.into())
}

fn json_to_py(py: Python<'_>, v: &JsonValue) -> PyResult<PyObject> {
    match v {
        JsonValue::Null => Ok(py.None()),
        JsonValue::Bool(b) => Ok(b.to_object(py)),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(u) = n.as_u64() {
                Ok(u.to_object(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py))
            } else {
                // Extreme values outside f64 range — fall back to string
                Ok(n.to_string().to_object(py))
            }
        }
        JsonValue::String(s) => Ok(s.to_object(py)),
        JsonValue::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into())
        }
        JsonValue::Object(obj) => {
            let dict = PyDict::new_bound(py);
            for (k, val) in obj {
                dict.set_item(k, json_to_py(py, val)?)?;
            }
            Ok(dict.into())
        }
    }
}

fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    if obj.is_none() {
        Ok(JsonValue::Null)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(JsonValue::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(JsonValue::from(i))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(JsonValue::from(f))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(JsonValue::String(s))
    } else if let Ok(list) = obj.downcast::<PyList>() {
        let mut arr = Vec::new();
        for item in list {
            arr.push(py_to_json(&item)?);
        }
        Ok(JsonValue::Array(arr))
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict {
            map.insert(k.extract::<String>()?, py_to_json(&v)?);
        }
        Ok(JsonValue::Object(map))
    } else {
        Ok(JsonValue::Null)
    }
}

fn parse_pydict(dict: Option<&Bound<'_, PyDict>>) -> PyResult<HashMap<String, JsonValue>> {
    let mut map = HashMap::new();
    if let Some(d) = dict {
        for (k, v) in d {
            map.insert(k.extract::<String>()?, py_to_json(&v)?);
        }
    }
    Ok(map)
}

fn parse_metadata_rows(
    metadatas: Option<&Bound<'_, PyAny>>,
    n: usize,
) -> PyResult<Vec<HashMap<String, JsonValue>>> {
    if let Some(m) = metadatas {
        if let Ok(list) = m.downcast::<PyList>() {
            if list.len() != n {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "metadatas length {} does not match ids/vectors length {}",
                    list.len(),
                    n
                )));
            }
            let mut out = Vec::with_capacity(n);
            for item in list {
                let dict = item.downcast::<PyDict>()?;
                out.push(parse_pydict(Some(dict))?);
            }
            Ok(out)
        } else {
            Ok(vec![HashMap::new(); n])
        }
    } else {
        Ok(vec![HashMap::new(); n])
    }
}

fn parse_document_rows(
    documents: Option<&Bound<'_, PyAny>>,
    n: usize,
) -> PyResult<Vec<Option<String>>> {
    if let Some(d) = documents {
        if let Ok(list) = d.downcast::<PyList>() {
            if list.len() != n {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "documents length {} does not match ids/vectors length {}",
                    list.len(),
                    n
                )));
            }
            let mut out = Vec::with_capacity(n);
            for item in list {
                out.push(item.extract::<Option<String>>()?);
            }
            Ok(out)
        } else {
            Ok(vec![None; n])
        }
    } else {
        Ok(vec![None; n])
    }
}

fn to_py_runtime(e: Box<dyn std::error::Error + Send + Sync>) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

#[allow(dead_code)]
fn parse_include_set(include: Option<Vec<String>>, defaults: &[&str]) -> std::collections::HashSet<String> {
    include
        .unwrap_or_else(|| defaults.iter().map(|s| s.to_string()).collect())
        .into_iter()
        .map(|s| s.to_ascii_lowercase())
        .collect()
}

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Database>()?;
    let db_cls = m.getattr("Database")?;
    m.add("TurboQuantDB", db_cls)?;
    Ok(())
}
