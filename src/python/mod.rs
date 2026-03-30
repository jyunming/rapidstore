#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unsafe_op_in_unsafe_fn)]
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::storage::engine::{
    BatchWriteItem, DistanceMetric, GetResult, SearchResult, TurboQuantEngine,
};

#[pyclass]
pub struct Database {
    engine: Arc<RwLock<TurboQuantEngine>>,
}

#[pymethods]
impl Database {
    #[staticmethod]
    #[pyo3(signature = (uri, dimension, bits, seed=42, local_dir=None, metric="ip"))]
    fn open(
        uri: &str,
        dimension: usize,
        bits: usize,
        seed: u64,
        local_dir: Option<String>,
        metric: &str,
    ) -> PyResult<Self> {
        let metric = parse_metric(metric)?;
        let local = local_dir.unwrap_or_else(|| uri.to_string());
        let engine = TurboQuantEngine::open_with_metric(uri, &local, dimension, bits, seed, metric)
            .map_err(to_py_runtime)?;
        Ok(Self {
            engine: Arc::new(RwLock::new(engine)),
        })
    }

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
                .map_err(to_py_runtime)
        })
    }

    #[pyo3(signature = (ids, vectors, metadatas=None, documents=None, mode="insert"))]
    fn insert_many(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        vectors: Vec<PyReadonlyArray1<f64>>,
        metadatas: Option<&Bound<'_, PyAny>>,
        documents: Option<&Bound<'_, PyAny>>,
        mode: &str,
    ) -> PyResult<()> {
        if ids.len() != vectors.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ids and vectors length mismatch",
            ));
        }

        let metas = parse_metadata_rows(py, metadatas, ids.len())?;
        let docs = parse_document_rows(documents, ids.len())?;

        let mut items = Vec::with_capacity(ids.len());
        for i in 0..ids.len() {
            items.push(BatchWriteItem {
                id: ids[i].clone(),
                vector: vectors[i].as_array().to_owned(),
                metadata: metas[i].clone(),
                document: docs[i].clone(),
            });
        }

        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            run_batch_write(&mut engine, items, mode)
        })
    }

    #[pyo3(signature = (ids, vectors, metadatas=None, documents=None, mode="insert"))]
    fn insert_batch(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        vectors: PyReadonlyArray2<f64>,
        metadatas: Option<&Bound<'_, PyAny>>,
        documents: Option<&Bound<'_, PyAny>>,
        mode: &str,
    ) -> PyResult<()> {
        let matrix = vectors.as_array();
        if ids.len() != matrix.nrows() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ids and vectors row count mismatch",
            ));
        }

        let metas = parse_metadata_rows(py, metadatas, ids.len())?;
        let docs = parse_document_rows(documents, ids.len())?;

        // We process in smaller chunks even here to avoid a giant BatchWriteItem vector
        let chunk_size = 5000;
        let mut start = 0;
        while start < ids.len() {
            let end = (start + chunk_size).min(ids.len());
            let mut items = Vec::with_capacity(end - start);
            for i in start..end {
                items.push(BatchWriteItem {
                    id: ids[i].clone(),
                    vector: matrix.row(i).to_owned(),
                    metadata: metas[i].clone(),
                    document: docs[i].clone(),
                });
            }

            py.allow_threads(|| {
                let mut engine = self.engine.write().unwrap();
                run_batch_write(&mut engine, items, mode)
            })?;
            start = end;
        }

        Ok(())
    }

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
                .map_err(to_py_runtime)
        })
    }

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

    #[pyo3(signature = (query, top_k, filter=None, use_ann=true, ann_search_list_size=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: PyReadonlyArray1<f64>,
        top_k: usize,
        filter: Option<&Bound<'_, PyDict>>,
        use_ann: bool,
        ann_search_list_size: Option<usize>,
    ) -> PyResult<PyObject> {
        let q = query.as_array().to_owned();
        let parsed_filter = parse_pydict(py, filter)?;
        let filter_ref = if parsed_filter.is_empty() {
            None
        } else {
            Some(&parsed_filter)
        };

        let results = py.allow_threads(|| {
            let engine = self.engine.read().unwrap();
            if use_ann {
                engine.search_with_filter_and_ann(&q, top_k, filter_ref, ann_search_list_size)
            } else {
                engine.search_with_filter(&q, top_k, filter_ref)
            }
            .map_err(to_py_runtime)
        })?;

        let py_list = PyList::empty_bound(py);
        for r in results {
            py_list.append(search_result_to_py(py, &r)?)?;
        }
        Ok(py_list.into())
    }

    #[pyo3(signature = (max_degree=32, search_list_size=32, alpha=1.2))]
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
                .map_err(to_py_runtime)
        })
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
        dict.set_item("live_vectors_bytes_estimate", stats.live_vectors_bytes_estimate)?;
        dict.set_item("metadata_entries", stats.metadata_entries)?;
        dict.set_item("metadata_bytes_estimate", stats.metadata_bytes_estimate)?;
        dict.set_item("ann_slot_count", stats.ann_slot_count)?;
        dict.set_item("graph_nodes", stats.graph_nodes)?;
        Ok(dict.into())
    }

    fn __len__(&self) -> PyResult<usize> {
        let count = self.engine.read().unwrap().vector_count();
        Ok(count as usize)
    }

    fn __repr__(&self) -> PyResult<String> {
        let engine = self.engine.read().unwrap();
        Ok(format!(
            "TurboQuantDB(d={}, b={}, vectors={})",
            engine.d,
            engine.b,
            engine.vector_count()
        ))
    }
}

fn run_batch_write(
    engine: &mut TurboQuantEngine,
    items: Vec<BatchWriteItem>,
    mode: &str,
) -> PyResult<()> {
    match mode.to_ascii_lowercase().as_str() {
        "insert" => engine.insert_many(items).map_err(to_py_runtime),
        "upsert" => engine.upsert_many(items).map_err(to_py_runtime),
        "update" => engine.update_many(items).map_err(to_py_runtime),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported mode '{}' (expected insert/upsert/update)",
            other
        ))),
    }
}

fn parse_metric(metric: &str) -> PyResult<DistanceMetric> {
    match metric.to_ascii_lowercase().as_str() {
        "ip" | "inner_product" | "inner-product" => Ok(DistanceMetric::Ip),
        "cosine" => Ok(DistanceMetric::Cosine),
        "l2" | "euclidean" => Ok(DistanceMetric::L2),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported metric '{}' (expected ip/cosine/l2)",
            other
        ))),
    }
}

fn parse_metadata_rows(
    py: Python<'_>,
    rows: Option<&Bound<'_, PyAny>>,
    expected: usize,
) -> PyResult<Vec<HashMap<String, JsonValue>>> {
    let mut out = Vec::with_capacity(expected);
    let Some(rows_any) = rows else {
        return Ok(vec![HashMap::new(); expected]);
    };
    let list = rows_any.downcast::<PyList>()?;
    if list.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "metadatas length mismatch",
        ));
    }
    for item in list.iter() {
        if item.is_none() {
            out.push(HashMap::new());
            continue;
        }
        let d = item.downcast::<PyDict>()?;
        out.push(parse_pydict(py, Some(d))?);
    }
    Ok(out)
}

fn parse_document_rows(
    rows: Option<&Bound<'_, PyAny>>,
    expected: usize,
) -> PyResult<Vec<Option<String>>> {
    let mut out = Vec::with_capacity(expected);
    let Some(rows_any) = rows else {
        return Ok(vec![None; expected]);
    };
    let list = rows_any.downcast::<PyList>()?;
    if list.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "documents length mismatch",
        ));
    }
    for item in list.iter() {
        if item.is_none() {
            out.push(None);
        } else {
            out.push(Some(item.extract::<String>()?));
        }
    }
    Ok(out)
}

fn search_result_to_py(py: Python<'_>, r: &SearchResult) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", &r.id)?;
    dict.set_item("score", r.score)?;
    let meta_dict = PyDict::new_bound(py);
    for (k, v) in &r.metadata {
        meta_dict.set_item(k, json_to_py(py, v)?)?;
    }
    dict.set_item("metadata", meta_dict)?;
    match &r.document {
        Some(doc) => dict.set_item("document", doc)?,
        None => dict.set_item("document", py.None())?,
    }
    Ok(dict.into())
}

fn get_result_to_py(py: Python<'_>, r: &GetResult) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", &r.id)?;
    let meta_dict = PyDict::new_bound(py);
    for (k, v) in &r.metadata {
        meta_dict.set_item(k, json_to_py(py, v)?)?;
    }
    dict.set_item("metadata", meta_dict)?;
    match &r.document {
        Some(doc) => dict.set_item("document", doc)?,
        None => dict.set_item("document", py.None())?,
    }
    Ok(dict.into())
}

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

fn py_to_json(_py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
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
    Ok(JsonValue::String(obj.str()?.to_string()))
}

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

fn to_py_runtime(e: Box<dyn std::error::Error + Send + Sync>) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Database>()?;
    let db_cls = m.getattr("Database")?;
    m.add("TurboQuantDB", db_cls)?;
    Ok(())
}






