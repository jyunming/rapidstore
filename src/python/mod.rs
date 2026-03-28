use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::PyReadonlyArray1;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde_json::Value as JsonValue;

use crate::storage::engine::TurboQuantEngine;

/// Thread-safe Python-accessible database handle.
/// RwLock allows concurrent reads (search) with exclusive writes (insert).
/// GIL is released during heavy Rust computation via py.allow_threads().
#[pyclass]
pub struct Database {
    engine: Arc<RwLock<TurboQuantEngine>>,
}

#[pymethods]
impl Database {
    /// Open or create a TurboQuantDB at the given path.
    /// `uri` can be a local directory path: "/data/my_db"
    /// Cloud URIs (s3://, gs://) are planned for Phase 5.
    #[staticmethod]
    #[pyo3(signature = (uri, dimension, bits, seed=42))]
    fn open(uri: &str, dimension: usize, bits: usize, seed: u64) -> PyResult<Self> {
        let engine = TurboQuantEngine::open(uri, dimension, bits, seed)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            engine: Arc::new(RwLock::new(engine)),
        })
    }

    /// Insert a single vector with optional metadata dictionary.
    fn insert(
        &self,
        py: Python<'_>,
        id: String,
        vector: PyReadonlyArray1<f64>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let vec = vector.as_array().to_owned();
        let props = parse_pydict(py, metadata)?;

        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine.insert(id, &vec, props)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Search for the top_k most similar vectors. Returns a list of dicts.
    fn search(
        &self,
        py: Python<'_>,
        query: PyReadonlyArray1<f64>,
        top_k: usize,
    ) -> PyResult<PyObject> {
        let q = query.as_array().to_owned();

        let results = py.allow_threads(|| {
            let engine = self.engine.read().unwrap();
            engine.search(&q, top_k)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;

        let py_list = PyList::empty(py);
        for r in results {
            let dict = PyDict::new(py);
            dict.set_item("id", &r.id)?;
            dict.set_item("score", r.score)?;
            let meta_dict = PyDict::new(py);
            for (k, v) in &r.metadata {
                meta_dict.set_item(k, json_to_py(py, v)?)?;
            }
            dict.set_item("metadata", meta_dict)?;
            py_list.append(dict)?;
        }
        Ok(py_list.into())
    }

    /// Flush all buffered WAL entries to disk segments.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine.flush_wal_to_segment()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Close the database, flushing remaining buffers.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut engine = self.engine.write().unwrap();
            engine.close()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Return database statistics as a dict.
    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = {
            let engine = self.engine.read().unwrap();
            engine.stats()
        };
        let dict = PyDict::new(py);
        dict.set_item("vector_count", stats.vector_count)?;
        dict.set_item("segment_count", stats.segment_count)?;
        dict.set_item("buffered_vectors", stats.buffered_vectors)?;
        dict.set_item("dimension", stats.d)?;
        dict.set_item("bits", stats.b)?;
        Ok(dict.into())
    }

    fn __len__(&self) -> PyResult<usize> {
        let count = self.engine.read().unwrap().vector_count();
        Ok(count as usize)
    }

    fn __repr__(&self) -> PyResult<String> {
        let engine = self.engine.read().unwrap();
        Ok(format!("TurboQuantDB(d={}, b={}, vectors={})", engine.d, engine.b, engine.vector_count()))
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

/// Convert a Python object to a serde_json Value (best-effort)
fn py_to_json(_py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
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
    // Fallback: store string representation
    Ok(JsonValue::String(obj.str()?.to_string()))
}

/// Convert a serde_json Value to a Python object (pyo3 0.21 compatible)
fn json_to_py(py: Python<'_>, val: &JsonValue) -> PyResult<PyObject> {
    Ok(match val {
        JsonValue::Null => py.None(),
        JsonValue::Bool(b) => (*b).to_object(py),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() { i.to_object(py) }
            else if let Some(f) = n.as_f64() { f.to_object(py) }
            else { n.to_string().to_object(py) }
        },
        JsonValue::String(s) => s.to_object(py),
        JsonValue::Array(arr) => {
            let list = PyList::empty(py);
            for v in arr { list.append(json_to_py(py, v)?)?; }
            list.into()
        },
        JsonValue::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map { dict.set_item(k, json_to_py(py, v)?)?; }
            dict.into()
        },
    })
}

/// Register the Database class with the PyO3 module.
pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Database>()?;
    Ok(())
}
