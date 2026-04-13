#![allow(unsafe_op_in_unsafe_fn)]
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::storage::engine::{
    BatchWriteItem, BatchWriteMode, DistanceMetric, GetResult, Manifest, RerankPrecision,
    TurboQuantEngine,
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
    ///     rerank: Enable raw-vector reranking. Default ``False``.
    ///     fast_mode: When ``True`` (default), all ``bits`` go to the MSE codebook; the QJL
    ///                residual is not stored (minimum disk, fastest ingest). When ``False``,
    ///                the QJL residual is stored and used during reranking, giving +5–15 pp
    ///                R@1 at d ≥ 1536. For d < 512, QJL projections are too noisy and reduce
    ///                recall, so keep ``fast_mode=True`` for low-dimensional workloads.
    ///     rerank_precision: Raw-vector reranking precision:
    ///         - ``None`` (default): ``"int8"`` when ``rerank=True``, disabled otherwise.
    ///         - ``"int8"``: store raw vectors as per-vector-scaled INT8 (~75% less disk than f32).
    ///         - ``"int4"``: store raw vectors as INT4 (~87.5% less disk than f32, minor recall drop).
    ///         - ``"f16"``: store raw vectors as float16 (+n×d×2 bytes), exact reranking.
    ///         - ``"f32"``: store raw vectors as float32 (+n×d×4 bytes), maximum precision.
    ///         - ``"disabled"`` / ``"dequant"``: no extra storage; reranking uses dequantization
    ///           (mathematically equivalent to no rerank for IP metric — use only to save disk).
    ///     wal_flush_threshold: Number of buffered vectors before flushing WAL to a segment file.
    ///         Higher values give faster bulk ingest (fewer flush cycles) at the cost of more
    ///         data at risk if the process crashes before flush. Default ``5000``.
    ///         Set to ``100`` to restore old conservative behaviour (more flushes, same final
    ///         disk/RAM — ``close()`` always trims the file to the exact slot count).
    ///     normalize: When ``True`` the engine L2-normalises every inserted vector and every
    ///         query vector internally so that IP scoring equals cosine similarity.  Callers
    ///         that already emit unit vectors can set this to avoid repeating the normalisation
    ///         themselves.  Default ``False``.
    ///     quantizer_type: Which quantizer rotation to use:
    ///         - ``None`` / ``"dense"`` (default): Haar-uniform QR rotation + dense Gaussian
    ///           projection. n=d (no padding). Best recall; O(d²) ingest cost.
    ///           ``"exact"`` is accepted as a legacy alias.
    ///         - ``"srht"``: structured Walsh-Hadamard rotation. n=next_power_of_two(d).
    ///           O(d log d) ingest; ~25% more subspaces scored at search time.
    ///           Use for streaming or frequent-ingest workloads at high d.
    ///         See `docs/QUANTIZER_MODES.md` for a full CPU/RAM/disk/recall comparison.
    ///
    /// Returns:
    ///     An open :class:`Database` instance.
    ///
    /// Example::
    ///
    ///     db = Database.open("mydb", dimension=1536, bits=4, metric="cosine")
    ///     # Re-open an existing database without specifying parameters:
    ///     db = Database.open("mydb")
    ///     # Equivalent cosine-via-IP with auto-normalization:
    ///     db = Database.open("mydb", dimension=1536, bits=4, metric="ip", normalize=True)
    #[staticmethod]
    #[pyo3(signature = (path, dimension=None, bits=4, seed=42, metric="ip", rerank=false, fast_mode=true, rerank_precision=None, collection=None, wal_flush_threshold=None, normalize=false, quantizer_type=None))]
    fn open(
        path: String,
        dimension: Option<usize>,
        bits: usize,
        seed: u64,
        metric: &str,
        rerank: bool,
        fast_mode: bool,
        rerank_precision: Option<&str>,
        collection: Option<&str>,
        wal_flush_threshold: Option<usize>,
        normalize: bool,
        quantizer_type: Option<String>,
    ) -> PyResult<Self> {
        let engine_path = match collection {
            Some(col) if !col.is_empty() => {
                // Reject path traversal: collection must be a single, safe path component.
                let col_path = std::path::Path::new(col);
                if col_path.components().any(|c| {
                    matches!(
                        c,
                        std::path::Component::ParentDir
                            | std::path::Component::RootDir
                            | std::path::Component::CurDir
                    )
                }) || col == "."
                    || col.contains('/')
                    || col.contains('\\')
                {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "invalid collection name {:?}: must be a single path component with no separators",
                        col
                    )));
                }
                let p = std::path::Path::new(&path).join(col);
                p.to_string_lossy().into_owned()
            }
            _ => path.clone(),
        };
        std::fs::create_dir_all(&engine_path)
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;

        // If dimension is not provided, load it (and other fixed params) from
        // the existing manifest.  This allows callers to reopen a database
        // with just its path: `Database.open("./mydb")`.
        let manifest_path = format!("{}/manifest.json", engine_path);
        let (dimension, bits, seed, metric_str) = if let Some(d) = dimension {
            (d, bits, seed, metric.to_string())
        } else if std::path::Path::new(&manifest_path).exists() {
            let m = Manifest::load(&manifest_path).map_err(to_py_runtime)?;
            let m_metric = match m.metric {
                DistanceMetric::Ip => "ip",
                DistanceMetric::Cosine => "cosine",
                DistanceMetric::L2 => "l2",
            };
            (m.d, m.b, m.seed, m_metric.to_string())
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dimension is required when opening a new database",
            ));
        };

        let dist_metric = match metric_str.to_lowercase().as_str() {
            "ip" => DistanceMetric::Ip,
            "cosine" => DistanceMetric::Cosine,
            "l2" => DistanceMetric::L2,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid metric: {}",
                    metric_str
                )));
            }
        };

        let precision = match rerank_precision.map(|s| s.to_lowercase()) {
            // Default: INT8 per-vector-scaled when rerank=True.
            // INT8 gives near-identical recall to F16 at ~50% less disk for any d.
            // Dequantization reranking is mathematically equivalent to the quantized LUT score
            // for IP metric, so it produces no recall improvement without raw vector storage.
            None => {
                if rerank {
                    RerankPrecision::Int8 // ~50% less disk than f16, same recall
                } else {
                    RerankPrecision::Disabled
                }
            }
            Some(ref s) if s == "disabled" || s == "dequant" => RerankPrecision::Disabled,
            Some(ref s) if s == "int8" || s == "i8" => RerankPrecision::Int8,
            Some(ref s) if s == "int4" || s == "i4" => RerankPrecision::Int4,
            Some(ref s) if s == "f16" || s == "half" => RerankPrecision::F16,
            Some(ref s) if s == "f32" || s == "float" || s == "full" => RerankPrecision::F32,
            Some(ref s) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid rerank_precision: '{}'. Use 'int8' (default), 'int4', 'f16', 'f32', or None.",
                    s
                )));
            }
        };

        if bits == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("bits must be >= 1"));
        }
        if bits < 2 && !fast_mode {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "bits must be >= 2 when fast_mode=False (1 bit is reserved for QJL residual); got bits={}",
                bits
            )));
        }
        if bits > 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "bits must be <= 8; got {} (values above 8 cause exponential codebook-generation slowdown)",
                bits
            )));
        }
        if let Some(qt) = quantizer_type.as_deref() {
            let qt_lower = qt.to_ascii_lowercase();
            if qt_lower != "dense" && qt_lower != "exact" && qt_lower != "srht" {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid quantizer_type: '{}'. Valid values: 'dense', 'exact', 'srht'.",
                    qt
                )));
            }
        }
        if dimension == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dimension must be > 0",
            ));
        }
        if let Some(wft) = wal_flush_threshold {
            if wft == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "wal_flush_threshold must be >= 1",
                ));
            }
        }
        let engine = TurboQuantEngine::open_with_options(
            &engine_path,
            &engine_path,
            dimension,
            bits,
            seed,
            dist_metric,
            rerank,
            fast_mode,
            precision,
            wal_flush_threshold,
            normalize,
            quantizer_type,
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
        let vec = extract_vec1d(py, &vector)?;
        check_finite(&vec.view(), "vector")?;
        let props = parse_pydict(metadata)?;
        py.allow_threads(|| {
            let mut engine = self.write_engine()?;
            if vec.len() != engine.d {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "vector dimension mismatch: expected {}, got {}",
                    engine.d,
                    vec.len()
                )));
            }
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
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "ids length ({}) does not match number of vectors ({})",
                    ids.len(),
                    matrix.nrows()
                )));
            }
            {
                let engine = self.read_engine()?;
                if !matrix.is_empty() && matrix.ncols() != engine.d {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "vector dimension mismatch: expected {}, got {}",
                        engine.d,
                        matrix.ncols()
                    )));
                }
            }
            if matrix.iter().any(|&x| !x.is_finite()) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "vectors contain non-finite values (NaN or Inf)",
                ));
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
                    let mut engine = self.write_engine()?;
                    engine
                        .insert_many_with_mode(chunk_items, b_mode)
                        .map_err(to_py_runtime)
                })?;
            }
        } else {
            let v64 = vectors.extract::<PyReadonlyArray2<f64>>(py)?;
            let matrix = v64.as_array();
            if ids.len() != matrix.nrows() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "ids length ({}) does not match number of vectors ({})",
                    ids.len(),
                    matrix.nrows()
                )));
            }
            {
                let engine = self.read_engine()?;
                if !matrix.is_empty() && matrix.ncols() != engine.d {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "vector dimension mismatch: expected {}, got {}",
                        engine.d,
                        matrix.ncols()
                    )));
                }
            }
            if matrix.iter().any(|&x| !x.is_finite()) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "vectors contain non-finite values (NaN or Inf)",
                ));
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
                    let mut engine = self.write_engine()?;
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
        let vec = extract_vec1d(py, &vector)?;
        let props = parse_pydict(metadata)?;
        py.allow_threads(|| {
            let mut engine = self.write_engine()?;
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
        let vec = extract_vec1d(py, &vector)?;
        let props = parse_pydict(metadata)?;
        py.allow_threads(|| {
            let mut engine = self.write_engine()?;
            engine
                .update_with_document(id, &vec, props, document)
                .map_err(to_py_runtime)
        })
    }

    fn delete(&self, py: Python<'_>, id: String) -> PyResult<bool> {
        py.allow_threads(|| {
            let mut engine = self.write_engine()?;
            engine.delete(id).map_err(to_py_runtime)
        })
    }

    /// Delete multiple vectors in a single call.
    ///
    /// Args:
    ///     ids: List of IDs to delete. IDs not present are silently skipped.
    ///         May be empty when ``where_filter`` is provided.
    ///     where_filter: Optional metadata filter (same syntax as :meth:`search`).
    ///         When provided, all vectors matching the filter are deleted in
    ///         addition to any explicitly listed IDs.  Overlapping entries are
    ///         not double-counted.
    ///
    /// Returns:
    ///     The number of vectors that were found and deleted.
    ///
    /// Example::
    ///
    ///     deleted = db.delete_batch(["id1", "id2", "id3"])
    ///     # Delete all vectors older than 2020:
    ///     deleted = db.delete_batch(where_filter={"year": {"$lt": 2020}})
    #[pyo3(signature = (ids=vec![], where_filter=None))]
    fn delete_batch(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        where_filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<usize> {
        let has_filter = where_filter.is_some();
        let parsed_filter = parse_pydict(where_filter)?;
        if !parsed_filter.is_empty() {
            crate::storage::engine::filter::validate_filter_operators(&parsed_filter)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        }
        py.allow_threads(|| {
            let mut engine = self.write_engine()?;
            let mut deleted = if !ids.is_empty() {
                engine.delete_batch(ids).map_err(to_py_runtime)?
            } else {
                0
            };
            if has_filter {
                deleted += engine.delete_where(&parsed_filter).map_err(to_py_runtime)?;
            }
            Ok(deleted)
        })
    }

    /// Count vectors matching an optional metadata filter.
    ///
    /// When called without a filter this is O(1) and equivalent to
    /// ``db.stats()["vector_count"]``. With a filter it performs an O(n)
    /// scan and returns the number of matching active vectors.
    ///
    /// Args:
    ///     filter: Optional metadata filter dict using the same syntax as
    ///             :meth:`search`.
    ///
    /// Returns:
    ///     Integer count of matching vectors.
    ///
    /// Example::
    ///
    ///     total = db.count()
    ///     ml_docs = db.count(filter={"topic": "ml"})
    ///     recent = db.count(filter={"year": {"$gte": 2023}})
    #[pyo3(signature = (filter=None))]
    fn count(&self, py: Python<'_>, filter: Option<&Bound<'_, PyDict>>) -> PyResult<usize> {
        let parsed_filter = parse_pydict(filter)?;
        if !parsed_filter.is_empty() {
            crate::storage::engine::filter::validate_filter_operators(&parsed_filter)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        }
        let filter_ref = if parsed_filter.is_empty() {
            None
        } else {
            Some(&parsed_filter)
        };
        py.allow_threads(|| {
            let engine = self.read_engine()?;
            engine.count_with_filter(filter_ref).map_err(to_py_runtime)
        })
    }

    fn get(&self, py: Python<'_>, id: String) -> PyResult<Option<PyObject>> {
        let got = py.allow_threads(|| {
            let engine = self.read_engine()?;
            engine.get(&id).map_err(to_py_runtime)
        })?;
        Ok(match got {
            Some(g) => Some(get_result_to_py(py, &g)?),
            None => None,
        })
    }

    fn get_many(&self, py: Python<'_>, ids: Vec<String>) -> PyResult<PyObject> {
        let results = py.allow_threads(|| {
            let engine = self.read_engine()?;
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
        let engine = self.read_engine()?;
        Ok(engine.list_all())
    }

    /// Search for the nearest neighbours of a query vector.
    ///
    /// The HNSW index is used when an index has been built via
    /// :meth:`create_index` **and** ``_use_ann=True`` is passed.
    /// With ``_use_ann=False`` (the default) the search always uses exhaustive
    /// brute-force scoring, which gives the highest recall at the cost of
    /// linear scan time.
    ///
    /// Args:
    ///     query: Query vector (1-D numpy array, float32 or float64).
    ///     top_k: Number of results to return.
    ///     filter: Optional metadata filter dict.
    ///     ann_search_list_size: HNSW ef_search override (larger = more recall, slower).
    ///         Only relevant when ``_use_ann=True`` and an index exists.
    ///     rerank_factor: Oversampling multiplier for the rerank candidate pool.
    ///         When ``rerank=True``, the engine scores ``top_k * rerank_factor`` candidates
    ///         with the quantized scorer then re-scores with exact raw vectors.
    ///         Higher values improve recall at the cost of latency. Default ``None`` uses
    ///         built-in defaults (10× brute-force, 20× ANN). Range: 1–100.
    ///         Ignored when ``rerank=False``.
    ///     include: Subset of fields to return — ``["id", "score", "metadata", "document"]``.
    ///              Defaults to all four.
    ///
    /// Returns:
    ///     List of dicts, each with keys ``id``, ``score``, ``metadata``, ``document``.
    #[pyo3(signature = (query, top_k, filter=None, _use_ann=None, ann_search_list_size=None, include=None, rerank_factor=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: PyObject,
        top_k: i64,
        filter: Option<&Bound<'_, PyDict>>,
        _use_ann: Option<bool>,
        ann_search_list_size: Option<usize>,
        include: Option<Vec<String>>,
        rerank_factor: Option<usize>,
    ) -> PyResult<PyObject> {
        if top_k <= 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "top_k must be a positive integer, got {}",
                top_k
            )));
        }
        if let Some(rf) = rerank_factor {
            if !(1..=100).contains(&rf) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "rerank_factor must be between 1 and 100, got {}",
                    rf
                )));
            }
        }
        let top_k = usize::try_from(top_k).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "top_k is too large to represent on this platform: {}",
                top_k
            ))
        })?;
        if let Some(rf) = rerank_factor {
            top_k.checked_mul(rf).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("top_k * rerank_factor is too large")
            })?;
        }
        let q = extract_vec1d(py, &query)?;
        check_finite(&q.view(), "query vector")?;
        let parsed_filter = parse_pydict(filter)?;
        if !parsed_filter.is_empty() {
            crate::storage::engine::filter::validate_filter_operators(&parsed_filter)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        }
        let filter_ref = if parsed_filter.is_empty() {
            None
        } else {
            Some(&parsed_filter)
        };

        let inc = parse_include_set(include, &["id", "score", "metadata", "document"])?;

        let (results, is_l2) = py.allow_threads(|| {
            let engine = self.read_engine()?;
            if q.len() != engine.d {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "query dimension mismatch: expected {}, got {}",
                    engine.d,
                    q.len()
                )));
            }
            let is_l2 = matches!(engine.metric, crate::storage::engine::DistanceMetric::L2);
            let use_ann = _use_ann.unwrap_or_else(|| engine.auto_use_ann());
            let results = engine
                .search_with_filter_and_ann_include(
                    &q,
                    top_k,
                    filter_ref,
                    ann_search_list_size,
                    use_ann,
                    rerank_factor,
                    inc.contains("id"),
                    inc.contains("metadata"),
                    inc.contains("document"),
                )
                .map_err(to_py_runtime)?;
            Ok((results, is_l2))
        })?;

        let py_list = PyList::empty_bound(py);
        for r in results {
            let dict = PyDict::new_bound(py);
            if inc.contains("id") {
                dict.set_item("id", &r.id)?;
            }
            if inc.contains("score") {
                let user_score = if is_l2 { -r.score } else { r.score };
                dict.set_item("score", user_score)?;
            }
            if inc.contains("metadata") {
                let meta_dict = PyDict::new_bound(py);
                for (k, v) in r.metadata {
                    meta_dict.set_item(k, json_to_py(py, &v)?)?;
                }
                dict.set_item("metadata", meta_dict)?;
            }
            if inc.contains("document") {
                if let Some(doc) = r.document {
                    dict.set_item("document", doc)?;
                }
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
            let mut engine = self.write_engine()?;
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

    /// Incrementally extend an existing HNSW index with vectors inserted since the last
    /// ``create_index()`` call. If no graph exists, behaves like ``create_index()``.
    ///
    /// Significantly faster than a full rebuild for small deltas (< ~10% of corpus).
    /// For large deltas, prefer ``create_index()`` to get the full quality build.
    #[pyo3(signature = (max_degree=None, ef_construction=None, search_list_size=None, alpha=None, n_refinements=None))]
    fn update_index(
        &self,
        py: Python<'_>,
        max_degree: Option<usize>,
        ef_construction: Option<usize>,
        search_list_size: Option<usize>,
        alpha: Option<f64>,
        n_refinements: Option<usize>,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            let mut engine = self.write_engine()?;
            engine
                .create_index_incremental(
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
            let engine = self.read_engine()?;
            engine.stats()
        };
        let dict = PyDict::new_bound(py);
        dict.set_item("vector_count", stats.vector_count)?;
        dict.set_item("segment_count", stats.segment_count)?;
        dict.set_item("buffered_vectors", stats.buffered_vectors)?;
        dict.set_item("dimension", stats.d)?;
        dict.set_item("bits", stats.b)?;
        dict.set_item("metric", &stats.metric)?;
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
        dict.set_item("delta_size", stats.delta_size)?;
        // Computed estimate: in-memory footprint across all major buffers.
        let ram_estimate_bytes = stats.live_codes_bytes
            + stats.live_vectors_bytes_estimate
            + stats.metadata_bytes_estimate;
        dict.set_item("ram_estimate_bytes", ram_estimate_bytes)?;
        Ok(dict.into())
    }

    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut engine = self.write_engine()?;
            engine.flush_wal_to_segment().map_err(to_py_runtime)
        })
    }

    /// Force a maintenance checkpoint: flush WAL and compact segments when
    /// compaction threshold is reached.
    fn checkpoint(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut engine = self.write_engine()?;
            engine.checkpoint().map_err(to_py_runtime)
        })
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut engine = self.write_engine()?;
            engine.close().map_err(to_py_runtime)
        })
    }

    /// `len(db)` — total number of active vectors.
    fn __len__(&self) -> PyResult<usize> {
        let engine = self.read_engine()?;
        engine.count_with_filter(None).map_err(to_py_runtime)
    }

    /// `id in db` — True if the ID exists in the database.
    fn __contains__(&self, py: Python<'_>, id: String) -> PyResult<bool> {
        py.allow_threads(|| {
            let engine = self.read_engine()?;
            engine.get(&id).map(|r| r.is_some()).map_err(to_py_runtime)
        })
    }

    /// Update metadata and/or document for an existing ID without re-uploading
    /// the vector.  The quantised representation is untouched.
    ///
    /// Args:
    ///     id: ID to update. Raises ``RuntimeError`` if not found.
    ///     metadata: New metadata dict, or ``None`` to preserve existing.
    ///     document: New document string, or ``None`` to preserve existing.
    ///
    /// Example::
    ///
    ///     db.update_metadata("doc-1", metadata={"status": "published"})
    ///     db.update_metadata("doc-1", document="updated text")
    #[pyo3(signature = (id, metadata=None, document=None))]
    fn update_metadata(
        &self,
        py: Python<'_>,
        id: String,
        metadata: Option<&Bound<'_, PyDict>>,
        document: Option<String>,
    ) -> PyResult<()> {
        let props = parse_pydict(metadata)?;
        py.allow_threads(|| {
            let mut engine = self.write_engine()?;
            engine
                .update_metadata_only(&id, props, document)
                .map_err(to_py_runtime)
        })
    }

    /// Search with multiple query vectors in one call.
    ///
    /// Args:
    ///     query_embeddings: 2-D array of shape ``(N, D)``.
    ///     n_results: Number of results per query. Default 10.
    ///     where_filter: Optional metadata filter (same syntax as :meth:`search`).
    ///     ann_search_list_size: HNSW ef_search override (larger = more recall, slower).
    ///         Only relevant when ``_use_ann=True`` and an index exists.
    ///     rerank_factor: Oversampling multiplier for the rerank candidate pool (see
    ///         :meth:`search` for full description). Default ``None``.
    ///     include: Subset of fields to return — ``["id", "score", "metadata", "document"]``.
    ///         Defaults to all four.
    ///
    /// Note:
    ///     Pass ``_use_ann=True`` to engage the HNSW index (must be built first via
    ///     :meth:`create_index`). Default is ``False`` (exhaustive brute-force).
    ///
    /// Returns:
    ///     List of N result lists. Each result dict contains the keys requested
    ///     via ``include``.
    ///
    /// Example::
    ///
    ///     all_results = db.query(
    ///         query_embeddings=np.stack([q1, q2, q3]),
    ///         n_results=5,
    ///     )
    #[pyo3(signature = (query_embeddings, n_results=10, where_filter=None, _use_ann=None, ann_search_list_size=None, rerank_factor=None, include=None))]
    fn query(
        &self,
        py: Python<'_>,
        query_embeddings: PyObject,
        n_results: i64,
        where_filter: Option<&Bound<'_, PyDict>>,
        _use_ann: Option<bool>,
        ann_search_list_size: Option<usize>,
        rerank_factor: Option<usize>,
        include: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        if n_results <= 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_results must be a positive integer, got {}",
                n_results
            )));
        }
        let n_results = n_results as usize;
        let queries: Vec<ndarray::Array1<f64>> =
            if let Ok(m) = query_embeddings.extract::<PyReadonlyArray2<f32>>(py) {
                let rows: Vec<_> = m
                    .as_array()
                    .rows()
                    .into_iter()
                    .map(|r| r.mapv(|x| x as f64))
                    .collect();
                for (i, row) in rows.iter().enumerate() {
                    check_finite(&row.view(), &format!("query_embeddings row {}", i))?;
                }
                rows
            } else if let Ok(m) = query_embeddings.extract::<PyReadonlyArray2<f64>>(py) {
                let rows: Vec<_> = m
                    .as_array()
                    .rows()
                    .into_iter()
                    .map(|r| r.to_owned())
                    .collect();
                for (i, row) in rows.iter().enumerate() {
                    check_finite(&row.view(), &format!("query_embeddings row {}", i))?;
                }
                rows
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "query_embeddings must be a 2-D numpy array (float32 or float64)",
                ));
            };

        let parsed_filter = parse_pydict(where_filter)?;
        if !parsed_filter.is_empty() {
            crate::storage::engine::filter::validate_filter_operators(&parsed_filter)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        }
        let filter_ref = if parsed_filter.is_empty() {
            None
        } else {
            Some(&parsed_filter)
        };
        let inc = parse_include_set(include, &["id", "score", "metadata", "document"])?;
        let include_id = inc.contains("id");
        let include_score = inc.contains("score");
        let include_metadata = inc.contains("metadata");
        let include_document = inc.contains("document");
        let (batch, is_l2) = py.allow_threads(|| {
            let engine = self.read_engine()?;
            for (i, row) in queries.iter().enumerate() {
                if row.len() != engine.d {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "query dimension mismatch: expected {}, got {} (row {})",
                        engine.d,
                        row.len(),
                        i
                    )));
                }
            }
            let is_l2 = matches!(engine.metric, crate::storage::engine::DistanceMetric::L2);
            let batch = engine
                .search_batch_with_include(
                    &queries,
                    n_results,
                    filter_ref,
                    ann_search_list_size,
                    _use_ann,
                    rerank_factor,
                    include_id,
                    include_metadata,
                    include_document,
                )
                .map_err(to_py_runtime)?;
            Ok((batch, is_l2))
        })?;

        let outer = PyList::empty_bound(py);
        for results in batch {
            let inner = PyList::empty_bound(py);
            for r in results {
                let dict = PyDict::new_bound(py);
                if include_id {
                    dict.set_item("id", r.id)?;
                }
                if include_score {
                    dict.set_item("score", if is_l2 { -r.score } else { r.score })?;
                }
                if include_metadata {
                    let meta_dict = PyDict::new_bound(py);
                    for (k, v) in r.metadata {
                        meta_dict.set_item(k, json_to_py(py, &v)?)?;
                    }
                    dict.set_item("metadata", meta_dict)?;
                }
                if include_document {
                    if let Some(doc) = r.document {
                        dict.set_item("document", doc)?;
                    }
                }
                inner.append(dict)?;
            }
            outer.append(inner)?;
        }
        Ok(outer.into())
    }

    /// Return IDs with optional metadata filter and pagination.
    ///
    /// Args:
    ///     where_filter: Optional metadata filter dict.
    ///     limit: Maximum number of IDs to return. ``None`` returns all matching.
    ///     offset: Number of IDs to skip. Default 0.
    ///
    /// Returns:
    ///     List of matching IDs.
    ///
    /// Example::
    ///
    ///     # All IDs for a given category, page 2
    ///     ids = db.list_ids(where_filter={"kind": "A"}, limit=50, offset=50)
    #[pyo3(signature = (where_filter=None, limit=None, offset=0))]
    fn list_ids(
        &self,
        py: Python<'_>,
        where_filter: Option<&Bound<'_, PyDict>>,
        limit: Option<i64>,
        offset: i64,
    ) -> PyResult<Vec<String>> {
        if offset < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "offset must be >= 0, got {}",
                offset
            )));
        }
        if let Some(lim) = limit {
            if lim < 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "limit must be >= 0, got {}",
                    lim
                )));
            }
        }
        let offset = offset as usize;
        let limit = limit.map(|l| l as usize);
        let parsed_filter = parse_pydict(where_filter)?;
        if !parsed_filter.is_empty() {
            crate::storage::engine::filter::validate_filter_operators(&parsed_filter)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        }
        let filter_ref = if parsed_filter.is_empty() {
            None
        } else {
            Some(&parsed_filter)
        };
        py.allow_threads(|| {
            let engine = self.read_engine()?;
            engine
                .list_with_filter_page(filter_ref, limit, offset)
                .map_err(to_py_runtime)
        })
    }

    /// Return a ``{value: count}`` dict of all unique values of *field* across active vectors.
    ///
    /// Useful for enumerating distinct sources, categories, or any other metadata dimension
    /// without a full ``list_all()`` scan.  Supports dotted paths (e.g. ``"meta.source"``).
    /// Non-string values are stringified via their JSON representation.
    ///
    /// Args:
    ///     field: Metadata field name (or dotted path) to aggregate.
    ///
    /// Returns:
    ///     Dict mapping each unique field value to its occurrence count.
    ///
    /// Example::
    ///
    ///     counts = db.list_metadata_values("source")
    ///     # → {"docs/readme.md": 42, "src/main.py": 17}
    fn list_metadata_values(&self, py: Python<'_>, field: String) -> PyResult<PyObject> {
        let counts = py.allow_threads(|| {
            let engine = self.read_engine()?;
            engine.list_metadata_values(&field).map_err(to_py_runtime)
        })?;
        let dict = PyDict::new_bound(py);
        for (k, v) in counts {
            dict.set_item(k, v)?;
        }
        Ok(dict.into())
    }
}

impl Database {
    fn read_engine(&self) -> PyResult<std::sync::RwLockReadGuard<'_, TurboQuantEngine>> {
        self.engine.read().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "database lock poisoned: a previous operation panicked; re-open the database",
            )
        })
    }

    fn write_engine(&self) -> PyResult<std::sync::RwLockWriteGuard<'_, TurboQuantEngine>> {
        self.engine.write().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "database lock poisoned: a previous operation panicked; re-open the database",
            )
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
    } else if let Ok(tup) = obj.downcast::<PyTuple>() {
        // BUG-12: tuples were falling through to Null — convert same as lists
        let mut arr = Vec::new();
        for item in tup {
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

/// Extract a 1-D f64 array from a Python object — accepts float32/float64 numpy
/// arrays and plain Python lists of numbers (BUG-8 fix).
fn extract_vec1d(py: Python<'_>, obj: &PyObject) -> PyResult<ndarray::Array1<f64>> {
    if let Ok(v) = obj.extract::<PyReadonlyArray1<f32>>(py) {
        return Ok(v.as_array().mapv(|x| x as f64));
    }
    if let Ok(v) = obj.extract::<PyReadonlyArray1<f64>>(py) {
        return Ok(v.as_array().to_owned());
    }
    // Fall back to plain Python list of numbers.
    let bound = obj.bind(py);
    if let Ok(list) = bound.downcast::<PyList>() {
        let v: Result<Vec<f64>, _> = list.iter().map(|x| x.extract::<f64>()).collect();
        return Ok(ndarray::Array1::from(v.map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "vector list elements must be numeric (int or float)",
            )
        })?));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "vector must be a numpy ndarray (float32 or float64) or a Python list of numbers",
    ))
}

fn check_finite(vec: &ndarray::ArrayView1<f64>, label: &str) -> PyResult<()> {
    if vec.iter().any(|&x| !x.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{} contains non-finite values (NaN or Inf)",
            label
        )));
    }
    Ok(())
}

fn parse_include_set(
    include: Option<Vec<String>>,
    defaults: &[&str],
) -> PyResult<std::collections::HashSet<String>> {
    let items = include.unwrap_or_else(|| defaults.iter().map(|s| s.to_string()).collect());
    let valid: std::collections::HashSet<&str> = defaults.iter().copied().collect();
    for item in &items {
        if !valid.contains(item.to_ascii_lowercase().as_str()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid include field {:?}; valid values are: {}",
                item,
                defaults.join(", ")
            )));
        }
    }
    Ok(items.into_iter().map(|s| s.to_ascii_lowercase()).collect())
}

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Database>()?;
    let db_cls = m.getattr("Database")?;
    m.add("TurboQuantDB", db_cls)?;
    Ok(())
}
