"""
LanceDB compatibility shim for tqdb.

Provides a ``connect()`` function that mirrors the LanceDB Python API surface
(v0.x / v1.x).  Internally each "table" is stored as a ``tqdb.Database``
under ``{uri}/{table_name}/``.

Supported usage pattern::

    from tqdb.lancedb_compat import connect

    db = connect("/data/lancedb")
    tbl = db.create_table("docs", data=pa_table)
    results = tbl.search(query_vec).metric("dot").limit(10).to_list()

PyArrow is required for data ingestion (``import pyarrow``).  Pandas is
optional (needed only for ``.to_pandas()``).

Intentionally not implemented (raises ``NotImplementedError`` or is a no-op):
- Remote / cloud URIs (``s3://``, ``gs://``, ``az://``)
- ``create_fts_index``, ``create_scalar_index``
- ``drop_database``
- Complex SQL WHERE predicates beyond the supported subset. The parser
  accepts ``id IN (...)``, ``field IN (...)``, string equality/inequality
  (``field = 'val'`` and ``field != 'val'``), numeric equality, and
  numeric comparisons (``>``, ``>=``, ``<``, ``<=``); anything else
  raises ``NotImplementedError``
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from .tqdb import Database
except ImportError:
    from tqdb.tqdb import Database  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Metric mapping
# ---------------------------------------------------------------------------

_METRIC_MAP: Dict[str, str] = {
    "dot": "ip",
    "ip": "ip",
    "cosine": "cosine",
    "l2": "l2",
    "euclidean": "l2",
}


def _map_metric(m: str) -> str:
    key = m.lower()
    if key not in _METRIC_MAP:
        raise ValueError(f"Unsupported metric '{m}'. Use: dot, ip, cosine, l2, euclidean.")
    return _METRIC_MAP[key]


def _validate_name_component(name: str, label: str) -> None:
    """Reject names with path traversal sequences."""
    if not name or name == ".." or "/" in name or "\\" in name or "\0" in name or ".." in name.split("/"):
        raise ValueError(f"{label} contains invalid characters or path traversal sequences: {name!r}")


# ---------------------------------------------------------------------------
# SQL WHERE parser (minimal subset)
# ---------------------------------------------------------------------------

_ID_IN_PATTERN = re.compile(
    r"""^\s*id\s+IN\s*\(([^)]+)\)\s*$""", re.IGNORECASE
)
_FIELD_IN_PATTERN = re.compile(
    r"""^\s*(\w+)\s+IN\s*\(([^)]+)\)\s*$""", re.IGNORECASE
)
_FIELD_EQ_STR_PATTERN = re.compile(
    r"""^\s*(\w+)\s*=\s*'([^']*)'\s*$"""
)
_FIELD_NEQ_STR_PATTERN = re.compile(
    r"""^\s*(\w+)\s*!=\s*'([^']*)'\s*$"""
)
_FIELD_EQ_NUM_PATTERN = re.compile(
    r"""^\s*(\w+)\s*=\s*(\d+(?:\.\d+)?)\s*$"""
)
_FIELD_CMP_PATTERN = re.compile(
    r"""^\s*(\w+)\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)\s*$"""
)

_CMP_OPS = {">=": "$gte", "<=": "$lte", ">": "$gt", "<": "$lt"}


def _parse_sql_where(where: str) -> Dict[str, Any]:
    """
    Parse a minimal SQL WHERE clause into a tqdb filter dict.

    Supported forms:
    - ``id IN ('a', 'b', 'c')``
    - ``field IN ('a', 'b')``
    - ``field = 'value'``
    - ``field != 'value'``
    - ``field = 42``
    - ``field > 5``, ``field >= 5``, ``field < 5``, ``field <= 5``

    Anything else raises ``NotImplementedError``.
    """
    m = _ID_IN_PATTERN.match(where)
    if m:
        ids_raw = m.group(1)
        parts = [s.strip() for s in ids_raw.split(",")]
        if not parts[-1]:
            raise ValueError(f"Syntax error in IN clause (trailing comma?): '{where}'")
        ids = [s.strip("'\"") for s in parts]
        return {"id": {"$in": ids}}
    m = _FIELD_IN_PATTERN.match(where)
    if m:
        field = m.group(1)
        vals_raw = m.group(2)
        parts = [s.strip() for s in vals_raw.split(",")]
        if not parts[-1]:  # trailing comma leaves an empty last part
            raise ValueError(f"Syntax error in IN clause (trailing comma?): '{where}'")
        vals = [s.strip("'\"") for s in parts]
        return {field: {"$in": vals}}
    m = _FIELD_EQ_STR_PATTERN.match(where)
    if m:
        field, value = m.group(1), m.group(2)
        return {field: {"$eq": value}}
    m = _FIELD_NEQ_STR_PATTERN.match(where)
    if m:
        field, value = m.group(1), m.group(2)
        return {field: {"$ne": value}}
    m = _FIELD_EQ_NUM_PATTERN.match(where)
    if m:
        field = m.group(1)
        num = float(m.group(2))
        return {field: {"$eq": num}}
    m = _FIELD_CMP_PATTERN.match(where)
    if m:
        field, op, num_str = m.group(1), m.group(2), m.group(3)
        return {field: {_CMP_OPS[op]: float(num_str)}}
    raise NotImplementedError(
        f"Complex SQL WHERE clause not supported: '{where}'. "
        "Supported: 'field IN (...)', \"field = 'value'\", \"field != 'value'\", "
        "'field = 42', 'field > 5', etc."
    )


# ---------------------------------------------------------------------------
# _VecStore — side-car float32 vector store for to_arrow/to_pandas support
# ---------------------------------------------------------------------------

class _VecStore:
    """
    Persists original float32 vectors so ``to_arrow()`` / ``to_pandas()``
    can include the ``vector`` column (tqdb quantizes vectors on write and
    does not expose the raw float32 values through ``get_many()``).
    Thread-safe via an internal lock.
    """

    def __init__(self, directory: str):
        import threading
        self._path = os.path.join(directory, "_vecs.npz")
        self._lock = threading.Lock()

    def _load(self):
        """Caller holds lock."""
        if not os.path.exists(self._path):
            return [], None
        with np.load(self._path, allow_pickle=False) as data:
            return data["ids"].tolist(), data["vecs"]

    def _save(self, ids: list, vecs: np.ndarray) -> None:
        """Caller holds lock."""
        np.savez(self._path, ids=np.asarray(ids, dtype=str), vecs=vecs.astype(np.float32))

    def add(self, new_ids: List[str], new_vecs: np.ndarray) -> None:
        """Upsert: existing entries with same IDs are replaced."""
        with self._lock:
            ids, vecs = self._load()
            if ids:
                new_id_set = set(new_ids)
                keep_idx = [i for i, id_ in enumerate(ids) if id_ not in new_id_set]
                ids = [ids[i] for i in keep_idx]
                vecs = vecs[keep_idx] if (vecs is not None and keep_idx) else np.empty((0, new_vecs.shape[1]), dtype=np.float32)
            else:
                vecs = np.empty((0, new_vecs.shape[1]), dtype=np.float32)
            self._save(ids + list(new_ids), np.concatenate([vecs, new_vecs.astype(np.float32)], axis=0))

    def remove(self, del_ids: List[str]) -> None:
        with self._lock:
            ids, vecs = self._load()
            if not ids or vecs is None:
                return
            del_set = set(del_ids)
            keep_idx = [i for i, id_ in enumerate(ids) if id_ not in del_set]
            if not keep_idx:
                if os.path.exists(self._path):
                    os.remove(self._path)
                return
            self._save([ids[i] for i in keep_idx], vecs[keep_idx])

    def get_all(self) -> Dict[str, List[float]]:
        with self._lock:
            ids, vecs = self._load()
        if not ids or vecs is None:
            return {}
        return {id_: vecs[i].tolist() for i, id_ in enumerate(ids)}

    def get_by_ids(self, query_ids: List[str]) -> Dict[str, List[float]]:
        with self._lock:
            ids, vecs = self._load()
        if not ids or vecs is None:
            return {}
        id_to_row = {id_: i for i, id_ in enumerate(ids)}
        return {id_: vecs[id_to_row[id_]].tolist() for id_ in query_ids if id_ in id_to_row}


# ---------------------------------------------------------------------------
# Data ingestion helpers
# ---------------------------------------------------------------------------

def _extract_rows(data: Any) -> List[Dict[str, Any]]:
    """
    Convert PyArrow Table or list[dict] to a list of dicts.

    Each dict must contain a ``vector`` key.
    """
    try:
        import pyarrow as pa  # type: ignore
        if isinstance(data, pa.Table):
            return data.to_pylist()
    except ImportError:
        pass
    if isinstance(data, list):
        return data
    raise TypeError(
        f"data must be a PyArrow Table or list[dict], got {type(data).__name__}"
    )


# ---------------------------------------------------------------------------
# MergeInsertBuilder
# ---------------------------------------------------------------------------

class MergeInsertBuilder:
    """
    Minimal ``merge_insert`` builder matching the LanceDB fluent API.

    Only ``when_matched_update_all()`` + ``when_not_matched_insert_all()``
    are implemented; other clauses are accepted and ignored.
    """

    def __init__(self, table: "CompatTable", on: str):
        self._table = table
        self._on = on
        self._update_all = False
        self._insert_all = False

    def when_matched_update_all(self, condition: Optional[str] = None) -> "MergeInsertBuilder":
        self._update_all = True
        return self

    def when_not_matched_insert_all(self) -> "MergeInsertBuilder":
        self._insert_all = True
        return self

    def when_not_matched_by_source_delete(self, condition: Optional[str] = None) -> "MergeInsertBuilder":
        # Not supported — silently accepted so callers don't crash
        return self

    def execute(self, data: Any) -> None:
        rows = _extract_rows(data)
        if not rows:
            return
        # Map on-column → tqdb id
        ids = [str(r.get(self._on, r.get("id", i))) for i, r in enumerate(rows)]
        vecs_raw = [r.get("vector") for r in rows]
        if any(v is None for v in vecs_raw):
            raise ValueError("All rows in merge_insert data must have a 'vector' key.")
        vecs = np.stack([np.asarray(v, dtype=np.float32) for v in vecs_raw])
        metas = [{k: v for k, v in r.items() if k not in ("id", self._on, "vector", "document")} for r in rows]
        docs = [r.get("document") for r in rows]
        # upsert covers both update-when-matched and insert-when-not-matched
        mode = "upsert"
        db = self._table._open_db()
        db.insert_batch(ids, vecs, metas, docs, mode)
        self._table._vec_store.add(ids, vecs)
        if self._table._dim is None:
            self._table._dim = vecs.shape[1]
        self._table._persist_dim()


# ---------------------------------------------------------------------------
# CompatQuery  (fluent builder)
# ---------------------------------------------------------------------------

class CompatQuery:
    """
    Fluent query builder matching LanceDB's
    ``tbl.search(q).metric("dot").limit(k).where(f).to_list()`` pattern.

    Pass ``query=None`` to perform a full-table scan (no vector ranking).
    """

    def __init__(self, table: "CompatTable", query: Any):
        self._table = table
        # None → full-table scan; otherwise convert to float32 vector
        if query is None:
            self._query = None
        else:
            self._query = np.asarray(query, dtype=np.float32).flatten()
        self._metric: Optional[str] = None
        self._k: int = 10
        self._where: Optional[str] = None
        self._select: Optional[List[str]] = None

    def metric(self, m: str) -> "CompatQuery":
        self._metric = _map_metric(m)
        return self

    def limit(self, k: int) -> "CompatQuery":
        if k < 0:
            raise ValueError(f"limit must be non-negative, got {k}")
        self._k = k
        return self

    def where(self, filter_str: str, prefilter: bool = False) -> "CompatQuery":
        self._where = filter_str
        return self

    def select(self, columns: List[str]) -> "CompatQuery":
        self._select = columns
        return self

    # Silently accepted, no-op (tqdb has no IVF nprobe tuning)
    def nprobes(self, n: int) -> "CompatQuery":
        return self

    def refine_factor(self, n: int) -> "CompatQuery":
        return self

    def to_list(self) -> List[Dict[str, Any]]:
        tqdb_filter: Optional[Dict[str, Any]] = None
        id_allowset: Optional[set] = None
        if self._where:
            parsed = _parse_sql_where(self._where)
            id_cond = parsed.get("id")
            if id_cond and "$in" in id_cond:
                id_allowset = set(id_cond["$in"])
            else:
                tqdb_filter = parsed

        # Full-table scan when query is None
        if self._query is None:
            db = self._table._open_db()
            if id_allowset:
                all_ids = list(id_allowset)
            elif tqdb_filter:
                all_ids = db.list_ids(where_filter=tqdb_filter)
            else:
                all_ids = db.list_all()
            all_ids = all_ids[: self._k] if self._k > 0 else all_ids
            records = [r for r in db.get_many(all_ids) if r is not None]
            rows = []
            for r in records:
                row: Dict[str, Any] = {"id": r["id"]}
                meta = r.get("metadata") or {}
                for k, v in meta.items():
                    row[k] = v
                if r.get("document") is not None:
                    row["document"] = r["document"]
                if self._select:
                    row = {k: row[k] for k in self._select if k in row}
                rows.append(row)
            return rows

        # tqdb bakes metric into the DB at creation; per-query metric override is ignored.
        if self._metric is not None and self._metric != self._table._metric:
            import warnings
            warnings.warn(
                f"Metric override '{self._metric}' ignored; table was created with "
                f"metric='{self._table._metric}'. Re-create the table to change metric.",
                stacklevel=2,
            )
        db = self._table._open_db()
        # If id-filtering, over-fetch so post-filter can find enough results
        fetch_k = len(db) if id_allowset else self._k
        results = db.search(self._query, fetch_k, filter=tqdb_filter)
        if id_allowset:
            results = [r for r in results if r["id"] in id_allowset][: self._k]

        rows = []
        for r in results:
            row = {"id": r["id"], "_distance": r["score"]}
            meta = r.get("metadata") or {}
            for k, v in meta.items():
                row[k] = v
            if r.get("document") is not None:
                row["document"] = r["document"]
            if self._select:
                row = {k: row[k] for k in self._select if k in row}
            rows.append(row)
        return rows

    def to_arrow(self):
        import pyarrow as pa  # type: ignore
        rows = self.to_list()
        if not rows:
            return pa.table({})
        return pa.Table.from_pylist(rows)

    def to_pandas(self):
        import pandas as pd  # type: ignore
        return pd.DataFrame(self.to_list())


# ---------------------------------------------------------------------------
# CompatTable
# ---------------------------------------------------------------------------

class CompatTable:
    """Wraps a ``tqdb.Database`` behind the LanceDB Table interface."""

    def __init__(self, path: str, name: str, metric: str):
        self._path = path
        self._name = name
        self._metric = metric
        self._dim: Optional[int] = None
        self._db: Optional[Database] = None
        self._vec_store = _VecStore(path)
        # Recover dim from _lance_meta.json (written on first add)
        meta_path = os.path.join(path, "_lance_meta.json")
        if os.path.exists(meta_path):
            import json
            with open(meta_path) as f:
                info = json.load(f)
            self._dim = info.get("dim")
        # Fallback: read from tqdb manifest.json when _lance_meta.json is absent
        if self._dim is None:
            manifest = os.path.join(path, "manifest.json")
            if os.path.exists(manifest):
                import json
                with open(manifest) as f:
                    self._dim = json.load(f).get("d")

    def _open_db(self) -> Database:
        if self._db is not None:
            return self._db
        if self._dim is not None:
            self._db = Database.open(self._path, self._dim, metric=self._metric)
            return self._db
        raise RuntimeError(
            "Cannot open table: dimension unknown. "
            "Call add() or create_table(data=...) first."
        )

    def _persist_dim(self) -> None:
        """Persist dim (and metric) into _lance_meta.json after first ingestion."""
        import json
        meta_path = os.path.join(self._path, "_lance_meta.json")
        info: Dict[str, Any] = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                info = json.load(f)
        info["dim"] = self._dim
        info["metric"] = self._metric
        with open(meta_path, "w") as f:
            json.dump(info, f)

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return self.count_rows()

    @property
    def schema(self):
        """PyArrow schema inferred from stored metadata + vector dimension."""
        import pyarrow as pa  # type: ignore
        fields = [pa.field("id", pa.string())]
        if self._dim is not None:
            fields.append(pa.field("vector", pa.list_(pa.float32(), self._dim)))
        # Try to infer additional metadata fields from a sample record
        if os.path.exists(os.path.join(self._path, "manifest.json")):
            try:
                db = self._open_db()
                sample_ids = db.list_ids(limit=1)
                if sample_ids:
                    rec = db.get(sample_ids[0])
                    if rec and rec.get("metadata"):
                        for k, v in rec["metadata"].items():
                            if isinstance(v, bool):
                                fields.append(pa.field(k, pa.bool_()))
                            elif isinstance(v, int):
                                fields.append(pa.field(k, pa.int64()))
                            elif isinstance(v, float):
                                fields.append(pa.field(k, pa.float64()))
                            else:
                                fields.append(pa.field(k, pa.string()))
            except Exception:
                pass
        fields.append(pa.field("document", pa.string()))
        return pa.schema(fields)

    def add(self, data: Any, mode: str = "append") -> None:
        if mode not in ("append", "overwrite"):
            raise ValueError(f"Unsupported mode '{mode}'. Use 'append' or 'overwrite'.")
        rows = _extract_rows(data)
        if not rows:
            return
        vectors = [r["vector"] for r in rows]
        if isinstance(vectors[0], (list, tuple)):
            vecs = np.asarray(vectors, dtype=np.float32)
        else:
            vecs = np.stack([np.asarray(v, dtype=np.float32) for v in vectors])
        dim = vecs.shape[1]
        if self._dim is None:
            self._dim = dim

        ids = [str(r.get("id", i)) for i, r in enumerate(rows)]
        metas = []
        docs = []
        for r in rows:
            meta = {k: v for k, v in r.items() if k not in ("id", "vector", "document")}
            metas.append(meta if meta else {})
            docs.append(r.get("document"))

        if mode == "overwrite":
            # Wipe and recreate; invalidate cached db handle
            if os.path.exists(self._path):
                shutil.rmtree(self._path)
            os.makedirs(self._path, exist_ok=True)
            self._dim = dim
            self._db = None
            self._vec_store = _VecStore(self._path)

        if self._db is None:
            self._dim = dim
            self._db = Database.open(self._path, self._dim, metric=self._metric)
        write_mode = "upsert" if mode == "overwrite" else "insert"
        self._db.insert_batch(ids, vecs, metas, docs, write_mode)
        self._vec_store.add(ids, vecs)
        self._persist_dim()

    def search(self, query: Any) -> CompatQuery:
        """Return a query builder. Pass ``None`` for a full-table scan."""
        return CompatQuery(self, query)

    def delete(self, where: str) -> None:
        tqdb_filter = _parse_sql_where(where)
        db = self._open_db()
        if "id" in tqdb_filter and "$in" in tqdb_filter["id"]:
            ids = tqdb_filter["id"]["$in"]
            db.delete_batch(ids)
            self._vec_store.remove(ids)
        else:
            # filter-based delete
            results = db.list_ids(where_filter=tqdb_filter)
            if results:
                db.delete_batch(results)
                self._vec_store.remove(results)

    def update(self, where: str, values: Dict[str, Any]) -> None:
        """Update rows matching the SQL WHERE clause with new column values."""
        tqdb_filter = _parse_sql_where(where)
        db = self._open_db()
        # Resolve matching IDs — handle id-based filters as direct lookups
        # since `id` is the tqdb primary key, not a metadata field.
        if "id" in tqdb_filter:
            id_cond = tqdb_filter["id"]
            if "$in" in id_cond:
                ids = id_cond["$in"]
            elif "$eq" in id_cond:
                ids = [str(id_cond["$eq"])]
            else:
                ids = db.list_ids(where_filter=tqdb_filter)
        else:
            ids = db.list_ids(where_filter=tqdb_filter)
        if not ids:
            return
        vector_val = values.get("vector")
        doc_val = values.get("document")
        meta_vals = {k: v for k, v in values.items() if k not in ("vector", "document", "id")}
        for id_ in ids:
            if vector_val is not None:
                vec = np.asarray(vector_val, dtype=np.float32)
                db.update(id_, vec, meta_vals or None, doc_val)
                self._vec_store.add([id_], vec.reshape(1, -1))
            else:
                db.update_metadata(id_, meta_vals or None, doc_val)

    def merge_insert(self, on: str) -> MergeInsertBuilder:
        """Return a fluent merge-insert builder keyed on the given column."""
        return MergeInsertBuilder(self, on)

    def head(self, n: int = 5):
        """Return the first ``n`` rows as a PyArrow Table."""
        import pyarrow as pa  # type: ignore
        if not os.path.exists(os.path.join(self._path, "manifest.json")):
            return pa.table({})
        db = self._open_db()
        ids = db.list_ids(limit=n)
        if not ids:
            return pa.table({})
        records = [r for r in db.get_many(ids) if r is not None]
        vec_map = self._vec_store.get_by_ids(ids)
        rows = []
        for r in records:
            row = {"id": r["id"]}
            if r["id"] in vec_map:
                row["vector"] = vec_map[r["id"]]
            meta = r.get("metadata") or {}
            row.update(meta)
            if r.get("document") is not None:
                row["document"] = r["document"]
            rows.append(row)
        return pa.Table.from_pylist(rows)

    def count_rows(self, filter: Optional[str] = None) -> int:
        if not os.path.exists(os.path.join(self._path, "manifest.json")):
            return 0
        db = self._open_db()
        if filter:
            parsed = _parse_sql_where(filter)
            id_cond = parsed.get("id")
            if id_cond and "$in" in id_cond:
                id_set = set(id_cond["$in"])
                return sum(1 for id_ in db.list_all() if id_ in id_set)
            return db.count(filter=parsed)
        return len(db)

    def optimize(self, cleanup_older_than=None, delete_unverified: bool = False) -> None:
        """No-op stub — tqdb handles compaction automatically."""
        pass

    def create_index(
        self,
        metric: str = "L2",
        index_type: str = "IVF_PQ",
        num_partitions: int = 256,
        num_sub_vectors: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Delegate to tqdb.create_index() with sane defaults; IVF_PQ params ignored."""
        db = self._open_db()
        db.create_index()

    def to_arrow(self):
        """Return all rows as a PyArrow Table, including the ``vector`` column."""
        import pyarrow as pa  # type: ignore
        if not os.path.exists(os.path.join(self._path, "manifest.json")):
            return pa.table({})
        db = self._open_db()
        ids = db.list_all()
        if not ids:
            return pa.table({})
        records = [r for r in db.get_many(ids) if r is not None]
        vec_map = self._vec_store.get_all()
        rows = []
        for r in records:
            row = {"id": r["id"]}
            if r["id"] in vec_map:
                row["vector"] = vec_map[r["id"]]
            meta = r.get("metadata") or {}
            row.update(meta)
            if r.get("document") is not None:
                row["document"] = r["document"]
            rows.append(row)
        return pa.Table.from_pylist(rows)

    def to_list(self) -> List[Dict[str, Any]]:
        """Return all rows as a list of dicts, including the ``vector`` column."""
        if not os.path.exists(os.path.join(self._path, "manifest.json")):
            return []
        db = self._open_db()
        ids = db.list_all()
        if not ids:
            return []
        records = [r for r in db.get_many(ids) if r is not None]
        vec_map = self._vec_store.get_all()
        rows = []
        for r in records:
            row = {"id": r["id"]}
            if r["id"] in vec_map:
                row["vector"] = vec_map[r["id"]]
            meta = r.get("metadata") or {}
            row.update(meta)
            if r.get("document") is not None:
                row["document"] = r["document"]
            rows.append(row)
        return rows

    def to_pandas(self):
        """Return all rows as a Pandas DataFrame, including the ``vector`` column."""
        import pandas as pd  # type: ignore
        if not os.path.exists(os.path.join(self._path, "manifest.json")):
            return pd.DataFrame()
        db = self._open_db()
        ids = db.list_all()
        if not ids:
            return pd.DataFrame()
        records = [r for r in db.get_many(ids) if r is not None]
        vec_map = self._vec_store.get_all()
        rows = []
        for r in records:
            row = {"id": r["id"]}
            if r["id"] in vec_map:
                row["vector"] = vec_map[r["id"]]
            meta = r.get("metadata") or {}
            row.update(meta)
            if r.get("document") is not None:
                row["document"] = r["document"]
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CompatConnection
# ---------------------------------------------------------------------------

class CompatConnection:
    """LanceDB-compatible connection backed by tqdb."""

    def __init__(self, uri: str):
        if uri.startswith(("s3://", "gs://", "az://")):
            raise NotImplementedError(
                f"Cloud URIs are not supported in the tqdb LanceDB shim: {uri}"
            )
        self._root = os.path.abspath(uri)
        os.makedirs(self._root, exist_ok=True)
        import threading
        self._lock = threading.Lock()

    def _table_dir(self, name: str) -> str:
        return os.path.join(self._root, name)

    def _meta_path(self, name: str) -> str:
        return os.path.join(self._table_dir(name), "_lance_meta.json")

    def _is_table(self, name: str) -> bool:
        return os.path.isdir(self._table_dir(name)) and (
            os.path.exists(self._meta_path(name))
            or os.path.exists(os.path.join(self._table_dir(name), "manifest.json"))
        )

    def _load_metric(self, name: str) -> str:
        mp = self._meta_path(name)
        if os.path.exists(mp):
            import json
            with open(mp) as f:
                return json.load(f).get("metric", "ip")
        return "ip"

    def _save_meta(self, name: str, metric: str) -> None:
        import json
        with open(self._meta_path(name), "w") as f:
            json.dump({"metric": metric}, f)

    def create_table(
        self,
        name: str,
        data: Any = None,
        schema=None,
        mode: str = "create",
    ) -> CompatTable:
        _validate_name_component(name, "table name")
        with self._lock:
            if mode not in ("create", "overwrite"):
                raise ValueError(f"Unsupported mode '{mode}'. Use 'create' or 'overwrite'.")

            if mode == "create" and self._is_table(name):
                raise ValueError(f"Table '{name}' already exists. Use mode='overwrite' to replace it.")

            tbl_dir = self._table_dir(name)
            if mode == "overwrite" and os.path.exists(tbl_dir):
                shutil.rmtree(tbl_dir)
            os.makedirs(tbl_dir, exist_ok=True)

            metric = "ip"  # default; LanceDB default metric is "dot" ≡ ip
            self._save_meta(name, metric)
            tbl = CompatTable(tbl_dir, name, metric)
            if data is not None:
                tbl.add(data, mode="append")
        return tbl

    def open_table(self, name: str) -> CompatTable:
        if not self._is_table(name):
            raise ValueError(f"Table '{name}' not found at {self._root}.")
        metric = self._load_metric(name)
        return CompatTable(self._table_dir(name), name, metric)

    def drop_table(self, name: str) -> None:
        tbl_dir = self._table_dir(name)
        if not os.path.exists(tbl_dir):
            raise ValueError(f"Table '{name}' not found.")
        shutil.rmtree(tbl_dir)

    def table_names(self) -> List[str]:
        return sorted(
            entry.name
            for entry in os.scandir(self._root)
            if entry.is_dir() and (
                os.path.exists(os.path.join(entry.path, "_lance_meta.json"))
                or os.path.exists(os.path.join(entry.path, "manifest.json"))
            )
        )


def connect(uri: str) -> CompatConnection:
    """
    Factory matching ``lancedb.connect(uri)`` signature.

    ``uri`` must be a local directory path.  Cloud URIs raise
    ``NotImplementedError``.
    """
    return CompatConnection(uri)
