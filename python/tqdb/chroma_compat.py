"""
ChromaDB compatibility shim for tqdb.

Provides a drop-in ``PersistentClient`` that mirrors the chromadb ≥ 1.5 API
surface. Internally every collection is stored as a ``tqdb.Database`` under
``{path}/{collection_name}/``.

Supported entry points::

    from tqdb.chroma_compat import PersistentClient

    client = PersistentClient(path="/data/chroma")
    col = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})
    col.add(ids=["a"], embeddings=[[0.1, 0.2, ...]])
    results = col.query(query_embeddings=[[0.1, 0.2, ...]], n_results=5)

Intentionally not implemented (raises ``NotImplementedError``):
- ``HttpClient``, ``Settings``, server/cloud mode
- ``chromadb.Client()`` (ephemeral in-memory client)
- ``where_document`` filtering (no full-text search in tqdb)
- ``query_texts`` / ``add(documents=...)`` without a pre-computed ``embeddings``
  argument (call ``embedding_function`` yourself, then pass results as
  ``embeddings``).  Use ``embedding_function`` param to delegate automatically.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from .tqdb import Database
except ImportError:
    from tqdb.tqdb import Database  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HNSW_SPACE_TO_METRIC: Dict[str, str] = {
    "cosine": "cosine",
    "ip": "ip",
    "l2": "l2",
}


def _parse_metric(metadata: Optional[Dict[str, Any]]) -> str:
    if metadata is None:
        return "ip"
    space = metadata.get("hnsw:space", "ip")
    return _HNSW_SPACE_TO_METRIC.get(str(space).lower(), "ip")


def _sanitize_metadata(m: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce non-scalar metadata values to str (Chroma only allows str/int/float/bool)."""
    out: Dict[str, Any] = {}
    for k, v in m.items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _validate_collection_name(name: str) -> None:
    """Reject names with path traversal sequences."""
    if not name or name == ".." or "/" in name or "\\" in name or "\0" in name or ".." in name.split("/"):
        raise ValueError(f"Collection name contains invalid characters or path traversal sequences: {name!r}")


_VALID_GET_INCLUDE = {"ids", "metadatas", "documents", "embeddings"}
_VALID_QUERY_INCLUDE = {"ids", "metadatas", "documents", "embeddings", "distances"}


def _apply_filter(records: List[Dict[str, Any]], where: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Very small subset of Chroma where-filter evaluation."""
    def _matches(metadata: Dict[str, Any], expr: Dict[str, Any]) -> bool:
        if "$and" in expr:
            return all(_matches(metadata, sub) for sub in expr["$and"])
        if "$or" in expr:
            return any(_matches(metadata, sub) for sub in expr["$or"])
        for field, cond in expr.items():
            val = metadata.get(field)
            if isinstance(cond, dict):
                op, rhs = next(iter(cond.items()))
                _KNOWN_OPS = {
                    "$eq", "$ne", "$gt", "$gte", "$lt", "$lte",
                    "$in", "$nin", "$exists", "$contains",
                }
                if op not in _KNOWN_OPS:
                    raise ValueError(f"Unknown filter operator '{op}'")
                if op == "$eq" and val != rhs:
                    return False
                elif op == "$ne" and val == rhs:
                    return False
                elif op == "$gt" and not (val is not None and val > rhs):
                    return False
                elif op == "$gte" and not (val is not None and val >= rhs):
                    return False
                elif op == "$lt" and not (val is not None and val < rhs):
                    return False
                elif op == "$lte" and not (val is not None and val <= rhs):
                    return False
                elif op == "$in" and val not in rhs:
                    return False
                elif op == "$nin" and val in rhs:
                    return False
                elif op == "$exists":
                    present = field in metadata
                    if rhs and not present:
                        return False
                    if not rhs and present:
                        return False
                elif op == "$contains":
                    if not (isinstance(val, str) and rhs in val):
                        return False
            else:
                # bare equality
                if val != cond:
                    return False
        return True

    return [r for r in records if _matches(r.get("metadata") or {}, where)]


# ---------------------------------------------------------------------------
# _VecStore — side-car float32 vector store
# ---------------------------------------------------------------------------

class _VecStore:
    """
    Persists original float32 vectors alongside the tqdb quantized store.
    Required to implement Chroma's ``include=["embeddings"]`` API surface.
    Stored as ``_vecs.npz`` in the collection directory.
    Thread-safe via an internal lock.
    """

    def __init__(self, directory: str):
        import threading
        self._path = os.path.join(directory, "_vecs.npz")
        self._lock = threading.Lock()

    def _load(self):
        """Returns (ids: list[str], vecs: ndarray[float32] | None). Caller holds lock."""
        if not os.path.exists(self._path):
            return [], None
        with np.load(self._path, allow_pickle=False) as data:
            ids = data["ids"].tolist()
            vecs = data["vecs"]
        return ids, vecs

    def _save(self, ids: list, vecs: np.ndarray) -> None:
        """Caller holds lock."""
        np.savez(self._path, ids=np.asarray(ids, dtype=str), vecs=vecs.astype(np.float32))

    def add(self, new_ids: List[str], new_vecs: np.ndarray) -> None:
        """Upsert vectors — existing entries with the same IDs are replaced."""
        with self._lock:
            ids, vecs = self._load()
            if ids:
                new_id_set = set(new_ids)
                keep_idx = [i for i, id_ in enumerate(ids) if id_ not in new_id_set]
                ids = [ids[i] for i in keep_idx]
                vecs = vecs[keep_idx] if (vecs is not None and len(keep_idx) > 0) else np.empty((0, new_vecs.shape[1]), dtype=np.float32)
            else:
                vecs = np.empty((0, new_vecs.shape[1]), dtype=np.float32)
            all_ids = ids + list(new_ids)
            all_vecs = np.concatenate([vecs, new_vecs.astype(np.float32)], axis=0)
            self._save(all_ids, all_vecs)

    def remove(self, del_ids: List[str]) -> None:
        """Remove entries by ID."""
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
            kept_ids = [ids[i] for i in keep_idx]
            kept_vecs = vecs[keep_idx]
            self._save(kept_ids, kept_vecs)

    def get_by_ids(self, query_ids: List[str]) -> Dict[str, List[float]]:
        """Return {id: vector_as_list} for each requested ID that exists."""
        with self._lock:
            ids, vecs = self._load()
        if not ids or vecs is None:
            return {}
        id_to_row = {id_: i for i, id_ in enumerate(ids)}
        result = {}
        for id_ in query_ids:
            if id_ in id_to_row:
                result[id_] = vecs[id_to_row[id_]].tolist()
        return result

    def get_all(self) -> Dict[str, List[float]]:
        """Return all stored vectors as {id: vector_as_list}."""
        with self._lock:
            ids, vecs = self._load()
        if not ids or vecs is None:
            return {}
        return {id_: vecs[i].tolist() for i, id_ in enumerate(ids)}

    def relocate(self, new_directory: str) -> None:
        """Update the storage path after a directory rename."""
        with self._lock:
            self._path = os.path.join(new_directory, "_vecs.npz")


# ---------------------------------------------------------------------------
# CollectionInfo — returned by list_collections()
# ---------------------------------------------------------------------------

class CollectionInfo:
    """Minimal collection info object matching chromadb ≥ 0.4 Collection interface."""

    __slots__ = ("name", "id", "metadata")

    def __init__(self, name: str, id: str, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.id = id
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"Collection(name={self.name!r}, id={self.id!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CollectionInfo):
            return self.id == other.id
        return NotImplemented


# ---------------------------------------------------------------------------
# CompatCollection
# ---------------------------------------------------------------------------

class CompatCollection:
    """
    Wraps a ``tqdb.Database`` behind the chromadb Collection interface.

    The ``name`` and ``metadata`` are persisted in a sidecar
    ``_chroma_meta.json`` file beside the tqdb data directory.
    """

    def __init__(
        self,
        path: str,
        name: str,
        metric: str,
        embedding_function=None,
    ):
        self._path = path          # full path to the collection directory
        self._name = name
        self._metric = metric
        self._embedding_function = embedding_function
        self._db: Optional[Database] = None
        self._dim: Optional[int] = None
        self._vec_store = _VecStore(path)
        # Load dim from manifest.json if the DB already exists
        manifest = os.path.join(path, "manifest.json")
        if os.path.exists(manifest):
            try:
                import json as _json
                with open(manifest) as _f:
                    self._dim = _json.load(_f)["d"]
            except Exception:
                pass

    def _open_db(self, dim: Optional[int]) -> Database:
        if self._db is not None:
            return self._db
        if dim is not None:
            self._dim = dim
            self._db = Database.open(self._path, dim, metric=self._metric)
            return self._db
        # Reopen existing: read dim from manifest via stats
        if self._dim is not None:
            self._db = Database.open(self._path, self._dim, metric=self._metric)
            return self._db
        raise RuntimeError(
            "Cannot open collection: dimension unknown. "
            "Call add() with embeddings first."
        )

    def _ensure_dim(self, embeddings: List[List[float]]) -> int:
        if not embeddings:
            raise ValueError("embeddings list is empty")
        dim = len(embeddings[0])
        if self._dim is None:
            self._dim = dim
        elif self._dim != dim:
            raise ValueError(
                f"Embedding dimension {dim} does not match collection dimension {self._dim}"
            )
        return self._dim

    def _embed(self, texts: List[str], embeddings: Optional[List[List[float]]]) -> List[List[float]]:
        if embeddings is not None:
            return embeddings
        if self._embedding_function is not None:
            result = self._embedding_function(texts)
            if isinstance(result, np.ndarray):
                return result.tolist()
            return list(result)
        raise ValueError(
            "No embedding_function set. Pass embeddings explicitly, or "
            "install tqdb[embed] and pass an embedding_function."
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> str:
        """Stable UUID5 derived from the collection's storage path."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, "tqdb:" + os.path.abspath(self._path)))

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        """Collection metadata as stored in ``_chroma_meta.json``."""
        meta_path = os.path.join(self._path, "_chroma_meta.json")
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path) as f:
                info = json.load(f)
            return info.get("metadata")
        except Exception:
            return None

    def count(self) -> int:
        if not os.path.exists(os.path.join(self._path, "manifest.json")):
            return 0
        db = self._open_db(None)
        return len(db)

    def add(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        if embeddings is None:
            if documents is not None:
                embeddings = self._embed(documents, None)
            else:
                raise ValueError(
                    "No embedding_function set. Pass embeddings explicitly, or "
                    "install tqdb[embed] and pass an embedding_function."
                )
        dim = self._ensure_dim(embeddings)
        vecs = np.asarray(embeddings, dtype=np.float32)
        metas = [_sanitize_metadata(m) if m else {} for m in (metadatas or [{}] * len(ids))]
        db = self._open_db(dim)
        db.insert_batch(ids, vecs, metas, documents, "insert")
        self._vec_store.add(ids, vecs)

    def upsert(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        if embeddings is None:
            if documents is not None:
                embeddings = self._embed(documents, None)
            else:
                raise ValueError(
                    "No embedding_function set. Pass embeddings explicitly, or "
                    "install tqdb[embed] and pass an embedding_function."
                )
        dim = self._ensure_dim(embeddings)
        vecs = np.asarray(embeddings, dtype=np.float32)
        metas = [_sanitize_metadata(m) if m else {} for m in (metadatas or [{}] * len(ids))]
        db = self._open_db(dim)
        db.insert_batch(ids, vecs, metas, documents, "upsert")
        self._vec_store.add(ids, vecs)

    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        db = self._open_db(None)
        emb_ids: List[str] = []
        emb_vecs: List[List[float]] = []
        emb_metas: List[Dict[str, Any]] = []
        emb_docs: List[Optional[str]] = []
        for i, id_ in enumerate(ids):
            emb = embeddings[i] if embeddings else None
            meta = _sanitize_metadata(metadatas[i]) if metadatas else None
            doc = documents[i] if documents else None
            if emb is not None:
                emb_ids.append(id_)
                emb_vecs.append(emb)
                emb_metas.append(meta or {})
                emb_docs.append(doc)
            else:
                db.update_metadata(id_, meta, doc)
        if emb_ids:
            vecs = np.asarray(emb_vecs, dtype=np.float32)
            db.insert_batch(emb_ids, vecs, emb_metas, emb_docs, "update")
            self._vec_store.add(emb_ids, vecs)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        db = self._open_db(None)
        if ids and not where:
            db.delete_batch(ids)
            self._vec_store.remove(ids)
            return
        # Filter path: use list_ids for efficient Rust-side filtering
        if where:
            filter_ids = db.list_ids(where_filter=where)
            if ids:
                id_set = set(ids)
                filter_ids = [fid for fid in filter_ids if fid in id_set]
            if filter_ids:
                db.delete_batch(filter_ids)
                self._vec_store.remove(filter_ids)

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if include is not None:
            unknown = set(include) - _VALID_GET_INCLUDE
            if unknown:
                raise ValueError(f"Unknown include fields: {unknown}. Valid: {_VALID_GET_INCLUDE}")
        if not os.path.exists(os.path.join(self._path, "manifest.json")):
            return {"ids": [], "metadatas": [], "documents": [], "embeddings": None}
        include_set = set(include or ["metadatas", "documents"])
        db = self._open_db(None)

        # BUG-C7: an explicitly-passed empty list means "return nothing";
        # only fall back to full scan when ids is None (not provided at all).
        if ids is not None and len(ids) == 0:
            records = []
        elif ids is not None and len(ids) > 0:
            records = [r for r in db.get_many(ids) if r is not None]
            if where:
                records = _apply_filter(records, where)
        elif where:
            matched_ids = db.list_ids(where_filter=where)
            records = [r for r in db.get_many(matched_ids) if r is not None]
        else:
            records = self._list_all_records(db)

        records = records[offset:]
        if limit is not None:
            records = records[:limit]

        result_ids = [r["id"] for r in records]

        embeddings_out = None
        if "embeddings" in include_set:
            vec_map = self._vec_store.get_by_ids(result_ids)
            embeddings_out = [vec_map.get(id_) for id_ in result_ids]

        return {
            "ids": result_ids,
            "metadatas": [r.get("metadata") for r in records] if "metadatas" in include_set else None,
            "documents": [r.get("document") for r in records] if "documents" in include_set else None,
            "embeddings": embeddings_out,
        }

    def query(
        self,
        query_embeddings: Optional[List[List[float]]] = None,
        query_texts: Optional[List[str]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if where_document is not None:
            raise NotImplementedError("where_document is not supported (tqdb has no full-text index)")
        if include is not None:
            unknown = set(include) - _VALID_QUERY_INCLUDE
            if unknown:
                raise ValueError(f"Unknown include fields: {unknown}. Valid: {_VALID_QUERY_INCLUDE}")
        query_embeddings = self._embed(query_texts or [], query_embeddings)
        include_set = set(include or ["metadatas", "documents", "distances"])
        db = self._open_db(None)

        all_ids, all_distances, all_metas, all_docs, all_embeddings = [], [], [], [], []
        for qvec in query_embeddings:
            q = np.asarray(qvec, dtype=np.float32)
            results = db.search(q, n_results, filter=where)
            row_ids = [r["id"] for r in results]
            all_ids.append(row_ids)
            all_distances.append([r["score"] for r in results])
            all_metas.append([r.get("metadata") for r in results])
            all_docs.append([r.get("document") for r in results])
            if "embeddings" in include_set:
                vec_map = self._vec_store.get_by_ids(row_ids)
                all_embeddings.append([vec_map.get(id_) for id_ in row_ids])

        return {
            "ids": all_ids,
            "distances": all_distances if "distances" in include_set else None,
            "metadatas": all_metas if "metadatas" in include_set else None,
            "documents": all_docs if "documents" in include_set else None,
            "embeddings": all_embeddings if "embeddings" in include_set else None,
        }

    def peek(self, limit: int = 10) -> Dict[str, Any]:
        return self.get(limit=limit)

    def modify(self, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        meta_path = os.path.join(self._path, "_chroma_meta.json")
        info: Dict[str, Any] = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                info = json.load(f)
        if metadata is not None:
            info["metadata"] = metadata
            new_metric = _parse_metric(metadata)
            if new_metric != self._metric:
                raise ValueError(
                    "Cannot change metric after collection is created. "
                    "Delete and recreate the collection."
                )
        if name is not None and name != self._name:
            # Validate before computing new path to prevent path traversal.
            _validate_collection_name(name)
            # BUG-C8: rename the underlying directory so the change is durable
            self._db = None  # release open handles before rename
            parent = os.path.dirname(self._path)
            new_path = os.path.join(parent, name)
            os.rename(self._path, new_path)
            self._path = new_path
            self._name = name
            self._vec_store.relocate(new_path)
            info["name"] = name
            meta_path = os.path.join(new_path, "_chroma_meta.json")
        else:
            if name is not None:
                info["name"] = name
        with open(meta_path, "w") as f:
            json.dump(info, f)

    def _list_all_records(self, db: Database) -> List[Dict[str, Any]]:
        ids = db.list_all()
        if not ids:
            return []
        return [r for r in db.get_many(ids) if r is not None]


# ---------------------------------------------------------------------------
# CompatClient
# ---------------------------------------------------------------------------

class CompatClient:
    """ChromaDB-compatible client backed by tqdb. See ``PersistentClient``."""

    def __init__(self, path: str):
        self._path = os.path.abspath(path)
        os.makedirs(self._path, exist_ok=True)
        import threading
        self._lock = threading.Lock()

    def _meta_path(self, name: str) -> str:
        return os.path.join(self._path, name, "_chroma_meta.json")

    def _load_meta(self, name: str) -> Optional[Dict[str, Any]]:
        mp = self._meta_path(name)
        if os.path.exists(mp):
            with open(mp) as f:
                return json.load(f)
        return None

    def _save_meta(self, name: str, metadata: Optional[Dict[str, Any]], metric: str) -> None:
        col_dir = os.path.join(self._path, name)
        os.makedirs(col_dir, exist_ok=True)
        with open(self._meta_path(name), "w") as f:
            json.dump({"name": name, "metadata": metadata, "metric": metric}, f)

    def _collection_dir(self, name: str) -> str:
        return os.path.join(self._path, name)

    def _is_collection(self, name: str) -> bool:
        d = self._collection_dir(name)
        return os.path.isdir(d) and os.path.exists(self._meta_path(name))

    def create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_function=None,
        get_or_create: bool = False,
    ) -> "CompatCollection":
        _validate_collection_name(name)
        with self._lock:
            if self._is_collection(name):
                if get_or_create:
                    return self.get_collection(name, embedding_function=embedding_function)
                raise ValueError(f"Collection '{name}' already exists. Use get_or_create_collection().")
            metric = _parse_metric(metadata)
            self._save_meta(name, metadata, metric)
            return CompatCollection(self._collection_dir(name), name, metric, embedding_function)

    def get_or_create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_function=None,
    ) -> "CompatCollection":
        return self.create_collection(name, metadata=metadata, embedding_function=embedding_function, get_or_create=True)

    def get_collection(
        self,
        name: str,
        embedding_function=None,
    ) -> "CompatCollection":
        if not self._is_collection(name):
            raise ValueError(f"Collection '{name}' not found.")
        info = self._load_meta(name) or {}
        metric = info.get("metric", "ip")
        return CompatCollection(self._collection_dir(name), name, metric, embedding_function)

    def delete_collection(self, name: str) -> None:
        if not self._is_collection(name):
            raise ValueError(f"Collection '{name}' not found.")
        shutil.rmtree(self._collection_dir(name))

    def list_collections(self) -> List["CollectionInfo"]:
        """Return a sorted list of CollectionInfo objects (mirrors chromadb ≥ 0.4 API)."""
        results = []
        for entry in sorted(os.scandir(self._path), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            meta_file = os.path.join(entry.path, "_chroma_meta.json")
            if not os.path.exists(meta_file):
                continue
            try:
                with open(meta_file) as f:
                    info = json.load(f)
            except Exception:
                info = {}
            name = info.get("name", entry.name)
            cid = info.get("id") or str(uuid.uuid5(uuid.NAMESPACE_URL, "tqdb:" + entry.path))
            metadata = info.get("metadata")
            results.append(CollectionInfo(name=name, id=cid, metadata=metadata))
        return results

    def count_collections(self) -> int:
        return len(self.list_collections())

    def heartbeat(self) -> int:
        """Return current time in nanoseconds (mirrors chromadb.Client.heartbeat())."""
        import time
        return int(time.time() * 1e9)

    def reset(self) -> None:
        """Delete all collections under this path."""
        # Iterate directory entries directly (not metadata-provided names) so
        # reset() is robust against metadata corruption or name divergence.
        if not os.path.isdir(self._path):
            return
        for entry in os.scandir(self._path):
            if entry.is_dir():
                shutil.rmtree(entry.path, ignore_errors=True)


def PersistentClient(path: str, **_kwargs) -> CompatClient:
    """
    Factory matching ``chromadb.PersistentClient(path=...)`` signature.

    Extra keyword arguments (e.g. ``settings=``) are accepted and ignored.
    """
    return CompatClient(path)
