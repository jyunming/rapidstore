"""ColBERT-style multi-vector documents for TurboQuantDB.

Each document gets N token vectors; queries are also tokenized to K vectors.
Scoring is MaxSim::

    score(q, d) = Σ_i max_j <q_i, d_j>

This module is a **Python-layer wrapper** over the existing single-vector
``Database``. Token vectors are stored as regular slots in the underlying
TQDB; a JSON sidecar tracks the ``doc_id → [token_id, ...]`` mapping, and a
``_VecStore`` keeps raw float32 token vectors for exact MaxSim scoring.

Future v0.9 release will move multi-vector into the engine for tighter
storage and native filter pushdown — this module's public API is designed
to stay stable across that move.

Quick start::

    from tqdb import Database
    from tqdb.multivector import MultiVectorStore

    db = Database.open("./mvstore", dimension=96, bits=4, metric="cosine")
    store = MultiVectorStore(db)

    # Insert a document with N=8 token vectors and the source text.
    store.insert("doc1", token_vectors_8x96, document="full text here")

    # Query with K query-token vectors → MaxSim ranking over candidate docs.
    hits = store.search(query_token_vectors_K_x_96, top_k=10)
    # hits: list of {"doc_id": str, "score": float, "document": str | None}

The wrapper keeps every TQDB primitive untouched so existing single-vector
databases coexist with multi-vector ones in the same process — just give
each its own directory.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .tqdb import Database


# ── per-token sidecar (raw float32) ─────────────────────────────────────


class _RawTokenVecStore:
    """Per-token raw float32 store, persisted as ``_mv_tokens.npz``.

    Same shape as the existing ``_VecStore`` in ``chroma_compat.py``, but
    parameterised on a custom filename so multivector and chroma compat can
    coexist in the same directory if a user ever does both.
    """

    def __init__(self, directory: str, filename: str = "_mv_tokens.npz"):
        self._path = os.path.join(directory, filename)
        self._lock = threading.Lock()

    def _load(self) -> Tuple[List[str], Optional[np.ndarray]]:
        if not os.path.exists(self._path):
            return [], None
        with np.load(self._path, allow_pickle=False) as data:
            return data["ids"].tolist(), data["vecs"]

    def _save(self, ids: List[str], vecs: np.ndarray) -> None:
        np.savez(self._path, ids=np.array(ids, dtype=str), vecs=vecs.astype(np.float32))

    def add(self, new_ids: List[str], new_vecs: np.ndarray) -> None:
        with self._lock:
            ids, vecs = self._load()
            if ids:
                new_set = set(new_ids)
                keep = [i for i, id_ in enumerate(ids) if id_ not in new_set]
                ids = [ids[i] for i in keep]
                vecs = (
                    vecs[keep] if (vecs is not None and keep)
                    else np.empty((0, new_vecs.shape[1]), dtype=np.float32)
                )
            else:
                vecs = np.empty((0, new_vecs.shape[1]), dtype=np.float32)
            all_ids = ids + list(new_ids)
            all_vecs = np.concatenate([vecs, new_vecs.astype(np.float32)], axis=0)
            self._save(all_ids, all_vecs)

    def remove(self, del_ids: List[str]) -> None:
        with self._lock:
            ids, vecs = self._load()
            if not ids or vecs is None:
                return
            del_set = set(del_ids)
            keep = [i for i, id_ in enumerate(ids) if id_ not in del_set]
            if not keep:
                if os.path.exists(self._path):
                    os.remove(self._path)
                return
            self._save([ids[i] for i in keep], vecs[keep])

    def get_many(self, ids: Iterable[str]) -> Dict[str, np.ndarray]:
        with self._lock:
            all_ids, vecs = self._load()
        if not all_ids or vecs is None:
            return {}
        index = {id_: i for i, id_ in enumerate(all_ids)}
        out: Dict[str, np.ndarray] = {}
        for id_ in ids:
            if id_ in index:
                out[id_] = vecs[index[id_]]
        return out


# ── doc_id ↔ token_ids index ────────────────────────────────────────────


class _DocIndex:
    """Maps ``doc_id`` to its list of internal token_ids. Persisted as JSON.

    JSON keeps the file human-debuggable; for very large stores we'd swap
    to msgpack or bincode, but at multi-vector scale a few k-MB JSON is fine.
    """

    def __init__(self, directory: str, filename: str = "_mv_index.json"):
        self._path = os.path.join(directory, filename)
        self._lock = threading.Lock()
        self._mapping: Dict[str, List[str]] = self._load()

    def _load(self) -> Dict[str, List[str]]:
        if not os.path.exists(self._path):
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Validate shape — corrupt sidecars become empty rather than crash.
                if isinstance(data, dict) and all(
                    isinstance(v, list) for v in data.values()
                ):
                    return data
        except (json.JSONDecodeError, OSError):
            pass
        return {}

    def _save(self) -> None:
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._mapping, f, separators=(",", ":"))
        os.replace(tmp, self._path)  # atomic on POSIX + modern Windows

    def set(self, doc_id: str, token_ids: List[str]) -> None:
        with self._lock:
            self._mapping[doc_id] = token_ids
            self._save()

    def remove(self, doc_id: str) -> Optional[List[str]]:
        with self._lock:
            tids = self._mapping.pop(doc_id, None)
            if tids is not None:
                self._save()
            return tids

    def get(self, doc_id: str) -> Optional[List[str]]:
        return self._mapping.get(doc_id)

    def doc_ids(self) -> List[str]:
        return list(self._mapping.keys())

    def token_to_doc(self) -> Dict[str, str]:
        """Inverse map (built on demand) — one entry per token across all docs."""
        return {tid: did for did, tids in self._mapping.items() for tid in tids}

    def __len__(self) -> int:
        return len(self._mapping)


# ── public API ──────────────────────────────────────────────────────────


class MultiVectorStore:
    """ColBERT-style late-interaction store layered over a TQDB ``Database``.

    Args:
        db: An open ``Database``. Recommended ``metric="cosine"`` and
            ``bits=4`` for unit-norm token embeddings (e.g. ColBERTv2 d=96).
        directory: Where to put the sidecars (``_mv_tokens.npz``,
            ``_mv_index.json``). Defaults to the database's local directory
            inferred from ``db.stats()["disk_path"]`` when present, else the
            current working directory.
    """

    def __init__(self, db: Database, directory: str) -> None:
        self._db = db
        Path(directory).mkdir(parents=True, exist_ok=True)
        self._dir = directory
        self._raw = _RawTokenVecStore(directory)
        self._index = _DocIndex(directory)

    @classmethod
    def open(
        cls,
        path: str,
        dimension: int,
        *,
        bits: int = 4,
        metric: str = "cosine",
        **db_kwargs: Any,
    ) -> "MultiVectorStore":
        """Open or create both the underlying Database and the multivector
        sidecars at ``path``."""
        db = Database.open(path, dimension=dimension, bits=bits, metric=metric, **db_kwargs)
        return cls(db, directory=path)

    # ── insert ──────────────────────────────────────────────────────────

    def insert(
        self,
        doc_id: str,
        vectors: np.ndarray,
        document: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Insert ``vectors`` (shape ``(N, D)``) as the token set for ``doc_id``.

        If ``doc_id`` already exists, it is replaced — old token vectors are
        deleted from both the engine and the sidecar before inserting new ones.
        Returns the list of internal token IDs that were stored.
        """
        if isinstance(vectors, list):
            vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(
                f"vectors must be 2-D (N, D); got shape {vectors.shape}"
            )
        n = vectors.shape[0]
        if n == 0:
            raise ValueError("must provide at least one token vector per doc")

        # Replace semantics: drop stale token IDs first.
        old = self._index.remove(doc_id)
        if old:
            self._db.delete_batch(old)
            self._raw.remove(old)

        # Generate fresh token IDs. UUIDs guarantee uniqueness without
        # depending on doc_id formatting (no separator-collision risks).
        token_ids = [f"_mv_{uuid.uuid4().hex[:16]}" for _ in range(n)]

        meta_base: Dict[str, Any] = dict(metadata or {})
        meta_base["_mv_doc"] = doc_id  # so list_ids({"_mv_doc": doc_id}) works
        metas = [meta_base] * n
        docs = [document] * n if document is not None else None

        self._db.insert_batch(token_ids, vectors.astype(np.float32), metas, docs, "upsert")
        self._raw.add(token_ids, vectors.astype(np.float32))
        self._index.set(doc_id, token_ids)
        return token_ids

    def insert_many(
        self,
        items: Sequence[Tuple[str, np.ndarray, Optional[str], Optional[Dict[str, Any]]]],
    ) -> None:
        """Bulk-insert. Each item is ``(doc_id, vectors, document, metadata)``.

        Calls ``insert`` per item; could be optimized to batch the engine
        round-trips but keeps the implementation simple for v0.8.
        """
        for doc_id, vectors, document, metadata in items:
            self.insert(doc_id, vectors, document, metadata)

    # ── delete ──────────────────────────────────────────────────────────

    def delete(self, doc_id: str) -> bool:
        """Delete ``doc_id`` and its token vectors. Returns False if unknown."""
        token_ids = self._index.remove(doc_id)
        if not token_ids:
            return False
        self._db.delete_batch(token_ids)
        self._raw.remove(token_ids)
        return True

    # ── search ──────────────────────────────────────────────────────────

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 10,
        *,
        oversample: int = 4,
        candidate_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run MaxSim retrieval for ``query_vectors`` (shape ``(K, D)``).

        Steps:

        1. **Candidate generation** — fan out one TQDB ``search`` per query
           token, asking each for ``oversample × top_k`` token-level hits.
        2. **Doc-level grouping** — union all hit tokens, derive their
           doc_ids via the sidecar.
        3. **Exact MaxSim scoring** — for each candidate doc, fetch all its
           raw token vectors, compute ``Σ_i max_j <q_i, d_j>`` in NumPy.
        4. **Top-k by fused score** — sort and slice.

        ``candidate_filter`` is forwarded to the underlying engine search to
        restrict the candidate-token universe (e.g. metadata predicates).
        """
        if isinstance(query_vectors, list):
            query_vectors = np.asarray(query_vectors, dtype=np.float32)
        if query_vectors.ndim != 2:
            raise ValueError(
                f"query_vectors must be 2-D (K, D); got {query_vectors.shape}"
            )
        if top_k <= 0 or query_vectors.shape[0] == 0:
            return []
        if len(self._index) == 0:
            return []

        per_q_top = max(top_k * oversample, top_k)
        token_to_doc = self._index.token_to_doc()

        # Candidate generation: union of all query-token top hits.
        candidate_docs: set[str] = set()
        for qv in query_vectors:
            kwargs: Dict[str, Any] = {"top_k": per_q_top}
            if candidate_filter is not None:
                kwargs["filter"] = candidate_filter
            results = self._db.search(qv.astype(np.float32), **kwargs)
            for r in results:
                doc_id = token_to_doc.get(r["id"])
                if doc_id is not None:
                    candidate_docs.add(doc_id)

        if not candidate_docs:
            return []

        # Exact MaxSim per candidate. Fetch all of each doc's token vectors
        # in one batch to amortise the file-read cost.
        scored: List[Tuple[str, float]] = []
        all_tokens = [tid for did in candidate_docs for tid in (self._index.get(did) or [])]
        token_vecs = self._raw.get_many(all_tokens)

        q = query_vectors.astype(np.float32)  # (K, D)
        for doc_id in candidate_docs:
            tids = self._index.get(doc_id) or []
            d_vecs = [token_vecs[t] for t in tids if t in token_vecs]
            if not d_vecs:
                continue
            d = np.stack(d_vecs).astype(np.float32)  # (N, D)
            sims = q @ d.T  # (K, N)
            max_per_query = sims.max(axis=1)  # (K,)
            score = float(max_per_query.sum())
            scored.append((doc_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        # Hydrate result rows with the document text from the engine (one
        # representative token row per doc carries the same document string).
        out: List[Dict[str, Any]] = []
        for doc_id, score in scored:
            tids = self._index.get(doc_id) or []
            document: Optional[str] = None
            metadata: Dict[str, Any] = {}
            if tids:
                rec = self._db.get(tids[0])
                if rec is not None:
                    document = rec.get("document")
                    metadata = {
                        k: v for k, v in (rec.get("metadata") or {}).items()
                        if k != "_mv_doc"
                    }
            out.append({
                "doc_id": doc_id,
                "score": score,
                "document": document,
                "metadata": metadata,
            })
        return out

    # ── inspection / lifecycle ──────────────────────────────────────────

    def doc_ids(self) -> List[str]:
        """All currently indexed doc_ids."""
        return self._index.doc_ids()

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, doc_id: str) -> bool:
        return self._index.get(doc_id) is not None

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Return ``{"doc_id", "n_tokens", "document", "metadata"}`` for ``doc_id``,
        or ``None`` if unknown."""
        tids = self._index.get(doc_id)
        if not tids:
            return None
        rec = self._db.get(tids[0]) or {}
        return {
            "doc_id": doc_id,
            "n_tokens": len(tids),
            "document": rec.get("document"),
            "metadata": {
                k: v for k, v in (rec.get("metadata") or {}).items()
                if k != "_mv_doc"
            },
        }
