"""LangChain v2 ``VectorStore`` implementation backed by ``tqdb.Database``.

Usage::

    from langchain_openai import OpenAIEmbeddings
    from tqdb.vectorstore import TurboQuantVectorStore

    embed = OpenAIEmbeddings()
    store = TurboQuantVectorStore.from_texts(
        texts=["doc 1", "doc 2"],
        embedding=embed,
        path="./mydb",
    )
    docs = store.similarity_search("query", k=4)

The class implements the full LangChain v2 ``VectorStore`` ABC: ``add_texts``,
``add_documents``, ``similarity_search``, ``similarity_search_with_score``,
``similarity_search_by_vector``, ``delete``, ``from_texts``,
``from_documents``, ``as_retriever``, ``_select_relevance_score_fn``,
``get_by_ids``.

LangChain itself is an optional dependency: this module imports it lazily and
raises a clear ``ImportError`` on first use if it isn't installed.
"""

from __future__ import annotations

import uuid
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .tqdb import Database
from ._filter_translator import langchain_filter_to_mongo


def _require_langchain():
    try:
        from langchain_core.documents import Document
        from langchain_core.embeddings import Embeddings
        from langchain_core.vectorstores import VectorStore
    except ImportError as e:
        raise ImportError(
            "TurboQuantVectorStore requires the `langchain-core` package. "
            "Install with: pip install 'tqdb[langchain]' "
            "or pip install langchain-core"
        ) from e
    return Document, Embeddings, VectorStore


# Bind the LangChain types lazily by building the class inside a factory.
# The first call materialises the real subclass; subsequent calls return it
# from cache so the user always sees the same identity (`isinstance` works).

_BUILT: Optional[type] = None


def _build_class():
    global _BUILT
    if _BUILT is not None:
        return _BUILT
    Document, Embeddings, VectorStore = _require_langchain()

    class TurboQuantVectorStore(VectorStore):
        """LangChain VectorStore backed by a TurboQuantDB ``Database``.

        Construction options:

        - ``TurboQuantVectorStore(db, embedding=...)`` wraps an already-open ``Database``.
        - ``TurboQuantVectorStore.open(path, embedding=..., dimension=...)`` opens / creates one.
        - ``TurboQuantVectorStore.from_texts(texts, embedding, path=...)`` builds + populates one.
        """

        def __init__(
            self,
            db: Database,
            embedding: Optional["Embeddings"] = None,
        ) -> None:
            self._db = db
            self._embedding = embedding

        # ── construction helpers ─────────────────────────────────────────

        @classmethod
        def open(
            cls,
            path: str,
            embedding: Optional["Embeddings"] = None,
            **db_kwargs: Any,
        ) -> "TurboQuantVectorStore":
            """Open or create the underlying database and wrap it."""
            return cls(Database.open(path, **db_kwargs), embedding=embedding)

        @classmethod
        def from_texts(
            cls,
            texts: List[str],
            embedding: "Embeddings",
            metadatas: Optional[List[dict]] = None,
            *,
            path: str = "./tqdb_store",
            ids: Optional[List[str]] = None,
            bits: int = 4,
            metric: str = "cosine",
            **kwargs: Any,
        ) -> "TurboQuantVectorStore":
            """Build a fresh store from raw texts. ``embedding`` is required.

            Extra ``**kwargs`` are forwarded to ``Database.open`` so callers can
            pass DB-side options (``rerank``, ``fast_mode``, ``normalize``,
            ``rerank_precision``, ``wal_flush_threshold``, ``quantizer_type``,
            ``seed``) through the LangChain constructor path.
            """
            if not texts:
                raise ValueError("from_texts requires at least one text")
            vecs = embedding.embed_documents(list(texts))
            arr = np.asarray(vecs, dtype=np.float32)
            d = arr.shape[1]
            db = Database.open(path, dimension=d, bits=bits, metric=metric, **kwargs)
            store = cls(db, embedding=embedding)
            store.add_texts(list(texts), metadatas=metadatas, ids=ids, _vectors=arr)
            return store

        @classmethod
        def from_documents(
            cls,
            documents: List["Document"],
            embedding: "Embeddings",
            **kwargs: Any,
        ) -> "TurboQuantVectorStore":
            texts = [d.page_content for d in documents]
            metadatas = [d.metadata for d in documents]
            return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)

        # ── add ──────────────────────────────────────────────────────────

        def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            *,
            ids: Optional[List[str]] = None,
            _vectors: Optional[np.ndarray] = None,
            **kwargs: Any,
        ) -> List[str]:
            """Insert texts into the store. ``embedding`` is consulted unless
            ``_vectors`` (a private fast-path) is provided."""
            texts = list(texts)
            if not texts:
                return []
            if _vectors is None:
                if self._embedding is None:
                    raise ValueError(
                        "add_texts needs an embedding function — pass `embedding=` "
                        "to the constructor or supply pre-computed vectors via "
                        "the engine API directly."
                    )
                vecs = np.asarray(
                    self._embedding.embed_documents(texts), dtype=np.float32
                )
            else:
                vecs = _vectors
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in texts]
            self._db.insert_batch(
                ids,
                vecs,
                metadatas if metadatas is not None else [None] * len(texts),
                texts,
                "upsert",
            )
            return ids

        def add_documents(
            self,
            documents: List["Document"],
            ids: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> List[str]:
            return self.add_texts(
                [d.page_content for d in documents],
                metadatas=[d.metadata for d in documents],
                ids=ids,
                **kwargs,
            )

        # ── delete / read ────────────────────────────────────────────────

        def delete(
            self,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> Optional[bool]:
            if ids is None:
                return None
            n = self._db.delete_batch(ids)
            return n > 0

        def get_by_ids(self, ids: Sequence[str]) -> List["Document"]:
            Document, _, _ = _require_langchain()
            recs = self._db.get_many(list(ids))
            out = []
            for rec in recs:
                if rec is None:
                    continue
                out.append(Document(
                    id=rec["id"],
                    page_content=rec.get("document") or "",
                    metadata=rec.get("metadata") or {},
                ))
            return out

        # ── search ───────────────────────────────────────────────────────

        def similarity_search(
            self,
            query: str,
            k: int = 4,
            filter: Optional[dict] = None,
            **kwargs: Any,
        ) -> List["Document"]:
            return [doc for doc, _ in self.similarity_search_with_score(
                query, k=k, filter=filter, **kwargs
            )]

        def similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
            filter: Optional[dict] = None,
            **kwargs: Any,
        ) -> List[Tuple["Document", float]]:
            if self._embedding is None:
                raise ValueError(
                    "similarity_search needs an embedding function — pass "
                    "`embedding=` to the constructor or use "
                    "`similarity_search_by_vector`."
                )
            qvec = np.asarray(
                self._embedding.embed_query(query), dtype=np.float32
            )
            return self._search_with_score_by_vector(qvec, k=k, filter=filter, **kwargs)

        def similarity_search_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[dict] = None,
            **kwargs: Any,
        ) -> List["Document"]:
            qvec = np.asarray(embedding, dtype=np.float32)
            return [doc for doc, _ in self._search_with_score_by_vector(
                qvec, k=k, filter=filter, **kwargs
            )]

        def _search_with_score_by_vector(
            self,
            qvec: np.ndarray,
            k: int,
            filter: Optional[dict],
            **kwargs: Any,
        ) -> List[Tuple["Document", float]]:
            Document, _, _ = _require_langchain()
            tqdb_filter = langchain_filter_to_mongo(filter)
            search_kwargs: dict = {"top_k": k}
            if tqdb_filter is not None:
                search_kwargs["filter"] = tqdb_filter
            for k_ in ("hybrid", "rerank_factor", "_use_ann", "ann_search_list_size", "nprobe"):
                if k_ in kwargs:
                    search_kwargs[k_] = kwargs[k_]
            results = self._db.search(qvec, **search_kwargs)
            out: List[Tuple["Document", float]] = []
            for r in results:
                doc = Document(
                    id=r["id"],
                    page_content=r.get("document") or "",
                    metadata=r.get("metadata") or {},
                )
                out.append((doc, float(r.get("score", 0.0))))
            return out

        # ── score normalisation ──────────────────────────────────────────

        def _select_relevance_score_fn(self):
            """Map TQDB's IP / cosine / L2 score to LangChain's [0, 1] relevance.

            For cosine and inner product on unit vectors, the score is already
            in roughly [-1, 1]; rescale to [0, 1]. For L2 (lower-is-better),
            invert.
            """
            metric = self._db.stats().get("metric", "ip")
            if metric == "l2":
                return lambda s: 1.0 / (1.0 + max(s, 0.0))
            # ip / cosine: score in [-1, 1] for unit vectors → [0, 1].
            return lambda s: max(0.0, min(1.0, (float(s) + 1.0) / 2.0))

        @property
        def embeddings(self) -> Optional["Embeddings"]:
            return self._embedding

    _BUILT = TurboQuantVectorStore
    return _BUILT


def __getattr__(name: str):
    """PEP 562 module-level ``__getattr__``: lazy-load the class on first access.

    Lets users do ``from tqdb.vectorstore import TurboQuantVectorStore`` without
    paying the LangChain import cost until they actually need the class.
    """
    if name == "TurboQuantVectorStore":
        return _build_class()
    raise AttributeError(name)
