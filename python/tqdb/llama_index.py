"""LlamaIndex ``BasePydanticVectorStore`` implementation backed by tqdb.

Usage::

    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.schema import Document
    from tqdb.llama_index import TurboQuantVectorStore

    vstore = TurboQuantVectorStore.open("./mydb", dimension=1536)
    storage = StorageContext.from_defaults(vector_store=vstore)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage)

LlamaIndex is an optional dependency. The module imports it lazily and raises
a clear ``ImportError`` on first use if it isn't installed.
"""

from __future__ import annotations

import uuid
from typing import Any, List, Optional

import numpy as np

from .tqdb import Database
from ._filter_translator import llama_index_filters_to_mongo


def _require_llama_index():
    try:
        from llama_index.core.schema import BaseNode, MetadataMode, TextNode
        from llama_index.core.vector_stores.types import (
            BasePydanticVectorStore,
            VectorStoreQuery,
            VectorStoreQueryResult,
        )
    except ImportError as e:
        raise ImportError(
            "TurboQuantVectorStore for LlamaIndex requires the "
            "`llama-index-core` package. Install with: "
            "pip install 'tqdb[llamaindex]' or pip install llama-index-core"
        ) from e
    return (
        BaseNode,
        MetadataMode,
        TextNode,
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )


_BUILT: Optional[type] = None


def _build_class():
    global _BUILT
    if _BUILT is not None:
        return _BUILT
    (
        BaseNode,
        MetadataMode,
        TextNode,
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
    ) = _require_llama_index()

    class TurboQuantVectorStore(BasePydanticVectorStore):
        """LlamaIndex VectorStore backed by a TurboQuantDB ``Database``.

        Constructed directly with an open ``Database`` or via the ``open()``
        / ``from_path()`` helpers.

        Class flags:

        - ``stores_text = True`` — TQDB stores the document text alongside the
          vector, so retrieval round-trips full nodes.
        - ``flat_metadata = True`` — TQDB metadata indexes only scalar fields;
          callers who need nested objects should flatten before insert.
        """

        stores_text: bool = True
        flat_metadata: bool = True

        # Pydantic v2 needs to know the underlying Database is opaque.
        model_config = {"arbitrary_types_allowed": True}

        def __init__(self, db: Database, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            # Bypass pydantic field validation for the runtime-only db handle.
            object.__setattr__(self, "_db", db)

        # ── construction ─────────────────────────────────────────────────

        @classmethod
        def open(
            cls,
            path: str,
            **db_kwargs: Any,
        ) -> "TurboQuantVectorStore":
            return cls(Database.open(path, **db_kwargs))

        @property
        def client(self) -> Database:
            """LlamaIndex convention: expose the underlying client."""
            return self._db

        # ── add ──────────────────────────────────────────────────────────

        def add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
            if not nodes:
                return []
            ids: List[str] = []
            vecs: List[List[float]] = []
            metadatas: List[Optional[dict]] = []
            documents: List[Optional[str]] = []

            for node in nodes:
                if node.embedding is None:
                    raise ValueError(
                        f"node {getattr(node, 'node_id', '?')} has no embedding; "
                        "compute embeddings before calling add() or use the "
                        "VectorStoreIndex flow that populates them"
                    )
                node_id = node.node_id or str(uuid.uuid4())
                ids.append(node_id)
                vecs.append(node.embedding)
                # LlamaIndex separates metadata from "extra info"; we store the
                # main metadata dict and add `_node_type` so we can reconstruct
                # the node class on query.
                meta = dict(node.metadata or {})
                meta["_node_type"] = type(node).__name__
                metadatas.append(meta)
                # Use `MetadataMode.NONE` to get the raw text (no header injection).
                documents.append(
                    node.get_content(metadata_mode=MetadataMode.NONE) or None
                )

            arr = np.asarray(vecs, dtype=np.float32)
            self._db.insert_batch(ids, arr, metadatas, documents, "upsert")
            return ids

        # ── delete ───────────────────────────────────────────────────────

        def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
            """Delete by ``ref_doc_id`` (the LlamaIndex per-document handle)."""
            # LlamaIndex passes one ref_doc_id at a time; TQDB takes a list.
            self._db.delete_batch([ref_doc_id])

        def delete_nodes(
            self,
            node_ids: Optional[List[str]] = None,
            filters: Optional[Any] = None,
            **kwargs: Any,
        ) -> None:
            """Delete by explicit node IDs and/or a filter (LlamaIndex 0.10+ API)."""
            tqdb_filter = llama_index_filters_to_mongo(filters)
            if node_ids:
                self._db.delete_batch(node_ids)
            if tqdb_filter is not None:
                self._db.delete_batch(where_filter=tqdb_filter)

        def clear(self) -> None:
            """Delete every node in the store."""
            ids = self._db.list_all()
            if ids:
                self._db.delete_batch(ids)

        # ── query ────────────────────────────────────────────────────────

        def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
            qvec = query.query_embedding
            if qvec is None:
                raise ValueError(
                    "TurboQuantVectorStore.query requires "
                    "VectorStoreQuery.query_embedding to be set"
                )
            top_k = query.similarity_top_k or 4
            tqdb_filter = llama_index_filters_to_mongo(query.filters)

            kwargs_: dict = {"top_k": top_k}
            if tqdb_filter is not None:
                kwargs_["filter"] = tqdb_filter
            results = self._db.search(np.asarray(qvec, dtype=np.float32), **kwargs_)

            ids: List[str] = []
            similarities: List[float] = []
            nodes: List[BaseNode] = []
            for r in results:
                node_id = r["id"]
                ids.append(node_id)
                similarities.append(float(r.get("score", 0.0)))
                meta = dict(r.get("metadata") or {})
                meta.pop("_node_type", None)
                nodes.append(TextNode(
                    id_=node_id,
                    text=r.get("document") or "",
                    metadata=meta,
                ))
            return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

        # ── persistence ──────────────────────────────────────────────────

        def persist(
            self,
            persist_path: Optional[str] = None,
            fs: Optional[Any] = None,
        ) -> None:
            """LlamaIndex calls ``persist`` on store snapshots; TQDB writes
            through on every operation, so this is just a checkpoint."""
            self._db.checkpoint()

    _BUILT = TurboQuantVectorStore
    return _BUILT


def __getattr__(name: str):
    if name == "TurboQuantVectorStore":
        return _build_class()
    raise AttributeError(name)
