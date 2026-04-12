"""LangChain-style retriever wrapper around TurboQuantDB."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

try:
    from .tqdb import Database
except ImportError:
    class Database:  # type: ignore
        @staticmethod
        def open(
            path: str,
            dimension: int,
            bits: int = 4,
            seed: int = 42,
            metric: str = "ip",
            rerank: bool = False,
            fast_mode: bool = True,
            rerank_precision: Optional[str] = None,
        ):
            raise RuntimeError("tqdb extension not available")


# ---------------------------------------------------------------------------
# Document — minimal LangChain-compatible document type
# ---------------------------------------------------------------------------

try:
    from langchain_core.documents import Document  # type: ignore
except ImportError:
    try:
        from langchain.schema import Document  # type: ignore
    except ImportError:
        class Document:  # type: ignore
            """Minimal LangChain Document stub (used when langchain is not installed)."""

            def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
                self.page_content = page_content
                self.metadata = metadata or {}

            def __repr__(self) -> str:
                snippet = self.page_content[:60].replace("\n", " ")
                return f"Document(page_content={snippet!r}, metadata={self.metadata!r})"

            def __eq__(self, other: object) -> bool:
                if isinstance(other, Document):
                    return self.page_content == other.page_content and self.metadata == other.metadata
                return NotImplemented


# ---------------------------------------------------------------------------
# TurboQuantRetriever
# ---------------------------------------------------------------------------

class TurboQuantRetriever:
    """Retriever wrapper around TurboQuantDB for use in RAG pipelines.

    Implements the LangChain VectorStore + BaseRetriever interface surface,
    including LCEL ``invoke()``, ``similarity_search()``,
    ``similarity_search_with_score()``, ``add_documents()``, and more.

    Args:
        db_path: Directory where the TurboQuantDB files are stored.
        dimension: Embedding dimension (must match your embedding model).
        bits: Bits per vector (2 or 4). Higher = better recall, more disk.
        seed: Random seed for reproducibility.
        metric: Distance metric — ``"ip"`` (inner product) or ``"l2"``.
        rerank_precision: Optional rerank dtype (``"int8"`` or ``"f16"``).
        fast_mode: When ``True`` (default), MSE-only quantization — fastest
            ingest.  Set ``False`` to enable QJL residuals for +5–15 pp R@1
            at d ≥ 1536.
        embedding_function: Optional callable ``(texts: List[str]) ->
            List[List[float]]`` used by ``invoke()``,
            ``get_relevant_documents()``, ``add_documents()``, and
            ``from_texts()``.
    """

    def __init__(
        self,
        db_path: str,
        dimension: int = 1536,
        bits: int = 4,
        seed: int = 42,
        metric: str = "ip",
        rerank_precision: Optional[str] = None,
        fast_mode: bool = True,
        embedding_function=None,
    ):
        self.db = Database.open(
            db_path, dimension, bits=bits, seed=seed, metric=metric,
            rerank_precision=rerank_precision, fast_mode=fast_mode,
        )
        self.doc_store: Dict[str, Dict[str, Any]] = {}
        self.embedding_function = embedding_function

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if self.embedding_function is None:
            raise ValueError(
                "embedding_function is required for text-based queries. "
                "Pass embedding_function= when constructing TurboQuantRetriever."
            )
        result = self.embedding_function(texts)
        return np.asarray(result, dtype=np.float32)

    def _results_to_documents(
        self, results: List[Dict[str, Any]]
    ) -> List[Document]:
        docs = []
        for r in results:
            doc_id = r["id"] if isinstance(r, dict) else r[0]
            text = ""
            metadata: Dict[str, Any] = {}
            if doc_id in self.doc_store:
                entry = self.doc_store[doc_id]
                text = entry.get("text", "")
                metadata = dict(entry.get("metadata", {}))
            # Merge any document field stored directly in tqdb
            if isinstance(r, dict):
                if r.get("document") and not text:
                    text = r["document"]
                if r.get("metadata"):
                    metadata.update(r["metadata"])
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Insert texts and their embeddings into the database.

        Args:
            texts: Raw text strings to store.
            embeddings: Corresponding embedding vectors (one per text).
            metadatas: Optional per-document metadata dicts.
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        start = len(self.doc_store)
        ids = [f"doc_{start + i}" for i in range(len(texts))]
        for i, doc_id in enumerate(ids):
            self.doc_store[doc_id] = {"text": texts[i], "metadata": metadatas[i]}

        vectors = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
        self.db.insert_batch(ids, vectors, metadatas, texts, "insert")

    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> List[str]:
        """Insert LangChain ``Document`` objects into the database.

        Args:
            documents: List of ``Document`` with ``page_content`` and ``metadata``.
            embeddings: Pre-computed vectors. If omitted, ``embedding_function``
                is called to compute them.

        Returns:
            List of inserted IDs.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        if embeddings is None:
            vecs = self._embed_texts(texts)
            emb_list = vecs.tolist()
        else:
            emb_list = embeddings

        start = len(self.doc_store)
        ids = [f"doc_{start + i}" for i in range(len(texts))]
        for i, doc_id in enumerate(ids):
            self.doc_store[doc_id] = {"text": texts[i], "metadata": metadatas[i]}

        vectors = np.ascontiguousarray(np.asarray(emb_list, dtype=np.float32))
        self.db.insert_batch(ids, vectors, metadatas, texts, "insert")
        return ids

    def delete(self, ids: List[str]) -> int:
        """Delete documents by ID. Returns the number of documents deleted."""
        for id_ in ids:
            self.doc_store.pop(id_, None)
        return self.db.delete_batch(ids)

    # ------------------------------------------------------------------
    # Search API
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Return the top-k most similar documents as ``Document`` objects.

        Args:
            query_embedding: Query vector (same dimension as the database).
            k: Number of results to return.
            filter: Optional metadata filter dict (MongoDB-style operators).

        Returns:
            List of ``Document`` objects ordered by similarity.
        """
        vec = np.array(query_embedding, dtype=np.float32)
        results = self.db.search(vec, k, filter=filter)
        return self._results_to_documents(results)

    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return top-k results as ``(Document, score)`` tuples.

        Args:
            query_embedding: Query vector.
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of ``(Document, score)`` tuples.
        """
        vec = np.array(query_embedding, dtype=np.float32)
        results = self.db.search(vec, k, filter=filter)
        docs = self._results_to_documents(results)
        scores = [r["score"] if isinstance(r, dict) else r[1] for r in results]
        return list(zip(docs, scores))

    # ------------------------------------------------------------------
    # LangChain BaseRetriever + LCEL Runnable interface
    # ------------------------------------------------------------------

    def get_relevant_documents(
        self, query: str, *, k: int = 4, filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Legacy LangChain ``BaseRetriever.get_relevant_documents()`` interface.

        Embeds ``query`` using ``self.embedding_function``, then calls
        ``similarity_search()``. Falls back to a case-insensitive text scan
        over stored documents when no ``embedding_function`` is set.
        """
        if self.embedding_function is None:
            # Fallback: text scan over doc_store (useful in tests / offline mode)
            q_lower = query.lower()
            matches = [
                Document(page_content=entry["text"], metadata=dict(entry.get("metadata", {})))
                for entry in self.doc_store.values()
                if q_lower in entry.get("text", "").lower()
            ]
            return matches[:k]
        vecs = self._embed_texts([query])
        return self.similarity_search(vecs[0].tolist(), k=k, filter=filter)

    def invoke(
        self,
        input: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        """LCEL ``Runnable.invoke()`` interface — accepts a query string.

        Args:
            input: Query string to embed and search.
            config: Ignored (LangChain RunnableConfig).

        Returns:
            List of relevant ``Document`` objects.
        """
        k = kwargs.pop("k", 4)
        filter = kwargs.pop("filter", None)
        return self.get_relevant_documents(input, k=k, filter=filter)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding=None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        db_path: str = "./tqdb_rag",
        dimension: Optional[int] = None,
        bits: int = 4,
        metric: str = "ip",
        embeddings: Optional[List[List[float]]] = None,
        **kwargs,
    ) -> "TurboQuantRetriever":
        """Create a ``TurboQuantRetriever`` from a list of texts.

        Args:
            texts: List of text strings to embed and store.
            embedding: Callable or LangChain ``Embeddings`` object with an
                ``embed_documents(texts)`` method. Either this or ``embeddings``
                must be provided.
            metadatas: Optional per-document metadata dicts.
            db_path: Storage directory for the tqdb database.
            dimension: Embedding dimension. Inferred from the first batch if
                omitted.
            bits: Quantization bits (2 or 4).
            metric: Distance metric.
            embeddings: Pre-computed vectors (alternative to ``embedding``
                callable). Useful in tests or when embeddings are already
                available.
            **kwargs: Forwarded to ``TurboQuantRetriever.__init__``.

        Returns:
            A populated ``TurboQuantRetriever`` instance.
        """
        embed_fn = None
        if embeddings is not None:
            # Pre-computed vectors provided directly
            raw_vecs = embeddings
        elif embedding is not None:
            # Support both raw callables and LangChain Embeddings objects
            if hasattr(embedding, "embed_documents"):
                raw_vecs = embedding.embed_documents(texts)
                embed_fn = embedding.embed_documents
            else:
                raw_vecs = embedding(texts)
                embed_fn = embedding
        else:
            raise ValueError("Either embedding (callable) or embeddings (list of vectors) must be provided.")

        vecs = np.asarray(raw_vecs, dtype=np.float32)
        if dimension is None:
            dimension = vecs.shape[1]

        retriever = cls(
            db_path=db_path,
            dimension=dimension,
            bits=bits,
            metric=metric,
            embedding_function=embed_fn,
            **kwargs,
        )
        retriever.add_texts(texts, vecs.tolist(), metadatas)
        return retriever

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def as_retriever(self, **kwargs) -> "TurboQuantRetriever":
        """Return self (already a retriever). Accepts and ignores kwargs."""
        return self

    def query(self, query_embeddings: np.ndarray, n_results: int = 10, where_filter=None) -> List[List[Dict[str, Any]]]:
        """Batch search — returns ``list[list[dict]]`` (one list per query).

        Args:
            query_embeddings: ``np.ndarray`` of shape ``(N, D)``.
            n_results: Number of results per query.
            where_filter: Optional metadata filter.
        """
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        out = []
        for q in query_embeddings:
            results = self.db.search(q.astype(np.float32), n_results, filter=where_filter)
            row = []
            for r in results:
                doc_id = r["id"]
                score = r["score"]
                text = ""
                if doc_id in self.doc_store:
                    text = self.doc_store[doc_id].get("text", "")
                elif r.get("document"):
                    text = r["document"]
                row.append({"id": doc_id, "score": score, "text": text})
            out.append(row)
        return out
