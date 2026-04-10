"""LangChain-style retriever wrapper around TurboQuantDB."""

import numpy as np
from typing import Any, Dict, List, Optional

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
            rerank: bool = True,
            fast_mode: bool = False,
            rerank_precision: Optional[str] = None,
        ):
            raise RuntimeError("tqdb extension not available")


class TurboQuantRetriever:
    """Retriever wrapper around TurboQuantDB for use in RAG pipelines.

    Stores text documents alongside their embeddings and exposes a
    ``similarity_search`` interface compatible with LangChain-style chains.

    Args:
        db_path: Directory where the TurboQuantDB files are stored.
        dimension: Embedding dimension (must match your embedding model).
        bits: Bits per vector (2 or 4). Higher = better recall, more disk.
        seed: Random seed for reproducibility.
        metric: Distance metric — ``"ip"`` (inner product) or ``"l2"``.
        rerank_precision: Optional rerank dtype (``"f32"`` or ``"f16"``).
        fast_mode: When False (default), QJL residual is stored and used
            during reranking for best recall. Set True for ~30% faster ingest
            (MSE-only, no recall benefit from rerank).
    """

    def __init__(
        self,
        db_path: str,
        dimension: int = 1536,
        bits: int = 4,
        seed: int = 42,
        metric: str = "ip",
        rerank_precision: Optional[str] = None,
        fast_mode: bool = False,
    ):
        self.db = Database.open(
            db_path, dimension, bits=bits, seed=seed, metric=metric,
            rerank_precision=rerank_precision, fast_mode=fast_mode,
        )
        self.doc_store: Dict[str, Dict[str, Any]] = {}

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
        if hasattr(self.db, "insert_batch"):
            self.db.insert_batch(ids, vectors, metadatas, texts, "insert")
            return
        if hasattr(self.db, "insert_many"):
            self.db.insert_many(ids, vectors, metadatas, texts, "insert")
            return

        for i, doc_id in enumerate(ids):
            self.db.insert(doc_id, vectors[i], metadatas[i], texts[i])

    def similarity_search(self, query_embedding: List[float], k: int = 4) -> List[Dict[str, Any]]:
        """Return the top-k most similar documents for a query embedding.

        Args:
            query_embedding: Query vector (same dimension as the database).
            k: Number of results to return.

        Returns:
            List of dicts with keys ``"text"``, ``"metadata"``, and ``"score"``.
        """
        vec = np.array(query_embedding, dtype=np.float32)
        results = self.db.search(vec, k)

        output: List[Dict[str, Any]] = []
        for r in results:
            if isinstance(r, (tuple, list)):
                doc_id, score = r[0], r[1]
            else:
                doc_id = r["id"]
                score = r["score"]
            if doc_id in self.doc_store:
                doc = self.doc_store[doc_id]
                output.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": score,
                })
        return output
