import numpy as np
from typing import Any, Dict, List, Optional

try:
    from .turboquantdb import Database
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
            raise RuntimeError("turboquantdb extension not available")


class TurboQuantRetriever:
    """Simple retriever wrapper around TurboQuantDB."""

    def __init__(
        self,
        db_path: str,
        dimension: int = 1536,
        bits: int = 4,
        seed: int = 42,
        metric: str = "ip",
        rerank_precision: Optional[str] = None,
    ):
        self.db = Database.open(
            db_path, dimension, bits=bits, seed=seed, metric=metric,
            rerank_precision=rerank_precision,
        )
        self.doc_store: Dict[str, Dict[str, Any]] = {}

    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if metadatas is None:
            metadatas = [{} for _ in texts]

        start = len(self.doc_store)
        ids = [f"doc_{start + i}" for i in range(len(texts))]
        for i, doc_id in enumerate(ids):
            self.doc_store[doc_id] = {"text": texts[i], "metadata": metadatas[i]}

        vectors = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float64))
        if hasattr(self.db, "insert_batch"):
            self.db.insert_batch(ids, vectors, metadatas, texts, "insert")
            return

        if hasattr(self.db, "insert_many"):
            self.db.insert_many(ids, [row for row in vectors], metadatas, texts, "insert")
            return

        for i, doc_id in enumerate(ids):
            self.db.insert(doc_id, vectors[i], metadatas[i], texts[i])

    def similarity_search(self, query_embedding: List[float], k: int = 4) -> List[Dict[str, Any]]:
        vec = np.array(query_embedding, dtype=np.float64)
        results = self.db.search(vec, k)

        output: List[Dict[str, Any]] = []
        for r in results:
            # Search returns dicts; guard against future tuple shape (id, score).
            if isinstance(r, dict):
                doc_id = r.get("id")
                score = r.get("score")
            else:
                doc_id, score = r[0], r[1]
            if doc_id in self.doc_store:
                doc = self.doc_store[doc_id]
                output.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": score,
                })
        return output
