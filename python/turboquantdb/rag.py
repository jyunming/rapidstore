import numpy as np
from typing import Any, Dict, List, Optional

try:
    from .turboquantdb import Database
except ImportError:
    class Database:  # type: ignore
        @staticmethod
        def open(uri: str, dimension: int, bits: int, seed: int = 42, local_dir: Optional[str] = None, metric: str = "ip"):
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
    ):
        self.db = Database.open(db_path, dimension, bits, seed, None, metric)
        self.doc_store: Dict[str, Dict[str, Any]] = {}

    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if metadatas is None:
            metadatas = [{} for _ in texts]

        for i, (text, emb, meta) in enumerate(zip(texts, embeddings, metadatas)):
            doc_id = f"doc_{len(self.doc_store) + i}"
            self.doc_store[doc_id] = {"text": text, "metadata": meta}
            vec = np.array(emb, dtype=np.float64)
            self.db.insert(doc_id, vec, meta, text)

    def similarity_search(self, query_embedding: List[float], k: int = 4) -> List[Dict[str, Any]]:
        vec = np.array(query_embedding, dtype=np.float64)
        results = self.db.search(vec, k)

        output: List[Dict[str, Any]] = []
        for r in results:
            doc_id = r.get("id")
            score = r.get("score")
            if doc_id in self.doc_store:
                doc = self.doc_store[doc_id]
                output.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": score,
                })
        return output
