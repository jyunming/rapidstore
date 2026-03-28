import numpy as np
from typing import List, Dict, Any, Optional
try:
    from .turboquantdb import Database
except ImportError:
    # Fallback/mock for typing when Rust extension isn't compiled
    class Database:
        def __init__(self, d: int, b: int, seed: int = 42): pass
        def insert(self, id: str, vector: np.ndarray, metadata: Optional[Dict] = None): pass
        def search(self, query: np.ndarray, top_k: int) -> List[tuple[str, float]]: return []

class TurboQuantRetriever:
    """
    A LangChain-compatible retriever interface for TurboQuant DB.
    """
    def __init__(self, dimension: int = 1536, bits: int = 4, seed: int = 42):
        self.db = Database(d=dimension, b=bits, seed=seed)
        self.doc_store = {}
        
    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None):
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        for i, (text, emb, meta) in enumerate(zip(texts, embeddings, metadatas)):
            doc_id = f"doc_{len(self.doc_store) + i}"
            self.doc_store[doc_id] = {"text": text, "metadata": meta}
            
            # Convert to float64 numpy array for the Rust DB
            vec = np.array(emb, dtype=np.float64)
            self.db.insert(doc_id, vec, None) # Metadata handled in Python layer for simplicity here
            
    def similarity_search(self, query_embedding: List[float], k: int = 4) -> List[Dict[str, Any]]:
        vec = np.array(query_embedding, dtype=np.float64)
        results = self.db.search(vec, k)
        
        output = []
        for doc_id, score in results:
            if doc_id in self.doc_store:
                doc = self.doc_store[doc_id]
                output.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": score
                })
        return output
