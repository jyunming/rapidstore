"""
Python API test suite for TurboQuantDB.

Run with:
    pip install pytest numpy
    pytest tests/test_python_api.py -v
"""
import os
import tempfile

import numpy as np
import pytest

import turboquantdb as tq
from turboquantdb import Database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_unit_vecs(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def open_db(path: str, d: int = 32, bits: int = 4, metric: str = "cosine", **kw) -> Database:
    return Database.open(path, d, bits=bits, seed=42, metric=metric, rerank=True, **kw)


# ---------------------------------------------------------------------------
# Open / reopen
# ---------------------------------------------------------------------------

class TestOpenReopen:
    def test_open_creates_directory(self, tmp_path):
        db_dir = str(tmp_path / "mydb")
        db = Database.open(db_dir, 16, bits=4)
        assert os.path.isdir(db_dir)

    def test_reopen_same_dimension(self, tmp_path):
        path = str(tmp_path / "db")
        db = open_db(path, d=16)
        db.insert("v0", np.ones(16, dtype=np.float32))
        del db
        db2 = open_db(path, d=16)
        assert db2.get("v0") is not None

    def test_invalid_metric_raises(self, tmp_path):
        with pytest.raises(Exception):
            Database.open(str(tmp_path / "db"), 16, metric="bad_metric")

    def test_invalid_rerank_precision_raises(self, tmp_path):
        with pytest.raises(Exception):
            Database.open(str(tmp_path / "db"), 16, rerank_precision="bad")


# ---------------------------------------------------------------------------
# insert / upsert / update / delete
# ---------------------------------------------------------------------------

class TestCRUD:
    def test_insert_and_get(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        vec = np.array([0.1] * 8, dtype=np.float32)
        db.insert("a", vec, metadata={"k": "v"}, document="hello")
        got = db.get("a")
        assert got is not None
        assert got["id"] == "a"
        assert got["metadata"]["k"] == "v"
        assert got["document"] == "hello"

    def test_insert_duplicate_raises(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        db.insert("a", np.ones(8, dtype=np.float32))
        with pytest.raises(Exception):
            db.insert("a", np.ones(8, dtype=np.float32))

    def test_upsert_replaces_existing_vector(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        db.insert("a", np.ones(8, dtype=np.float32), metadata={"v": 1})
        db.upsert("a", np.zeros(8, dtype=np.float32), metadata={"v": 2})
        got = db.get("a")
        assert got["metadata"]["v"] == 2

    def test_upsert_inserts_new(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        db.upsert("new_id", np.ones(8, dtype=np.float32))
        assert db.get("new_id") is not None

    def test_update_existing(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        db.insert("a", np.ones(8, dtype=np.float32), metadata={"x": 1})
        db.update("a", np.zeros(8, dtype=np.float32), metadata={"x": 2})
        got = db.get("a")
        assert got["metadata"]["x"] == 2

    def test_update_missing_raises(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        with pytest.raises(Exception):
            db.update("nonexistent", np.zeros(8, dtype=np.float32))

    def test_delete_existing(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        db.insert("a", np.ones(8, dtype=np.float32))
        result = db.delete("a")
        assert result is True
        assert db.get("a") is None

    def test_delete_missing_returns_false(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        result = db.delete("ghost")
        assert result is False

    def test_get_missing_returns_none(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        assert db.get("missing") is None


# ---------------------------------------------------------------------------
# insert_batch
# ---------------------------------------------------------------------------

class TestInsertBatch:
    def test_insert_batch_basic(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=16)
        vecs = random_unit_vecs(50, 16)
        ids = [str(i) for i in range(50)]
        db.insert_batch(ids, vecs)
        assert db.stats()["vector_count"] == 50

    def test_insert_batch_upsert_mode(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=16)
        vecs = random_unit_vecs(10, 16)
        ids = [str(i) for i in range(10)]
        db.insert_batch(ids, vecs, mode="insert")
        # Upsert the same IDs — should succeed without error.
        db.insert_batch(ids, vecs, mode="upsert")
        assert db.stats()["vector_count"] == 10

    def test_insert_batch_with_metadata(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        vecs = random_unit_vecs(5, 8)
        ids = [str(i) for i in range(5)]
        metas = [{"idx": i} for i in range(5)]
        db.insert_batch(ids, vecs, metadatas=metas)
        got = db.get("3")
        assert got["metadata"]["idx"] == 3

    def test_insert_batch_with_documents(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        vecs = random_unit_vecs(3, 8)
        ids = ["a", "b", "c"]
        docs = ["doc_a", "doc_b", "doc_c"]
        db.insert_batch(ids, vecs, documents=docs)
        assert db.get("b")["document"] == "doc_b"


# ---------------------------------------------------------------------------
# get_many / list_all
# ---------------------------------------------------------------------------

class TestGetMany:
    def test_get_many_mixed(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        db.insert("a", np.ones(8, dtype=np.float32))
        db.insert("b", np.zeros(8, dtype=np.float32))
        results = db.get_many(["a", "missing", "b"])
        assert results[0]["id"] == "a"
        assert results[1] is None
        assert results[2]["id"] == "b"

    def test_list_all(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        for i in range(5):
            db.insert(str(i), np.ones(8, dtype=np.float32) * i)
        ids = db.list_all()
        assert set(ids) == {"0", "1", "2", "3", "4"}


# ---------------------------------------------------------------------------
# search — basic
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_returns_top_k(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=16)
        vecs = random_unit_vecs(100, 16)
        ids = [str(i) for i in range(100)]
        db.insert_batch(ids, vecs)
        results = db.search(vecs[0], top_k=5)
        assert len(results) == 5

    def test_search_result_fields(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        db.insert("x", np.ones(8, dtype=np.float32), metadata={"t": "y"}, document="doc")
        results = db.search(np.ones(8, dtype=np.float32), top_k=1)
        r = results[0]
        assert "id" in r
        assert "score" in r
        assert "metadata" in r
        assert "document" in r

    def test_search_sorted_descending(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=16)
        vecs = random_unit_vecs(50, 16)
        db.insert_batch([str(i) for i in range(50)], vecs)
        results = db.search(vecs[0], top_k=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_exact_match_top1(self, tmp_path):
        """Inserting a query vector should return itself as top-1."""
        db = open_db(str(tmp_path / "db"), d=32)
        vecs = random_unit_vecs(200, 32)
        db.insert_batch([str(i) for i in range(200)], vecs)
        results = db.search(vecs[5], top_k=1)
        assert results[0]["id"] == "5"

    def test_search_l2_metric(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=16, metric="l2")
        vecs = (np.random.default_rng(1).standard_normal((50, 16))).astype(np.float32)
        db.insert_batch([str(i) for i in range(50)], vecs)
        results = db.search(vecs[0], top_k=3)
        assert len(results) == 3

    def test_search_ip_metric(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=16, metric="ip")
        vecs = random_unit_vecs(50, 16)
        db.insert_batch([str(i) for i in range(50)], vecs)
        results = db.search(vecs[0], top_k=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# search — metadata filtering
# ---------------------------------------------------------------------------

class TestMetadataFilter:
    def _setup(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        vecs = random_unit_vecs(20, 8, seed=5)
        for i in range(20):
            db.insert(
                str(i),
                vecs[i],
                metadata={"year": 2020 + (i % 5), "topic": "ml" if i % 2 == 0 else "cv"},
            )
        return db, vecs

    def test_equality_filter(self, tmp_path):
        db, vecs = self._setup(tmp_path)
        results = db.search(vecs[0], top_k=20, filter={"topic": "ml"})
        assert all(r["metadata"]["topic"] == "ml" for r in results)

    def test_gt_filter(self, tmp_path):
        db, vecs = self._setup(tmp_path)
        results = db.search(vecs[0], top_k=20, filter={"year": {"$gt": 2022}})
        assert all(r["metadata"]["year"] > 2022 for r in results)

    def test_gte_filter(self, tmp_path):
        db, vecs = self._setup(tmp_path)
        results = db.search(vecs[0], top_k=20, filter={"year": {"$gte": 2023}})
        assert all(r["metadata"]["year"] >= 2023 for r in results)

    def test_lte_filter(self, tmp_path):
        db, vecs = self._setup(tmp_path)
        results = db.search(vecs[0], top_k=20, filter={"year": {"$lte": 2021}})
        assert all(r["metadata"]["year"] <= 2021 for r in results)

    def test_ne_filter(self, tmp_path):
        db, vecs = self._setup(tmp_path)
        results = db.search(vecs[0], top_k=20, filter={"topic": {"$ne": "ml"}})
        assert all(r["metadata"]["topic"] != "ml" for r in results)

    def test_and_filter(self, tmp_path):
        db, vecs = self._setup(tmp_path)
        results = db.search(
            vecs[0], top_k=20,
            filter={"$and": [{"topic": "ml"}, {"year": {"$gte": 2022}}]},
        )
        assert all(r["metadata"]["topic"] == "ml" and r["metadata"]["year"] >= 2022 for r in results)

    def test_or_filter(self, tmp_path):
        db, vecs = self._setup(tmp_path)
        results = db.search(
            vecs[0], top_k=20,
            filter={"$or": [{"topic": "ml"}, {"year": {"$eq": 2020}}]},
        )
        assert all(
            r["metadata"]["topic"] == "ml" or r["metadata"]["year"] == 2020
            for r in results
        )

    def test_filter_no_match_returns_empty(self, tmp_path):
        db, vecs = self._setup(tmp_path)
        results = db.search(vecs[0], top_k=10, filter={"year": 9999})
        assert results == []


# ---------------------------------------------------------------------------
# create_index (HNSW)
# ---------------------------------------------------------------------------

class TestCreateIndex:
    def test_create_index_then_search(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=32)
        vecs = random_unit_vecs(500, 32, seed=7)
        db.insert_batch([str(i) for i in range(500)], vecs)
        db.create_index(max_degree=16, search_list_size=64)
        results = db.search(vecs[0], top_k=5)
        assert len(results) == 5

    def test_ann_search_uses_index(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=32)
        vecs = random_unit_vecs(300, 32, seed=8)
        db.insert_batch([str(i) for i in range(300)], vecs)
        db.create_index(max_degree=16, search_list_size=64)
        # _use_ann=True (default) uses HNSW; _use_ann=False uses brute-force.
        r_ann = db.search(vecs[10], top_k=5, _use_ann=True)
        r_bf = db.search(vecs[10], top_k=5, _use_ann=False)
        # Both must return at least top-1 correctly (exact vectors).
        assert r_ann[0]["id"] == r_bf[0]["id"] == "10"


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_fields_present(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=16)
        vecs = random_unit_vecs(10, 16)
        db.insert_batch([str(i) for i in range(10)], vecs)
        s = db.stats()
        assert "vector_count" in s
        assert "has_index" in s
        assert "total_disk_bytes" in s
        assert "ram_estimate_bytes" in s
        assert s["vector_count"] == 10
        assert s["has_index"] is False

    def test_stats_after_index(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=16)
        vecs = random_unit_vecs(50, 16)
        db.insert_batch([str(i) for i in range(50)], vecs)
        db.create_index()
        s = db.stats()
        assert s["has_index"] is True


# ---------------------------------------------------------------------------
# rerank_precision
# ---------------------------------------------------------------------------

class TestRerankPrecision:
    def test_default_no_live_vectors_bin(self, tmp_path):
        path = str(tmp_path / "db")
        db = Database.open(path, 16, bits=4, rerank_precision=None)
        db.insert_batch([str(i) for i in range(5)], random_unit_vecs(5, 16))
        assert not os.path.exists(os.path.join(path, "live_vectors.bin"))

    def test_f16_creates_correct_file_size(self, tmp_path):
        path = str(tmp_path / "db")
        n, d = 20, 64
        db = Database.open(path, d, bits=4, rerank_precision="f16")
        db.insert_batch([str(i) for i in range(n)], random_unit_vecs(n, d))
        size = os.path.getsize(os.path.join(path, "live_vectors.bin"))
        # mmap pre-allocates in GROW_SLOTS=16384 chunks; file is always a multiple of stride
        assert size >= n * d * 2 and size % (d * 2) == 0, f"Expected multiple of {d*2} bytes >= {n*d*2}, got {size}"

    def test_f32_creates_correct_file_size(self, tmp_path):
        path = str(tmp_path / "db")
        n, d = 20, 64
        db = Database.open(path, d, bits=4, rerank_precision="f32")
        db.insert_batch([str(i) for i in range(n)], random_unit_vecs(n, d))
        size = os.path.getsize(os.path.join(path, "live_vectors.bin"))
        # mmap pre-allocates in GROW_SLOTS=16384 chunks; file is always a multiple of stride
        assert size >= n * d * 4 and size % (d * 4) == 0, f"Expected multiple of {d*4} bytes >= {n*d*4}, got {size}"

    def test_f16_vs_default_recall(self, tmp_path):
        """f16 exact reranking should achieve recall >= dequant (not strictly, but should be competitive)."""
        d, n = 64, 500
        vecs = random_unit_vecs(n, d, seed=99)
        ids = [str(i) for i in range(n)]
        queries = random_unit_vecs(30, d, seed=100)

        # Ground truth via brute force
        sims = queries @ vecs.T
        gt = [set(np.argsort(-sims[i])[:10]) for i in range(len(queries))]

        def recall(db_path, precision):
            db = Database.open(db_path, d, bits=4, metric="cosine", rerank_precision=precision)
            db.insert_batch(ids, vecs)
            hits = 0
            for i, q in enumerate(queries):
                res = db.search(q, 10)
                hits += len({int(r["id"]) for r in res} & gt[i])
            return hits / (len(queries) * 10)

        r_default = recall(str(tmp_path / "dq"), None)
        r_f16 = recall(str(tmp_path / "f16"), "f16")

        # f16 should have recall >= default with some tolerance
        assert r_f16 >= r_default - 0.05, f"f16 recall {r_f16:.2%} significantly worse than dequant {r_default:.2%}"

    def test_invalid_precision_raises(self, tmp_path):
        with pytest.raises(Exception):
            Database.open(str(tmp_path / "db"), 16, rerank_precision="int8")


# ---------------------------------------------------------------------------
# persistence / WAL recovery
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_data_persists_across_reopen(self, tmp_path):
        path = str(tmp_path / "db")
        db = open_db(path, d=16)
        vecs = random_unit_vecs(20, 16)
        db.insert_batch([str(i) for i in range(20)], vecs)
        del db

        db2 = open_db(path, d=16)
        assert db2.stats()["vector_count"] == 20
        assert db2.get("5") is not None

    def test_delete_persists_across_reopen(self, tmp_path):
        path = str(tmp_path / "db")
        db = open_db(path, d=8)
        db.insert("a", np.ones(8, dtype=np.float32))
        db.delete("a")
        del db

        db2 = open_db(path, d=8)
        assert db2.get("a") is None

    def test_metadata_persists_across_reopen(self, tmp_path):
        path = str(tmp_path / "db")
        db = open_db(path, d=8)
        db.insert("a", np.ones(8, dtype=np.float32), metadata={"key": "value"}, document="doc text")
        del db

        db2 = open_db(path, d=8)
        got = db2.get("a")
        assert got["metadata"]["key"] == "value"
        assert got["document"] == "doc text"


# ---------------------------------------------------------------------------
# TurboQuantRetriever (RAG wrapper) tests
# ---------------------------------------------------------------------------

class TestTurboQuantRetriever:
    """Tests for the TurboQuantRetriever class in turboquantdb.rag."""

    DIM = 32

    def _make_embeddings(self, n: int, seed: int = 0) -> list:
        rng = np.random.default_rng(seed)
        vecs = rng.standard_normal((n, self.DIM)).astype(np.float64)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= norms
        return vecs.tolist()

    def test_import_retriever(self):
        """TurboQuantRetriever can be imported from turboquantdb.rag."""
        from turboquantdb.rag import TurboQuantRetriever  # noqa: F401

    def test_add_texts_without_metadata(self, tmp_path):
        """add_texts with no metadata arg uses empty dicts."""
        from turboquantdb.rag import TurboQuantRetriever

        retriever = TurboQuantRetriever(
            str(tmp_path / "db"), dimension=self.DIM, bits=4, seed=42
        )
        texts = ["hello world", "foo bar"]
        embeddings = self._make_embeddings(2, seed=1)
        retriever.add_texts(texts, embeddings)
        # doc_store should have two entries
        assert len(retriever.doc_store) == 2

    def test_add_texts_with_metadata(self, tmp_path):
        """add_texts stores metadata alongside the text."""
        from turboquantdb.rag import TurboQuantRetriever

        retriever = TurboQuantRetriever(
            str(tmp_path / "db"), dimension=self.DIM, bits=4, seed=42
        )
        texts = ["doc one", "doc two", "doc three"]
        embeddings = self._make_embeddings(3, seed=2)
        metadatas = [{"source": "a"}, {"source": "b"}, {"source": "c"}]
        retriever.add_texts(texts, embeddings, metadatas=metadatas)

        assert retriever.doc_store["doc_0"]["metadata"]["source"] == "a"
        assert retriever.doc_store["doc_1"]["text"] == "doc two"
        assert retriever.doc_store["doc_2"]["metadata"]["source"] == "c"

    def test_add_texts_accumulates_ids(self, tmp_path):
        """Calling add_texts twice generates non-overlapping IDs."""
        from turboquantdb.rag import TurboQuantRetriever

        retriever = TurboQuantRetriever(
            str(tmp_path / "db"), dimension=self.DIM, bits=4, seed=42
        )
        retriever.add_texts(["first"], self._make_embeddings(1, seed=0))
        retriever.add_texts(["second"], self._make_embeddings(1, seed=1))

        assert len(retriever.doc_store) == 2
        assert "doc_0" in retriever.doc_store
        assert "doc_1" in retriever.doc_store

    def test_similarity_search_returns_list_of_dicts(self, tmp_path):
        """similarity_search returns a list of dicts with text/metadata/score."""
        from turboquantdb.rag import TurboQuantRetriever

        retriever = TurboQuantRetriever(
            str(tmp_path / "db"), dimension=self.DIM, bits=4, seed=42
        )
        texts = [f"document {i}" for i in range(10)]
        embeddings = self._make_embeddings(10, seed=7)
        retriever.add_texts(texts, embeddings)

        query = self._make_embeddings(1, seed=99)[0]
        results = retriever.similarity_search(query, k=3)

        assert isinstance(results, list)
        assert len(results) <= 3
        for r in results:
            assert "text" in r, f"missing 'text' key in result: {r}"
            assert "metadata" in r, f"missing 'metadata' key in result: {r}"
            assert "score" in r, f"missing 'score' key in result: {r}"

    def test_similarity_search_result_text_is_original(self, tmp_path):
        """similarity_search results contain the original inserted text."""
        from turboquantdb.rag import TurboQuantRetriever

        retriever = TurboQuantRetriever(
            str(tmp_path / "db"), dimension=self.DIM, bits=4, seed=42
        )
        texts = ["alpha text", "beta text", "gamma text"]
        embeddings = self._make_embeddings(3, seed=5)
        retriever.add_texts(texts, embeddings)

        query = embeddings[0]  # query == first embedding
        results = retriever.similarity_search(query, k=1)

        assert len(results) == 1
        assert results[0]["text"] in texts

    def test_similarity_search_empty_db_returns_empty(self, tmp_path):
        """similarity_search on an empty database returns an empty list."""
        from turboquantdb.rag import TurboQuantRetriever

        retriever = TurboQuantRetriever(
            str(tmp_path / "db"), dimension=self.DIM, bits=4, seed=42
        )
        query = self._make_embeddings(1, seed=0)[0]
        results = retriever.similarity_search(query, k=5)
        assert results == []

    def test_similarity_search_k_limits_results(self, tmp_path):
        """similarity_search respects the k parameter."""
        from turboquantdb.rag import TurboQuantRetriever

        retriever = TurboQuantRetriever(
            str(tmp_path / "db"), dimension=self.DIM, bits=4, seed=42
        )
        texts = [f"doc {i}" for i in range(20)]
        embeddings = self._make_embeddings(20, seed=3)
        retriever.add_texts(texts, embeddings)

        query = self._make_embeddings(1, seed=10)[0]
        results = retriever.similarity_search(query, k=5)
        assert len(results) <= 5

    def test_similarity_search_metadata_preserved(self, tmp_path):
        """Metadata inserted via add_texts is returned in search results."""
        from turboquantdb.rag import TurboQuantRetriever

        retriever = TurboQuantRetriever(
            str(tmp_path / "db"), dimension=self.DIM, bits=4, seed=42
        )
        texts = ["tagged doc"]
        embeddings = self._make_embeddings(1, seed=11)
        retriever.add_texts(texts, embeddings, metadatas=[{"tag": "important"}])

        query = embeddings[0]
        results = retriever.similarity_search(query, k=1)
        assert len(results) == 1
        assert results[0]["metadata"].get("tag") == "important"


class TestTurboQuantRetrieverFallbackStub:
    """Tests the fallback stub behaviour when the C extension is unavailable."""

    def test_stub_raises_runtime_error(self):
        """Database.open on the stub raises RuntimeError."""
        # Import the rag module's fallback stub class directly by temporarily
        # hiding the real extension.
        import sys
        import importlib
        import types

        # Create a fake turboquantdb package that lacks the real extension.
        fake_pkg = types.ModuleType("turboquantdb_stub_test")
        # The rag module's try/except block falls back to a stub Database class
        # that raises RuntimeError on .open(). We can import rag with this
        # setup by temporarily replacing the turboquantdb.turboquantdb submodule.
        original = sys.modules.get("turboquantdb.turboquantdb")
        try:
            # Remove the real extension so the ImportError branch fires.
            sys.modules["turboquantdb.turboquantdb"] = None  # type: ignore
            # Force reimport of rag to pick up the stub path.
            if "turboquantdb.rag" in sys.modules:
                del sys.modules["turboquantdb.rag"]
            from turboquantdb import rag as rag_mod
            # The stub Database.open should raise RuntimeError.
            with pytest.raises(RuntimeError):
                rag_mod.Database.open("/tmp/stub_test_db", 16)
        finally:
            # Restore the real extension.
            if original is None:
                sys.modules.pop("turboquantdb.turboquantdb", None)
            else:
                sys.modules["turboquantdb.turboquantdb"] = original
            # Clear the rag module cache so future tests use the real extension.
            sys.modules.pop("turboquantdb.rag", None)


# ---------------------------------------------------------------------------
# TurboQuantRetriever (rag.py)
# ---------------------------------------------------------------------------

class TestTurboQuantRetriever:
    """Tests for the LangChain-style TurboQuantRetriever wrapper."""

    def test_import(self):
        from turboquantdb.rag import TurboQuantRetriever  # noqa: F401

    def test_add_texts_and_similarity_search(self, tmp_path):
        from turboquantdb.rag import TurboQuantRetriever

        d = 32
        retriever = TurboQuantRetriever(str(tmp_path / "rag_db"), dimension=d, bits=4, seed=42)

        rng = np.random.default_rng(0)
        texts = [f"document {i}" for i in range(20)]
        embeddings = rng.standard_normal((20, d)).tolist()

        retriever.add_texts(texts, embeddings)

        query = embeddings[0]
        results = retriever.similarity_search(query, k=3)

        assert len(results) == 3
        for r in results:
            assert "text" in r
            assert "metadata" in r
            assert "score" in r

    def test_add_texts_with_metadata(self, tmp_path):
        from turboquantdb.rag import TurboQuantRetriever

        d = 32
        retriever = TurboQuantRetriever(str(tmp_path / "rag_db"), dimension=d, bits=4, seed=42)

        rng = np.random.default_rng(1)
        texts = ["alpha", "beta", "gamma"]
        embeddings = rng.standard_normal((3, d)).tolist()
        metadatas = [{"tag": "a"}, {"tag": "b"}, {"tag": "c"}]

        retriever.add_texts(texts, embeddings, metadatas=metadatas)

        results = retriever.similarity_search(embeddings[1], k=1)
        assert results[0]["text"] == "beta"
        assert results[0]["metadata"]["tag"] == "b"

    def test_add_texts_without_metadata_defaults_empty(self, tmp_path):
        from turboquantdb.rag import TurboQuantRetriever

        d = 16
        retriever = TurboQuantRetriever(str(tmp_path / "rag_db"), dimension=d, bits=4, seed=42)

        rng = np.random.default_rng(2)
        texts = ["x", "y"]
        embeddings = rng.standard_normal((2, d)).tolist()
        retriever.add_texts(texts, embeddings)  # no metadatas arg

        results = retriever.similarity_search(embeddings[0], k=1)
        assert results[0]["metadata"] == {}

    def test_multiple_add_texts_accumulates(self, tmp_path):
        from turboquantdb.rag import TurboQuantRetriever

        d = 16
        retriever = TurboQuantRetriever(str(tmp_path / "rag_db"), dimension=d, bits=4, seed=42)

        rng = np.random.default_rng(3)
        texts1 = ["a", "b"]
        texts2 = ["c", "d", "e"]
        emb1 = rng.standard_normal((2, d)).tolist()
        emb2 = rng.standard_normal((3, d)).tolist()

        retriever.add_texts(texts1, emb1)
        retriever.add_texts(texts2, emb2)

        assert len(retriever.doc_store) == 5

    def test_similarity_search_empty_db_returns_empty(self, tmp_path):
        from turboquantdb.rag import TurboQuantRetriever

        d = 16
        retriever = TurboQuantRetriever(str(tmp_path / "rag_db"), dimension=d, bits=4, seed=42)
        rng = np.random.default_rng(4)
        results = retriever.similarity_search(rng.standard_normal(d).tolist(), k=5)
        assert results == []

    def test_fallback_stub_raises_runtime_error(self):
        """The fallback Database stub (used when extension is unavailable) must raise."""
        import importlib, sys

        # Temporarily hide the real extension to force the fallback path
        real_mod = sys.modules.pop("turboquantdb.turboquantdb", None)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_rag_stub_test",
                "python/turboquantdb/rag.py",
            )
            mod = importlib.util.module_from_spec(spec)
            # Patch the import inside rag.py to use the stub Database
            mod.__package__ = None  # prevent relative import
            # Execute with turboquantdb.turboquantdb hidden → ImportError → stub activated
            orig = sys.modules.get("turboquantdb.turboquantdb")
            sys.modules["turboquantdb.turboquantdb"] = None  # type: ignore
            try:
                spec.loader.exec_module(mod)
                with pytest.raises(RuntimeError, match="not available"):
                    mod.Database.open("path", 8)
            finally:
                if orig is None:
                    sys.modules.pop("turboquantdb.turboquantdb", None)
                else:
                    sys.modules["turboquantdb.turboquantdb"] = orig
        finally:
            if real_mod is not None:
                sys.modules["turboquantdb.turboquantdb"] = real_mod
