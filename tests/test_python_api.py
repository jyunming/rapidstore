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
