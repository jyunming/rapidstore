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

import tqdb as tq
from tqdb import Database


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
        assert s["dimension"] == 16
        assert s["bits"] == 4

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

    def test_open_parameterless_reopen(self, tmp_path):
        path = str(tmp_path / "db")
        db = Database.open(path, dimension=16, bits=4, metric="cosine")
        db.insert("a", np.ones(16, dtype=np.float32))
        del db

        # Reopen without specifying any parameters — reads from manifest.json
        db2 = Database.open(path)
        assert db2.count() == 1
        assert db2.get("a") is not None
        stats = db2.stats()
        assert stats["dimension"] == 16

    def test_open_parameterless_new_db_raises(self, tmp_path):
        path = str(tmp_path / "newdb")
        with pytest.raises(Exception, match="dimension"):
            Database.open(path)  # no manifest, no dimension → error


# ---------------------------------------------------------------------------
# count()
# ---------------------------------------------------------------------------

class TestCount:
    def test_count_no_filter_returns_total(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        assert db.count() == 0
        db.insert("a", np.ones(8, dtype=np.float32))
        db.insert("b", np.ones(8, dtype=np.float32) * 0.5)
        assert db.count() == 2

    def test_count_after_delete(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        db.insert("a", np.ones(8, dtype=np.float32))
        db.insert("b", np.ones(8, dtype=np.float32))
        db.delete("a")
        assert db.count() == 1

    def test_count_with_filter(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        db.insert("a", np.ones(8, dtype=np.float32), metadata={"topic": "ml"})
        db.insert("b", np.ones(8, dtype=np.float32), metadata={"topic": "nlp"})
        db.insert("c", np.ones(8, dtype=np.float32), metadata={"topic": "ml"})
        assert db.count(filter={"topic": "ml"}) == 2
        assert db.count(filter={"topic": "nlp"}) == 1
        assert db.count(filter={"topic": "cv"}) == 0

    def test_count_with_comparison_filter(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        for i in range(5):
            db.insert(str(i), np.ones(8, dtype=np.float32), metadata={"year": 2020 + i})
        assert db.count(filter={"year": {"$gte": 2023}}) == 2  # 2023, 2024


# ---------------------------------------------------------------------------
# delete_batch()
# ---------------------------------------------------------------------------

class TestDeleteBatch:
    def test_delete_batch_returns_count(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        db.insert("a", np.ones(8, dtype=np.float32))
        db.insert("b", np.ones(8, dtype=np.float32))
        db.insert("c", np.ones(8, dtype=np.float32))
        deleted = db.delete_batch(["a", "c", "missing"])
        assert deleted == 2
        assert db.count() == 1
        assert db.get("b") is not None

    def test_delete_batch_empty_list(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        db.insert("a", np.ones(8, dtype=np.float32))
        assert db.delete_batch([]) == 0
        assert db.count() == 1

    def test_delete_batch_all_missing(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        assert db.delete_batch(["x", "y", "z"]) == 0

    def test_delete_batch_persists_across_reopen(self, tmp_path):
        path = str(tmp_path / "db")
        db = open_db(path, d=8)
        for i in range(5):
            db.insert(str(i), np.ones(8, dtype=np.float32))
        db.delete_batch(["0", "2", "4"])
        del db
        db2 = open_db(path, d=8)
        assert db2.count() == 2
        assert db2.get("1") is not None
        assert db2.get("3") is not None
        assert db2.get("0") is None

    def test_delete_batch_where_filter(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        db.insert("keep-1", np.ones(8, dtype=np.float32), metadata={"tag": "keep"})
        db.insert("keep-2", np.ones(8, dtype=np.float32), metadata={"tag": "keep"})
        db.insert("del-1",  np.ones(8, dtype=np.float32), metadata={"tag": "old"})
        db.insert("del-2",  np.ones(8, dtype=np.float32), metadata={"tag": "old"})
        deleted = db.delete_batch(where_filter={"tag": {"$eq": "old"}})
        assert deleted == 2
        assert db.count() == 2
        assert db.get("keep-1") is not None
        assert db.get("del-1") is None

    def test_delete_batch_ids_and_filter_no_double_count(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        db.insert("a", np.ones(8, dtype=np.float32), metadata={"x": 1})
        db.insert("b", np.ones(8, dtype=np.float32), metadata={"x": 1})
        db.insert("c", np.ones(8, dtype=np.float32), metadata={"x": 2})
        # "a" is in both ids and matches filter — should be counted once
        deleted = db.delete_batch(["a"], where_filter={"x": {"$eq": 1}})
        assert deleted == 2  # "a" + "b" (no double-count)
        assert db.count() == 1
        assert db.get("c") is not None

    def test_delete_batch_filter_no_match(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        db.insert("a", np.ones(8, dtype=np.float32), metadata={"x": 1})
        assert db.delete_batch(where_filter={"x": {"$eq": 99}}) == 0
        assert db.count() == 1


# ---------------------------------------------------------------------------
# multi-collection (collection= parameter)
# ---------------------------------------------------------------------------

class TestMultiCollection:
    def test_collection_creates_subdirectory(self, tmp_path):
        base = str(tmp_path / "base")
        db = Database.open(base, 8, bits=4, collection="col1")
        assert os.path.isdir(os.path.join(base, "col1"))

    def test_two_collections_are_isolated(self, tmp_path):
        base = str(tmp_path / "base")
        col1 = Database.open(base, 8, bits=4, collection="col1")
        col2 = Database.open(base, 8, bits=4, collection="col2")
        col1.insert("shared-id", np.ones(8, dtype=np.float32))
        assert col1.count() == 1
        assert col2.count() == 0  # isolated

    def test_no_collection_uses_base_path(self, tmp_path):
        base = str(tmp_path / "base")
        db = Database.open(base, 8, bits=4)
        db.insert("v0", np.ones(8, dtype=np.float32))
        assert db.count() == 1

    def test_collection_persists_across_reopen(self, tmp_path):
        base = str(tmp_path / "base")
        db = Database.open(base, 8, bits=4, collection="docs")
        db.insert("v0", np.ones(8, dtype=np.float32))
        del db
        db2 = Database.open(base, 8, bits=4, collection="docs")
        assert db2.get("v0") is not None


# ---------------------------------------------------------------------------
# flush() and close()
# ---------------------------------------------------------------------------

class TestFlushClose:
    def test_flush_is_callable(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        db.insert("a", np.ones(8, dtype=np.float32))
        db.flush()  # should not raise

    def test_close_is_callable(self, tmp_path):
        db = open_db(str(tmp_path / "db"), d=8)
        db.insert("a", np.ones(8, dtype=np.float32))
        db.close()  # should not raise

    def test_flush_data_readable_after_reopen(self, tmp_path):
        path = str(tmp_path / "db")
        db = open_db(path, d=8)
        for i in range(10):
            db.insert(str(i), np.ones(8, dtype=np.float32))
        db.flush()
        del db
        db2 = open_db(path, d=8)
        assert db2.count() == 10


# ---------------------------------------------------------------------------
# TurboQuantRetriever (RAG wrapper) tests
# ---------------------------------------------------------------------------

class TestTurboQuantRetrieverOps:
    """Tests for the TurboQuantRetriever class in tqdb.rag."""

    DIM = 32

    def _make_embeddings(self, n: int, seed: int = 0) -> list:
        rng = np.random.default_rng(seed)
        vecs = rng.standard_normal((n, self.DIM)).astype(np.float64)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= norms
        return vecs.tolist()

    def test_import_retriever(self):
        """TurboQuantRetriever can be imported from tqdb.rag."""
        from tqdb.rag import TurboQuantRetriever  # noqa: F401

    def test_add_texts_without_metadata(self, tmp_path):
        """add_texts with no metadata arg uses empty dicts."""
        from tqdb.rag import TurboQuantRetriever

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
        from tqdb.rag import TurboQuantRetriever

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
        from tqdb.rag import TurboQuantRetriever

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
        from tqdb.rag import TurboQuantRetriever

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
        from tqdb.rag import TurboQuantRetriever

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
        from tqdb.rag import TurboQuantRetriever

        retriever = TurboQuantRetriever(
            str(tmp_path / "db"), dimension=self.DIM, bits=4, seed=42
        )
        query = self._make_embeddings(1, seed=0)[0]
        results = retriever.similarity_search(query, k=5)
        assert results == []

    def test_similarity_search_k_limits_results(self, tmp_path):
        """similarity_search respects the k parameter."""
        from tqdb.rag import TurboQuantRetriever

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
        from tqdb.rag import TurboQuantRetriever

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

class TestNewFilterOps:
    def test_in_operator(self, tmp_path):
        db = open_db(str(tmp_path))
        vecs = random_unit_vecs(5, 32)
        for i in range(5):
            db.insert(f"v{i}", vecs[i], metadata={"tag": chr(ord("A") + i)})
        results = db.search(vecs[0], 10, filter={"tag": {"$in": ["A", "C"]}})
        ids = {r["id"] for r in results}
        assert "v0" in ids and "v2" in ids
        assert "v1" not in ids and "v3" not in ids

    def test_nin_operator(self, tmp_path):
        db = open_db(str(tmp_path))
        vecs = random_unit_vecs(4, 32)
        for i in range(4):
            db.insert(f"v{i}", vecs[i], metadata={"tag": chr(ord("A") + i)})
        results = db.search(vecs[0], 10, filter={"tag": {"$nin": ["B", "C"]}})
        ids = {r["id"] for r in results}
        assert "v0" in ids and "v3" in ids
        assert "v1" not in ids and "v2" not in ids

    def test_exists_operator(self, tmp_path):
        db = open_db(str(tmp_path))
        vecs = random_unit_vecs(4, 32)
        db.insert("with", vecs[0], metadata={"extra": "yes"})
        db.insert("without", vecs[1], metadata={"other": "val"})
        db.insert("with2", vecs[2], metadata={"extra": "no"})
        results = db.search(vecs[0], 10, filter={"extra": {"$exists": True}})
        ids = {r["id"] for r in results}
        assert "with" in ids and "with2" in ids
        assert "without" not in ids

    def test_contains_operator(self, tmp_path):
        db = open_db(str(tmp_path))
        vecs = random_unit_vecs(4, 32)
        db.insert("a", vecs[0], metadata={"text": "hello world"})
        db.insert("b", vecs[1], metadata={"text": "goodbye"})
        db.insert("c", vecs[2], metadata={"text": "hello again"})
        results = db.search(vecs[0], 10, filter={"text": {"$contains": "hello"}})
        ids = {r["id"] for r in results}
        assert "a" in ids and "c" in ids
        assert "b" not in ids


# ---------------------------------------------------------------------------
# update_metadata
# ---------------------------------------------------------------------------

class TestUpdateMetadata:
    def test_update_metadata_only(self, tmp_path):
        db = open_db(str(tmp_path))
        vec = random_unit_vecs(1, 32)[0]
        db.insert("x", vec, metadata={"k": "old"}, document="orig")
        db.update_metadata("x", metadata={"k": "new"})
        rec = db.get("x")
        assert rec["metadata"]["k"] == "new"
        assert rec["document"] == "orig"

    def test_update_document_only(self, tmp_path):
        db = open_db(str(tmp_path))
        vec = random_unit_vecs(1, 32)[0]
        db.insert("x", vec, metadata={"k": "v"}, document="old")
        db.update_metadata("x", document="new doc")
        rec = db.get("x")
        assert rec["document"] == "new doc"
        assert rec["metadata"]["k"] == "v"

    def test_update_metadata_missing_id_raises(self, tmp_path):
        db = open_db(str(tmp_path))
        with pytest.raises(Exception):
            db.update_metadata("nonexistent", metadata={"k": "v"})

    def test_update_metadata_does_not_affect_search(self, tmp_path):
        db = open_db(str(tmp_path))
        vecs = random_unit_vecs(3, 32)
        for i in range(3):
            db.insert(f"v{i}", vecs[i], metadata={"cat": "A"})
        db.update_metadata("v1", metadata={"cat": "B"})
        results = db.search(vecs[0], 10, filter={"cat": "A"})
        ids = {r["id"] for r in results}
        assert "v0" in ids and "v2" in ids
        assert "v1" not in ids


# ---------------------------------------------------------------------------
# query (batch multi-query)
# ---------------------------------------------------------------------------

class TestQueryBatch:
    def test_basic_batch(self, tmp_path):
        db = open_db(str(tmp_path))
        vecs = random_unit_vecs(10, 32)
        for i in range(10):
            db.insert(f"v{i}", vecs[i])
        queries = np.stack([vecs[0], vecs[1], vecs[2]])
        all_results = db.query(queries, n_results=3)
        assert len(all_results) == 3
        for res in all_results:
            assert len(res) <= 3

    def test_batch_with_filter(self, tmp_path):
        db = open_db(str(tmp_path))
        vecs = random_unit_vecs(6, 32)
        for i in range(6):
            db.insert(f"v{i}", vecs[i], metadata={"grp": "A" if i < 3 else "B"})
        queries = np.stack([vecs[0], vecs[3]])
        all_results = db.query(queries, n_results=5, where_filter={"grp": "A"})
        for res in all_results:
            for r in res:
                assert r["metadata"]["grp"] == "A"

    def test_batch_float64(self, tmp_path):
        db = open_db(str(tmp_path))
        vecs = random_unit_vecs(5, 32)
        for i in range(5):
            db.insert(f"v{i}", vecs[i])
        queries = np.stack([vecs[0], vecs[1]]).astype(np.float64)
        all_results = db.query(queries, n_results=2)
        assert len(all_results) == 2


# ---------------------------------------------------------------------------
# list_ids (paginated, filtered)
# ---------------------------------------------------------------------------

class TestListIds:
    def test_list_all_ids(self, tmp_path):
        db = open_db(str(tmp_path))
        vecs = random_unit_vecs(5, 32)
        for i in range(5):
            db.insert(f"v{i}", vecs[i])
        ids = db.list_ids()
        assert set(ids) == {f"v{i}" for i in range(5)}

    def test_list_ids_with_filter(self, tmp_path):
        db = open_db(str(tmp_path))
        vecs = random_unit_vecs(6, 32)
        for i in range(6):
            db.insert(f"v{i}", vecs[i], metadata={"cat": "A" if i < 3 else "B"})
        ids = db.list_ids(where_filter={"cat": "A"})
        assert set(ids) == {"v0", "v1", "v2"}

    def test_list_ids_pagination(self, tmp_path):
        db = open_db(str(tmp_path))
        vecs = random_unit_vecs(10, 32)
        for i in range(10):
            db.insert(f"v{i}", vecs[i])
        page1 = db.list_ids(limit=5, offset=0)
        page2 = db.list_ids(limit=5, offset=5)
        assert len(page1) == 5
        assert len(page2) == 5
        assert set(page1).isdisjoint(set(page2))
        assert set(page1) | set(page2) == {f"v{i}" for i in range(10)}

    def test_list_ids_offset_beyond_end(self, tmp_path):
        db = open_db(str(tmp_path))
        vecs = random_unit_vecs(3, 32)
        for i in range(3):
            db.insert(f"v{i}", vecs[i])
        ids = db.list_ids(offset=10)
        assert ids == []


# ---------------------------------------------------------------------------
# Python dunders: __len__, __contains__
# ---------------------------------------------------------------------------

class TestDunders:
    def test_len(self, tmp_path):
        db = open_db(str(tmp_path))
        assert len(db) == 0
        vecs = random_unit_vecs(5, 32)
        for i in range(5):
            db.insert(f"v{i}", vecs[i])
        assert len(db) == 5
        db.delete("v0")
        assert len(db) == 4

    def test_contains(self, tmp_path):
        db = open_db(str(tmp_path))
        vecs = random_unit_vecs(3, 32)
        db.insert("exists", vecs[0])
        assert "exists" in db
        assert "missing" not in db


# ---------------------------------------------------------------------------
# search include= parameter
# ---------------------------------------------------------------------------

class TestSearchIncludeParam:
    def test_include_id_only(self, tmp_path):
        db = open_db(str(tmp_path))
        vecs = random_unit_vecs(3, 32)
        for i in range(3):
            db.insert(f"v{i}", vecs[i], metadata={"k": "v"}, document="doc")
        results = db.search(vecs[0], 3, include=["id"])
        assert len(results) > 0
        for r in results:
            assert "id" in r
            assert "score" not in r
            assert "metadata" not in r
            assert "document" not in r

    def test_include_defaults_all_fields(self, tmp_path):
        db = open_db(str(tmp_path))
        vecs = random_unit_vecs(3, 32)
        for i in range(3):
            db.insert(f"v{i}", vecs[i], metadata={"k": "v"}, document="doc")
        results = db.search(vecs[0], 3)
        for r in results:
            assert "id" in r
            assert "score" in r
            assert "metadata" in r


# ---------------------------------------------------------------------------
# TurboQuantRetriever (rag.py)
# ---------------------------------------------------------------------------

class TestTurboQuantRetriever:
    """Tests for the LangChain-style TurboQuantRetriever wrapper."""

    def test_import(self):
        from tqdb.rag import TurboQuantRetriever  # noqa: F401

    def test_add_texts_and_similarity_search(self, tmp_path):
        from tqdb.rag import TurboQuantRetriever

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
        from tqdb.rag import TurboQuantRetriever

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
        from tqdb.rag import TurboQuantRetriever

        d = 16
        retriever = TurboQuantRetriever(str(tmp_path / "rag_db"), dimension=d, bits=4, seed=42)

        rng = np.random.default_rng(2)
        texts = ["x", "y"]
        embeddings = rng.standard_normal((2, d)).tolist()
        retriever.add_texts(texts, embeddings)  # no metadatas arg

        results = retriever.similarity_search(embeddings[0], k=1)
        assert results[0]["metadata"] == {}

    def test_multiple_add_texts_accumulates(self, tmp_path):
        from tqdb.rag import TurboQuantRetriever

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
        from tqdb.rag import TurboQuantRetriever

        d = 16
        retriever = TurboQuantRetriever(str(tmp_path / "rag_db"), dimension=d, bits=4, seed=42)
        rng = np.random.default_rng(4)
        results = retriever.similarity_search(rng.standard_normal(d).tolist(), k=5)
        assert results == []

    # ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Recall quality gate (mirrors CI benchmark at smaller scale)
# ---------------------------------------------------------------------------

class TestRecallQualityGate:
    """Validates that brute-force recall meets the minimum bar for the CI config.

    Uses a small corpus (n=2000, d=64) so the test stays fast (~2s), while
    exercising the same code path as the CI benchmark (d=384, n=50000).
    The 0.70 floor is conservative — production configs typically reach 80%+.
    """

    def _ground_truth(self, corpus: np.ndarray, query: np.ndarray, k: int) -> set[str]:
        scores = corpus @ query
        top_idx = np.argsort(scores)[::-1][:k]
        return {f"v{i}" for i in top_idx}

    def test_brute_force_recall_bits4(self, tmp_path):
        """bits=4 brute-force should achieve ≥70 % Recall@10 on random unit vectors."""
        rng = np.random.default_rng(42)
        n, d, k = 2000, 64, 10
        corpus = rng.standard_normal((n, d)).astype(np.float32)
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)

        db = Database.open(str(tmp_path / "db"), d, bits=4, seed=42, metric="ip")
        ids = [f"v{i}" for i in range(n)]
        db.insert_batch(ids, corpus)

        queries = rng.standard_normal((20, d)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        recalls = []
        for q in queries:
            gt = self._ground_truth(corpus, q, k)
            results = db.search(q, top_k=k)
            retrieved = {r["id"] for r in results}
            recalls.append(len(gt & retrieved) / k)

        avg_recall = float(np.mean(recalls))
        assert avg_recall >= 0.70, (
            f"Brute-force Recall@{k} = {avg_recall:.1%} is below the 70% minimum. "
            "This may indicate a quantizer regression."
        )

    def test_brute_force_recall_bits2(self, tmp_path):
        """bits=2 brute-force should achieve ≥55 % Recall@10 (lower bits = higher distortion)."""
        rng = np.random.default_rng(0)
        n, d, k = 2000, 64, 10
        corpus = rng.standard_normal((n, d)).astype(np.float32)
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)

        db = Database.open(str(tmp_path / "db"), d, bits=2, seed=42, metric="ip")
        ids = [f"v{i}" for i in range(n)]
        db.insert_batch(ids, corpus)

        queries = rng.standard_normal((20, d)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        recalls = []
        for q in queries:
            gt = self._ground_truth(corpus, q, k)
            results = db.search(q, top_k=k)
            retrieved = {r["id"] for r in results}
            recalls.append(len(gt & retrieved) / k)

        avg_recall = float(np.mean(recalls))
        assert avg_recall >= 0.55, (
            f"Brute-force Recall@{k} = {avg_recall:.1%} is below the 55% minimum for bits=2. "
            "This may indicate a quantizer regression."
        )


    def test_fallback_stub_raises_runtime_error(self):
        """The fallback Database stub (used when extension is unavailable) must raise."""
        import importlib, sys

        # Temporarily hide the real extension to force the fallback path
        real_mod = sys.modules.pop("tqdb.tqdb", None)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_rag_stub_test",
                "python/tqdb/rag.py",
            )
            mod = importlib.util.module_from_spec(spec)
            # Patch the import inside rag.py to use the stub Database
            mod.__package__ = None  # prevent relative import
            # Execute with tqdb.tqdb hidden → ImportError → stub activated
            orig = sys.modules.get("tqdb.tqdb")
            sys.modules["tqdb.tqdb"] = None  # type: ignore
            try:
                spec.loader.exec_module(mod)
                with pytest.raises(RuntimeError, match="not available"):
                    mod.Database.open("path", 8)
            finally:
                if orig is None:
                    sys.modules.pop("tqdb.tqdb", None)
                else:
                    sys.modules["tqdb.tqdb"] = orig
        finally:
            if real_mod is not None:
                sys.modules["tqdb.tqdb"] = real_mod
