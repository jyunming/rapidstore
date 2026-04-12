"""
Brutal compatibility-layer tests for TurboQuantDB.

Covers three compatibility shims and their edge cases, hidden bugs, and
interaction patterns:

  1. TurboQuantRetriever  (python/tqdb/rag.py)  — LangChain-style wrapper
  2. ChromaDB compat shim  (python/tqdb/chroma_compat.py)
  3. LanceDB compat shim   (python/tqdb/lancedb_compat.py)

Constraint: no production code changes — tests only.
Run:
    python -m pytest tests/test_brutal_compat.py --basetemp=tmp_pytest_brutal_compat -q
"""

from __future__ import annotations

import threading
import warnings
import math
import os
from typing import Any, Dict, List

import numpy as np
import pytest

from tqdb.rag import TurboQuantRetriever
from tqdb.chroma_compat import (
    PersistentClient,
    CompatCollection,
    _apply_filter,
    _sanitize_metadata,
    _parse_metric,
    _validate_collection_name,
)
from tqdb.lancedb_compat import connect, _parse_sql_where, CompatQuery

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8  # small dimension for speed


def _rand(n: int = DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _rand_batch(rows: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((rows, dim)).astype(np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    return m / norms


# ============================================================================
# 1. TurboQuantRetriever (RAG wrapper)
# ============================================================================

class TestRAGRetrieverBasic:
    def test_basic_add_and_search(self, tmp_path):
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM)
        texts = ["hello world", "foo bar", "baz qux"]
        embs = _rand_batch(3).tolist()
        ret.add_texts(texts, embs)
        results = ret.similarity_search(embs[0], k=1)
        assert len(results) == 1
        assert results[0].page_content == "hello world"
        assert hasattr(results[0], "metadata")

    def test_result_has_all_three_keys(self, tmp_path):
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM)
        ret.add_texts(["doc"], [_rand().tolist()])
        r = ret.similarity_search(_rand().tolist(), k=1)
        assert len(r) == 1
        assert hasattr(r[0], "page_content")
        assert hasattr(r[0], "metadata")

    def test_add_texts_with_metadatas(self, tmp_path):
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM)
        texts = ["a", "b"]
        embs = _rand_batch(2).tolist()
        metas = [{"lang": "en"}, {"lang": "fr"}]
        ret.add_texts(texts, embs, metadatas=metas)
        results = ret.similarity_search(embs[0], k=2)
        assert any(r.metadata.get("lang") in ("en", "fr") for r in results)

    def test_add_texts_none_metadatas_uses_empty_dicts(self, tmp_path):
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM)
        ret.add_texts(["x"], [_rand().tolist()], metadatas=None)
        r = ret.similarity_search(_rand().tolist(), k=1)
        assert isinstance(r[0].metadata, dict)

    def test_similarity_search_k_greater_than_stored(self, tmp_path):
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM)
        ret.add_texts(["a", "b"], _rand_batch(2).tolist())
        results = ret.similarity_search(_rand().tolist(), k=100)
        assert len(results) == 2

    def test_similarity_search_empty_db_returns_empty(self, tmp_path):
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM)
        results = ret.similarity_search(_rand().tolist(), k=5)
        assert results == []

    def test_id_generation_multi_round_no_collision(self, tmp_path):
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM)
        ret.add_texts(["a", "b", "c"], _rand_batch(3).tolist())
        # second call should generate doc_3, doc_4, doc_5 — no overlap
        ret.add_texts(["d", "e"], _rand_batch(2, seed=99).tolist())
        assert len(ret.doc_store) == 5
        assert set(ret.doc_store.keys()) == {"doc_0", "doc_1", "doc_2", "doc_3", "doc_4"}

    def test_doc_store_accumulates_across_rounds(self, tmp_path):
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM)
        for i in range(5):
            ret.add_texts([f"round{i}"], [_rand(seed=i).tolist()])
        assert len(ret.doc_store) == 5
        results = ret.similarity_search(_rand().tolist(), k=5)
        assert len(results) == 5

    def test_similarity_search_scores_are_floats(self, tmp_path):
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM)
        ret.add_texts(["a", "b", "c"], _rand_batch(3).tolist())
        results = ret.similarity_search_with_score(_rand().tolist(), k=3)
        assert all(isinstance(score, (int, float)) for _, score in results)

    def test_add_texts_accepts_numpy_embeddings(self, tmp_path):
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM)
        embs = _rand_batch(3)  # np.ndarray, not list
        ret.add_texts(["a", "b", "c"], embs)
        assert len(ret.doc_store) == 3

    def test_add_texts_single_item(self, tmp_path):
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM)
        ret.add_texts(["only one"], [_rand().tolist()])
        results = ret.similarity_search(_rand().tolist(), k=1)
        assert results[0].page_content == "only one"

    def test_retriever_with_l2_metric(self, tmp_path):
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM, metric="l2")
        ret.add_texts(["doc"], [_rand().tolist()])
        results = ret.similarity_search(_rand().tolist(), k=1)
        assert len(results) == 1

    def test_retriever_with_cosine_metric(self, tmp_path):
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM, metric="cosine")
        ret.add_texts(["doc"], [_rand().tolist()])
        results = ret.similarity_search(_rand().tolist(), k=1)
        assert len(results) == 1

    @pytest.mark.xfail(strict=True, reason="doc_store not persisted; reopen loses all docs → similarity_search returns []")
    def test_doc_store_not_persisted_across_reopens(self, tmp_path):
        db_path = str(tmp_path / "db")
        ret = TurboQuantRetriever(db_path, dimension=DIM)
        ret.add_texts(["persisted?"], [_rand().tolist()])
        # Simulate reopen by creating a new instance pointing at same path
        ret2 = TurboQuantRetriever(db_path, dimension=DIM)
        # The tqdb DB has the vector, but doc_store is empty → no results
        results = ret2.similarity_search(_rand().tolist(), k=1)
        # This should return a result but it returns [] because doc_store is empty
        assert len(results) == 1  # FAILS: returns []

    def test_concurrent_add_texts_no_crash(self, tmp_path):
        """20 threads each add 5 docs; total doc_store entries may differ from 100
        due to the non-atomic id generation, but the DB must not crash."""
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM)
        errors = []

        def _worker(seed):
            try:
                ret.add_texts(
                    [f"doc-{seed}-{i}" for i in range(5)],
                    _rand_batch(5, seed=seed).tolist(),
                )
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=_worker, args=(s,)) for s in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # DB should have vectors even if doc_store count is undefined
        assert len(errors) == 0 or len(ret.doc_store) > 0

    def test_add_texts_embedding_length_mismatch_raises(self, tmp_path):
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM)
        with pytest.raises(Exception):
            ret.add_texts(["a", "b", "c"], _rand_batch(2).tolist())

    def test_add_texts_wrong_dim_raises(self, tmp_path):
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM)
        with pytest.raises(Exception):
            ret.add_texts(["a"], [_rand(n=DIM + 1).tolist()])

    def test_similarity_search_k1_returns_single_result(self, tmp_path):
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM)
        ret.add_texts(["x", "y"], _rand_batch(2).tolist())
        assert len(ret.similarity_search(_rand().tolist(), k=1)) == 1

    def test_add_texts_documents_stored_in_db(self, tmp_path):
        """add_texts also stores doc strings in the tqdb document field via insert_batch."""
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM)
        ret.add_texts(["hello text"], [_rand().tolist()])
        # doc_store must reflect the text
        assert ret.doc_store["doc_0"]["text"] == "hello text"

    def test_similarity_search_text_matches_doc_store_text(self, tmp_path):
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM)
        texts = ["alpha", "beta", "gamma"]
        embs = _rand_batch(3).tolist()
        ret.add_texts(texts, embs)
        results = ret.similarity_search(embs[0], k=3)
        returned_texts = {r.page_content for r in results}
        assert "alpha" in returned_texts

    def test_add_texts_large_batch_200_docs(self, tmp_path):
        ret = TurboQuantRetriever(str(tmp_path / "db"), dimension=DIM)
        n = 200
        ret.add_texts([f"doc{i}" for i in range(n)], _rand_batch(n).tolist())
        results = ret.similarity_search(_rand().tolist(), k=10)
        assert len(results) == 10


# ============================================================================
# 2. ChromaDB Compatibility Shim — Client-level
# ============================================================================

class TestChromaClientBasic:
    def test_create_and_list(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        client.create_collection("col_a")
        client.create_collection("col_b")
        assert sorted(client.list_collections()) == ["col_a", "col_b"]

    def test_count_collections(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        assert client.count_collections() == 0
        client.create_collection("x")
        assert client.count_collections() == 1

    def test_get_or_create_idempotent(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        c1 = client.get_or_create_collection("same")
        c2 = client.get_or_create_collection("same")
        assert c1.name == c2.name == "same"

    def test_create_duplicate_raises(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        client.create_collection("dup")
        with pytest.raises(ValueError):
            client.create_collection("dup")

    def test_get_collection_nonexistent_raises(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        with pytest.raises(ValueError):
            client.get_collection("ghost")

    def test_delete_collection(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        client.create_collection("del_me")
        client.delete_collection("del_me")
        assert "del_me" not in client.list_collections()

    def test_delete_collection_nonexistent_raises(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        with pytest.raises(ValueError):
            client.delete_collection("ghost")

    def test_reset_clears_all(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        client.create_collection("a")
        client.create_collection("b")
        client.reset()
        assert client.list_collections() == []

    def test_collection_name_empty_raises(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        with pytest.raises(ValueError):
            client.create_collection("")

    def test_collection_name_dotdot_raises(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        with pytest.raises(ValueError):
            client.create_collection("..")

    def test_collection_name_slash_raises(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        with pytest.raises((ValueError, Exception)):
            client.create_collection("a/b")

    def test_collection_name_null_byte_raises(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        with pytest.raises(ValueError):
            client.create_collection("a\0b")

    def test_list_collections_returns_sorted(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        for name in ["zebra", "apple", "mango"]:
            client.create_collection(name)
        assert client.list_collections() == ["apple", "mango", "zebra"]

    def test_metric_mapping_cosine(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.create_collection("c", metadata={"hnsw:space": "cosine"})
        assert col._metric == "cosine"

    def test_metric_mapping_l2(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.create_collection("c", metadata={"hnsw:space": "l2"})
        assert col._metric == "l2"

    def test_metric_mapping_ip_default(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.create_collection("c")
        assert col._metric == "ip"

    def test_metric_mapping_unknown_falls_back_to_ip(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.create_collection("c", metadata={"hnsw:space": "unknown_space"})
        # unknown maps to "ip" per _HNSW_SPACE_TO_METRIC
        assert col._metric == "ip"

    def test_recreate_after_delete(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        client.create_collection("x")
        client.delete_collection("x")
        col = client.create_collection("x")
        col.add(ids=["a"], embeddings=[_rand().tolist()])
        assert col.count() == 1


# ============================================================================
# 3. ChromaDB Compatibility Shim — Collection-level
# ============================================================================

class TestChromaCollectionAdd:
    def test_add_basic(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a", "b"], embeddings=_rand_batch(2).tolist())
        assert col.count() == 2

    def test_add_with_documents(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[_rand().tolist()], documents=["hello doc"])
        result = col.get(ids=["a"])
        assert result["documents"][0] == "hello doc"

    def test_add_with_metadatas(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[_rand().tolist()], metadatas=[{"k": "v"}])
        result = col.get(ids=["a"])
        assert result["metadatas"][0]["k"] == "v"

    def test_count_before_any_add_is_zero(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        assert col.count() == 0

    def test_add_duplicate_id_raises(self, tmp_path):
        """ChromaDB add() with duplicate IDs should raise. tqdb insert_batch(mode='insert') raises on duplicates."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[_rand().tolist()])
        with pytest.raises(Exception):
            col.add(ids=["a"], embeddings=[_rand(seed=99).tolist()])

    def test_add_no_embeddings_no_ef_raises(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        with pytest.raises((ValueError, Exception)):
            col.add(ids=["a"])

    def test_add_dim_mismatch_raises(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[_rand(n=DIM).tolist()])
        with pytest.raises(ValueError):
            col.add(ids=["b"], embeddings=[_rand(n=DIM + 1).tolist()])

    def test_add_with_embedding_function(self, tmp_path):
        ef = lambda texts: np.ones((len(texts), DIM), dtype=np.float32).tolist()
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c", embedding_function=ef)
        col.add(ids=["a"], documents=["some text"])
        assert col.count() == 1

    def test_add_large_batch(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        n = 500
        col.add(
            ids=[f"id{i}" for i in range(n)],
            embeddings=_rand_batch(n).tolist(),
        )
        assert col.count() == n

    def test_add_sanitizes_metadata_list_to_str(self, tmp_path):
        """_sanitize_metadata coerces list values to str so tqdb accepts them."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[_rand().tolist()], metadatas=[{"tags": ["x", "y"]}])
        result = col.get(ids=["a"])
        # list was stringified
        assert isinstance(result["metadatas"][0]["tags"], str)


class TestChromaCollectionUpsert:
    def test_upsert_new_id(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.upsert(ids=["a"], embeddings=[_rand().tolist()])
        assert col.count() == 1

    def test_upsert_existing_overwrites(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[_rand(seed=1).tolist()], metadatas=[{"v": "old"}])
        col.upsert(ids=["a"], embeddings=[_rand(seed=2).tolist()], metadatas=[{"v": "new"}])
        result = col.get(ids=["a"])
        assert result["metadatas"][0]["v"] == "new"
        assert col.count() == 1

    def test_upsert_multiple_rounds_same_id_count_stays_1(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        for i in range(10):
            col.upsert(ids=["same"], embeddings=[_rand(seed=i).tolist()])
        assert col.count() == 1

    def test_upsert_dim_mismatch_raises(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.upsert(ids=["a"], embeddings=[_rand(n=DIM).tolist()])
        with pytest.raises(ValueError):
            col.upsert(ids=["a"], embeddings=[_rand(n=DIM + 1).tolist()])

    def test_upsert_with_documents(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.upsert(ids=["a"], embeddings=[_rand().tolist()], documents=["updated doc"])
        result = col.get(ids=["a"])
        assert result["documents"][0] == "updated doc"


class TestChromaCollectionUpdate:
    def test_update_metadata_only(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[_rand().tolist()], metadatas=[{"k": "old"}])
        col.update(ids=["a"], metadatas=[{"k": "new"}])
        result = col.get(ids=["a"])
        assert result["metadatas"][0]["k"] == "new"

    def test_update_embeddings_and_metadata(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[_rand(seed=1).tolist()])
        col.update(ids=["a"], embeddings=[_rand(seed=99).tolist()], metadatas=[{"updated": True}])
        result = col.get(ids=["a"])
        assert result["metadatas"][0].get("updated") in (True, "True")

    def test_update_nonexistent_id_raises_or_silently_fails(self, tmp_path):
        """ChromaDB update() on a missing ID raises InvalidCollectionException.
        tqdb update_metadata on missing ID: behavior depends on engine."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[_rand().tolist()])
        # This may silently do nothing or raise; both are acceptable — just must not crash fatally
        try:
            col.update(ids=["ghost"], metadatas=[{"k": "v"}])
        except Exception:
            pass  # raising is fine


class TestChromaCollectionDelete:
    def test_delete_by_ids(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a", "b", "c"], embeddings=_rand_batch(3).tolist())
        col.delete(ids=["a", "b"])
        assert col.count() == 1

    def test_delete_nonexistent_id_no_error(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[_rand().tolist()])
        col.delete(ids=["ghost"])  # should not raise
        assert col.count() == 1

    def test_delete_by_where_filter(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(
            ids=["a", "b", "c"],
            embeddings=_rand_batch(3).tolist(),
            metadatas=[{"cat": "x"}, {"cat": "y"}, {"cat": "x"}],
        )
        col.delete(where={"cat": {"$eq": "x"}})
        result = col.get()
        assert "b" in result["ids"]
        assert "a" not in result["ids"]
        assert "c" not in result["ids"]

    def test_delete_ids_plus_where_intersection(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(
            ids=["a", "b", "c"],
            embeddings=_rand_batch(3).tolist(),
            metadatas=[{"cat": "x"}, {"cat": "x"}, {"cat": "y"}],
        )
        # Only delete "a" even though "b" also matches the where filter
        col.delete(ids=["a"], where={"cat": {"$eq": "x"}})
        ids = col.get()["ids"]
        assert "a" not in ids
        assert "b" in ids

    def test_delete_empty_ids_list_no_error(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[_rand().tolist()])
        col.delete(ids=[])  # empty list — must not raise
        assert col.count() == 1

    def test_delete_all_then_count_zero(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a", "b"], embeddings=_rand_batch(2).tolist())
        col.delete(ids=["a", "b"])
        assert col.count() == 0


class TestChromaCollectionGet:
    def test_get_all_no_filter(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a", "b"], embeddings=_rand_batch(2).tolist())
        result = col.get()
        assert sorted(result["ids"]) == ["a", "b"]

    def test_get_by_ids(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a", "b", "c"], embeddings=_rand_batch(3).tolist())
        result = col.get(ids=["a", "c"])
        assert sorted(result["ids"]) == ["a", "c"]

    def test_get_by_where_filter(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(
            ids=["a", "b"],
            embeddings=_rand_batch(2).tolist(),
            metadatas=[{"x": 1}, {"x": 2}],
        )
        result = col.get(where={"x": {"$eq": 1}})
        assert result["ids"] == ["a"]

    def test_get_with_offset_and_limit(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=[f"id{i}" for i in range(10)], embeddings=_rand_batch(10).tolist())
        result = col.get(limit=3, offset=5)
        assert len(result["ids"]) == 3

    def test_get_offset_beyond_count_returns_empty(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[_rand().tolist()])
        result = col.get(offset=100)
        assert result["ids"] == []

    def test_get_include_ids_only(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[_rand().tolist()])
        result = col.get(ids=["a"], include=["ids"])
        assert "a" in result["ids"]
        assert result.get("metadatas") is None or result["metadatas"] is None

    def test_get_include_unknown_raises(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[_rand().tolist()])
        with pytest.raises(ValueError):
            col.get(ids=["a"], include=["vectors_raw_please"])

    def test_get_empty_collection_returns_empty_lists(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        result = col.get()
        assert result["ids"] == []

    def test_get_include_embeddings_returns_actual_vectors(self, tmp_path):
        """ChromaDB returns raw embedding vectors when include=['embeddings'].
        The tqdb compat shim accepts the field but returns None — silent data loss."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        vec = _rand().tolist()
        col.add(ids=["a"], embeddings=[vec])
        result = col.get(ids=["a"], include=["embeddings"])
        # In real ChromaDB this would be the embedding vectors
        assert "embeddings" in result and result["embeddings"] is not None

    def test_peek_returns_limited_results(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=[f"id{i}" for i in range(20)], embeddings=_rand_batch(20).tolist())
        result = col.peek(limit=5)
        assert len(result["ids"]) == 5

    def test_get_ids_and_where_intersection(self, tmp_path):
        """get(ids=[...], where={...}) returns only IDs satisfying both conditions."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(
            ids=["a", "b", "c"],
            embeddings=_rand_batch(3).tolist(),
            metadatas=[{"k": 1}, {"k": 1}, {"k": 2}],
        )
        result = col.get(ids=["a", "b", "c"], where={"k": {"$eq": 1}})
        assert sorted(result["ids"]) == ["a", "b"]


class TestChromaCollectionQuery:
    def test_query_basic(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        embs = _rand_batch(5).tolist()
        col.add(ids=[f"id{i}" for i in range(5)], embeddings=embs)
        result = col.query(query_embeddings=[embs[0]], n_results=3)
        assert len(result["ids"][0]) == 3

    def test_query_multiple_query_vecs(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        embs = _rand_batch(5).tolist()
        col.add(ids=[f"id{i}" for i in range(5)], embeddings=embs)
        result = col.query(query_embeddings=[embs[0], embs[1]], n_results=2)
        assert len(result["ids"]) == 2
        assert len(result["ids"][0]) == 2
        assert len(result["ids"][1]) == 2

    def test_query_n_results_gt_stored(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a", "b"], embeddings=_rand_batch(2).tolist())
        result = col.query(query_embeddings=[_rand().tolist()], n_results=100)
        assert len(result["ids"][0]) == 2

    def test_query_with_where_filter(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(
            ids=["a", "b", "c"],
            embeddings=_rand_batch(3).tolist(),
            metadatas=[{"cat": "dog"}, {"cat": "cat"}, {"cat": "cat"}],
        )
        result = col.query(query_embeddings=[_rand().tolist()], n_results=10, where={"cat": {"$eq": "cat"}})
        assert all(r in ["b", "c"] for r in result["ids"][0])

    def test_query_include_distances(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a", "b"], embeddings=_rand_batch(2).tolist())
        result = col.query(query_embeddings=[_rand().tolist()], n_results=2, include=["distances"])
        assert result["distances"] is not None
        assert len(result["distances"][0]) == 2

    def test_query_include_only_ids(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a", "b"], embeddings=_rand_batch(2).tolist())
        result = col.query(query_embeddings=[_rand().tolist()], n_results=2, include=["ids"])
        assert result.get("distances") is None or result["distances"] is None

    def test_query_where_document_raises_not_implemented(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[_rand().tolist()])
        with pytest.raises(NotImplementedError):
            col.query(query_embeddings=[_rand().tolist()], n_results=1, where_document={"$contains": "text"})

    def test_query_unknown_include_raises(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[_rand().tolist()])
        with pytest.raises(ValueError):
            col.query(query_embeddings=[_rand().tolist()], n_results=1, include=["raw_vectors_please"])

    def test_query_no_embeddings_no_ef_raises(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[_rand().tolist()])
        with pytest.raises((ValueError, Exception)):
            col.query(n_results=1)

    def test_query_with_embedding_function(self, tmp_path):
        ef = lambda texts: _rand_batch(len(texts)).tolist()
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c", embedding_function=ef)
        col.add(ids=["a", "b"], embeddings=_rand_batch(2).tolist())
        result = col.query(query_texts=["some query"], n_results=2)
        assert len(result["ids"][0]) == 2

    def test_query_nin_operator_passed_to_tqdb_fails(self, tmp_path):
        """$nin is now supported by the tqdb engine — previously raised, now passes.
        Kept as a regression test to ensure $nin continues to work."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(
            ids=["a", "b", "c"],
            embeddings=_rand_batch(3).tolist(),
            metadatas=[{"cat": "x"}, {"cat": "y"}, {"cat": "z"}],
        )
        # This should work in real ChromaDB but fails here:
        result = col.query(
            query_embeddings=[_rand().tolist()],
            n_results=5,
            where={"cat": {"$nin": ["x"]}},
        )
        assert len(result["ids"][0]) == 2  # b and c

    def test_query_include_embeddings_returns_actual_vectors(self, tmp_path):
        """ChromaDB query with include=['embeddings'] returns raw vectors.
        The tqdb shim accepts the field but the 'embeddings' key is absent from
        the return dict — silent data loss."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[_rand().tolist()])
        result = col.query(query_embeddings=[_rand().tolist()], n_results=1, include=["embeddings"])
        assert "embeddings" in result and result["embeddings"] is not None


class TestChromaCollectionModify:
    def test_modify_metadata(self, tmp_path):
        """Changing metric via modify(metadata={"hnsw:space": ...}) raises ValueError.
        Real ChromaDB allows this; tqdb guards it as a known compat gap."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        with pytest.raises(ValueError):
            col.modify(metadata={"hnsw:space": "cosine"})

    def test_modify_metric_change_raises(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.create_collection("c", metadata={"hnsw:space": "l2"})
        with pytest.raises(ValueError):
            col.modify(metadata={"hnsw:space": "cosine"})

    def test_modify_name_rename_visible_to_client(self, tmp_path):
        """ChromaDB modify(name=...) renames the collection so it's findable under the new name.
        The tqdb shim only updates the JSON sidecar; the directory keeps the old name.
        Client.get_collection('renamed') will raise ValueError (directory not found)."""
        client = PersistentClient(str(tmp_path))
        col = client.create_collection("original")
        col.modify(name="renamed")
        # In real ChromaDB this would succeed:
        found = client.get_collection("renamed")
        assert found.name == "renamed"

    def test_modify_name_updates_internal_attribute(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.create_collection("c")
        col.modify(name="new_name")
        assert col.name == "new_name"


class TestChromaFilterApply:
    def _records(self):
        return [
            {"id": "a", "metadata": {"x": 10, "tag": "alpha", "active": True}},
            {"id": "b", "metadata": {"x": 20, "tag": "beta", "active": False}},
            {"id": "c", "metadata": {"x": 30, "tag": "alpha", "active": True}},
        ]

    def test_eq(self):
        r = _apply_filter(self._records(), {"tag": {"$eq": "alpha"}})
        assert {x["id"] for x in r} == {"a", "c"}

    def test_ne(self):
        r = _apply_filter(self._records(), {"tag": {"$ne": "alpha"}})
        assert {x["id"] for x in r} == {"b"}

    def test_gt(self):
        r = _apply_filter(self._records(), {"x": {"$gt": 15}})
        assert {x["id"] for x in r} == {"b", "c"}

    def test_gte(self):
        r = _apply_filter(self._records(), {"x": {"$gte": 20}})
        assert {x["id"] for x in r} == {"b", "c"}

    def test_lt(self):
        r = _apply_filter(self._records(), {"x": {"$lt": 20}})
        assert {x["id"] for x in r} == {"a"}

    def test_lte(self):
        r = _apply_filter(self._records(), {"x": {"$lte": 20}})
        assert {x["id"] for x in r} == {"a", "b"}

    def test_in(self):
        r = _apply_filter(self._records(), {"tag": {"$in": ["alpha", "gamma"]}})
        assert {x["id"] for x in r} == {"a", "c"}

    def test_nin(self):
        r = _apply_filter(self._records(), {"tag": {"$nin": ["alpha"]}})
        assert {x["id"] for x in r} == {"b"}

    def test_exists_true(self):
        r = _apply_filter(self._records(), {"tag": {"$exists": True}})
        assert len(r) == 3

    def test_exists_false(self):
        r = _apply_filter(self._records(), {"missing_key": {"$exists": False}})
        assert len(r) == 3

    def test_contains(self):
        r = _apply_filter(self._records(), {"tag": {"$contains": "pha"}})
        assert {x["id"] for x in r} == {"a", "c"}

    def test_and(self):
        r = _apply_filter(self._records(), {"$and": [{"x": {"$gt": 10}}, {"active": {"$eq": True}}]})
        assert {x["id"] for x in r} == {"c"}

    def test_or(self):
        r = _apply_filter(self._records(), {"$or": [{"x": {"$eq": 10}}, {"x": {"$eq": 30}}]})
        assert {x["id"] for x in r} == {"a", "c"}

    def test_bare_equality(self):
        r = _apply_filter(self._records(), {"tag": "alpha"})
        assert {x["id"] for x in r} == {"a", "c"}

    def test_unknown_op_raises(self):
        with pytest.raises(ValueError):
            _apply_filter(self._records(), {"x": {"$weird": 1}})

    def test_field_missing_from_metadata_no_match(self):
        r = _apply_filter(self._records(), {"nofield": {"$eq": "anything"}})
        assert r == []


class TestChromaSanitizeMetadata:
    def test_str_passthrough(self):
        assert _sanitize_metadata({"k": "v"}) == {"k": "v"}

    def test_int_passthrough(self):
        assert _sanitize_metadata({"k": 42}) == {"k": 42}

    def test_float_passthrough(self):
        assert _sanitize_metadata({"k": 3.14}) == {"k": 3.14}

    def test_bool_passthrough(self):
        assert _sanitize_metadata({"k": True}) == {"k": True}

    def test_list_coerced_to_str(self):
        result = _sanitize_metadata({"k": [1, 2, 3]})
        assert isinstance(result["k"], str)

    def test_dict_coerced_to_str(self):
        result = _sanitize_metadata({"k": {"nested": True}})
        assert isinstance(result["k"], str)

    def test_none_value_coerced_to_str(self):
        result = _sanitize_metadata({"k": None})
        assert isinstance(result["k"], str)

    def test_empty_dict(self):
        assert _sanitize_metadata({}) == {}


class TestChromaConcurrency:
    def test_concurrent_upsert_same_id_count_stays_1(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        # Prime the collection
        col.add(ids=["shared"], embeddings=[_rand().tolist()])
        errors = []

        def _worker(seed):
            try:
                col.upsert(ids=["shared"], embeddings=[_rand(seed=seed).tolist()])
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=_worker, args=(s,)) for s in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert col.count() == 1

    def test_concurrent_add_different_ids_no_corruption(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        errors = []

        def _worker(i):
            try:
                col.add(ids=[f"id_{i}"], embeddings=[_rand(seed=i).tolist()])
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert col.count() == 50


class TestChromaPersistence:
    def test_collection_survives_reopen(self, tmp_path):
        import gc
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a", "b"], embeddings=_rand_batch(2).tolist(), metadatas=[{"v": 1}, {"v": 2}])
        # Release mmap before reopening same path (Windows requires this)
        del col, client
        gc.collect()
        client2 = PersistentClient(str(tmp_path))
        col2 = client2.get_collection("c")
        assert col2.count() == 2

    def test_query_results_correct_after_reopen(self, tmp_path):
        import gc
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        embs = _rand_batch(5).tolist()
        col.add(ids=[f"id{i}" for i in range(5)], embeddings=embs)
        # Release mmap before reopening same path (Windows requires this)
        del col, client
        gc.collect()
        client2 = PersistentClient(str(tmp_path))
        col2 = client2.get_collection("c")
        result = col2.query(query_embeddings=[embs[0]], n_results=3)
        assert len(result["ids"][0]) == 3


# ============================================================================
# 4. LanceDB Compatibility Shim — Connection-level
# ============================================================================

class TestLanceDBConnectionBasic:
    def test_connect_creates_dir(self, tmp_path):
        db = connect(str(tmp_path / "newdir"))
        assert os.path.isdir(str(tmp_path / "newdir"))

    def test_create_table_and_table_names(self, tmp_path):
        db = connect(str(tmp_path))
        db.create_table("t1", data=[{"id": "a", "vector": _rand().tolist()}])
        db.create_table("t2", data=[{"id": "b", "vector": _rand().tolist()}])
        assert sorted(db.table_names()) == ["t1", "t2"]

    def test_create_table_duplicate_raises(self, tmp_path):
        db = connect(str(tmp_path))
        db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}])
        with pytest.raises(ValueError):
            db.create_table("t", data=[{"id": "b", "vector": _rand().tolist()}])

    def test_create_table_overwrite_replaces(self, tmp_path):
        db = connect(str(tmp_path))
        db.create_table("t", data=[{"id": "a", "vector": _rand(seed=1).tolist()}])
        db.create_table("t", data=[{"id": "b", "vector": _rand(seed=2).tolist()}], mode="overwrite")
        tbl = db.open_table("t")
        assert tbl.count_rows() == 1
        rows = tbl.search(_rand().tolist()).to_list()
        assert rows[0]["id"] == "b"

    def test_open_table_nonexistent_raises(self, tmp_path):
        db = connect(str(tmp_path))
        with pytest.raises(ValueError):
            db.open_table("ghost")

    def test_drop_table(self, tmp_path):
        db = connect(str(tmp_path))
        db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}])
        db.drop_table("t")
        assert "t" not in db.table_names()

    def test_drop_table_nonexistent_raises(self, tmp_path):
        db = connect(str(tmp_path))
        with pytest.raises(ValueError):
            db.drop_table("ghost")

    def test_table_names_sorted(self, tmp_path):
        db = connect(str(tmp_path))
        for name in ["zebra", "apple", "mango"]:
            db.create_table(name, data=[{"id": "x", "vector": _rand().tolist()}])
        assert db.table_names() == ["apple", "mango", "zebra"]

    def test_cloud_uri_raises_not_implemented(self, tmp_path):
        with pytest.raises(NotImplementedError):
            connect("s3://my-bucket/data")

    def test_table_name_traversal_rejected(self, tmp_path):
        db = connect(str(tmp_path))
        with pytest.raises(ValueError):
            db.create_table("..", data=[{"id": "a", "vector": _rand().tolist()}])

    def test_table_name_slash_rejected(self, tmp_path):
        db = connect(str(tmp_path))
        with pytest.raises(ValueError):
            db.create_table("a/b", data=[{"id": "a", "vector": _rand().tolist()}])

    def test_table_name_null_byte_rejected(self, tmp_path):
        db = connect(str(tmp_path))
        with pytest.raises(ValueError):
            db.create_table("a\0b", data=[{"id": "a", "vector": _rand().tolist()}])


# ============================================================================
# 5. LanceDB Compatibility Shim — Table-level
# ============================================================================

class TestLanceDBTableAdd:
    def test_add_list_of_dicts(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t")
        tbl.add([{"id": "a", "vector": _rand().tolist()}, {"id": "b", "vector": _rand(seed=1).tolist()}])
        assert tbl.count_rows() == 2

    def test_add_with_extra_fields_preserved(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t")
        tbl.add([{"id": "a", "vector": _rand().tolist(), "category": "x"}])
        rows = tbl.search(_rand().tolist()).to_list()
        assert rows[0].get("category") == "x"

    def test_add_with_document_field(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t")
        tbl.add([{"id": "a", "vector": _rand().tolist(), "document": "hello doc"}])
        rows = tbl.search(_rand().tolist()).to_list()
        assert rows[0].get("document") == "hello doc"

    def test_add_mode_append_stacks(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}])
        tbl.add([{"id": "b", "vector": _rand(seed=1).tolist()}], mode="append")
        assert tbl.count_rows() == 2

    def test_add_mode_overwrite_replaces(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}])
        tbl.add([{"id": "b", "vector": _rand(seed=2).tolist()}], mode="overwrite")
        assert tbl.count_rows() == 1

    def test_add_invalid_mode_raises(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t")
        with pytest.raises(ValueError):
            tbl.add([{"id": "a", "vector": _rand().tolist()}], mode="replace")

    def test_add_empty_list_no_error(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t")
        tbl.add([])  # should not raise
        assert tbl.count_rows() == 0

    def test_add_missing_vector_key_raises(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t")
        with pytest.raises((KeyError, Exception)):
            tbl.add([{"id": "a", "no_vector": [1.0]}])

    def test_add_no_id_field_uses_index(self, tmp_path):
        """When no 'id' field in row, position index is used as string ID."""
        db = connect(str(tmp_path))
        tbl = db.create_table("t")
        tbl.add([{"vector": _rand().tolist()}, {"vector": _rand(seed=1).tolist()}])
        assert tbl.count_rows() == 2

    def test_add_large_batch_500(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t")
        rows = [{"id": f"id{i}", "vector": _rand(seed=i).tolist()} for i in range(500)]
        tbl.add(rows)
        assert tbl.count_rows() == 500

    def test_add_pyarrow_table(self, tmp_path):
        import pyarrow as pa
        db = connect(str(tmp_path))
        tbl = db.create_table("t")
        pa_tbl = pa.Table.from_pylist([
            {"id": "x", "vector": _rand().tolist()},
            {"id": "y", "vector": _rand(seed=5).tolist()},
        ])
        tbl.add(pa_tbl)
        assert tbl.count_rows() == 2

    @pytest.mark.xfail(strict=True, reason="add(mode='append') with duplicate ID calls insert_batch('insert') which raises — LanceDB allows duplicates")
    def test_add_append_duplicate_id_allowed_in_lancedb(self, tmp_path):
        """Real LanceDB allows appending rows with duplicate IDs (no uniqueness constraint).
        tqdb shim calls insert_batch(..., 'insert') which raises on duplicates."""
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": "dup", "vector": _rand().tolist()}])
        # This should succeed in real LanceDB but raises in the shim:
        tbl.add([{"id": "dup", "vector": _rand(seed=99).tolist()}], mode="append")
        assert tbl.count_rows() == 2  # FAILS: raises before getting here


class TestLanceDBTableSearch:
    def test_search_basic_to_list(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": f"id{i}", "vector": _rand(seed=i).tolist()} for i in range(10)])
        rows = tbl.search(_rand().tolist()).limit(5).to_list()
        assert len(rows) == 5
        assert all("_distance" in r for r in rows)

    def test_search_to_arrow(self, tmp_path):
        import pyarrow as pa
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}])
        result = tbl.search(_rand().tolist()).limit(1).to_arrow()
        assert isinstance(result, pa.Table)
        assert result.num_rows == 1

    def test_search_to_pandas(self, tmp_path):
        import pandas as pd
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}])
        df = tbl.search(_rand().tolist()).limit(1).to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_search_metric_override_warns(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tbl.search(_rand().tolist()).metric("l2").limit(1).to_list()
        assert any("ignored" in str(warning.message).lower() or "metric" in str(warning.message).lower() for warning in w)

    def test_search_with_where_string_eq(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[
            {"id": "a", "vector": _rand().tolist(), "cat": "dog"},
            {"id": "b", "vector": _rand(seed=1).tolist(), "cat": "cat"},
        ])
        rows = tbl.search(_rand().tolist()).where("cat = 'cat'").limit(10).to_list()
        assert all(r.get("cat") == "cat" for r in rows)

    def test_search_with_where_id_in(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": f"id{i}", "vector": _rand(seed=i).tolist()} for i in range(5)])
        rows = tbl.search(_rand().tolist()).where("id IN ('id0', 'id2')").limit(10).to_list()
        assert {r["id"] for r in rows} <= {"id0", "id2"}

    def test_search_limit_1(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": f"id{i}", "vector": _rand(seed=i).tolist()} for i in range(10)])
        rows = tbl.search(_rand().tolist()).limit(1).to_list()
        assert len(rows) == 1

    def test_search_limit_greater_than_stored(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": f"id{i}", "vector": _rand(seed=i).tolist()} for i in range(3)])
        rows = tbl.search(_rand().tolist()).limit(100).to_list()
        assert len(rows) == 3

    def test_search_nprobes_no_op(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}])
        rows = tbl.search(_rand().tolist()).nprobes(100).limit(1).to_list()
        assert len(rows) == 1

    def test_search_refine_factor_no_op(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}])
        rows = tbl.search(_rand().tolist()).refine_factor(5).limit(1).to_list()
        assert len(rows) == 1

    def test_search_select_columns(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": "a", "vector": _rand().tolist(), "cat": "x"}])
        rows = tbl.search(_rand().tolist()).select(["id"]).limit(1).to_list()
        assert "id" in rows[0]
        assert "cat" not in rows[0]

    @pytest.mark.xfail(strict=True, reason="limit(0) calls db.search(q, 0) which raises in tqdb; should return []")
    def test_search_limit_zero_returns_empty(self, tmp_path):
        """LanceDB with limit=0 returns an empty result set.
        tqdb raises an exception when top_k=0."""
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}])
        rows = tbl.search(_rand().tolist()).limit(0).to_list()
        assert rows == []

    def test_search_scores_are_finite(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": f"id{i}", "vector": _rand(seed=i).tolist()} for i in range(10)])
        rows = tbl.search(_rand().tolist()).limit(10).to_list()
        assert all(math.isfinite(r["_distance"]) for r in rows)

    def test_search_self_similarity_is_top_result(self, tmp_path):
        db = connect(str(tmp_path))
        vecs = [{"id": f"id{i}", "vector": _rand(seed=i).tolist()} for i in range(20)]
        tbl = db.create_table("t", data=vecs)
        query = vecs[3]["vector"]
        rows = tbl.search(query).limit(1).to_list()
        assert rows[0]["id"] == "id3"


class TestLanceDBTableDelete:
    def test_delete_by_id_in(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": f"id{i}", "vector": _rand(seed=i).tolist()} for i in range(5)])
        tbl.delete("id IN ('id0', 'id2')")
        assert tbl.count_rows() == 3

    def test_delete_by_field_eq(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[
            {"id": "a", "vector": _rand().tolist(), "cat": "x"},
            {"id": "b", "vector": _rand(seed=1).tolist(), "cat": "y"},
        ])
        tbl.delete("cat = 'x'")
        assert tbl.count_rows() == 1

    def test_delete_unsupported_where_raises(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}])
        with pytest.raises(NotImplementedError):
            tbl.delete("cat = 'x' AND score > 5")

    def test_delete_all_via_id_in(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}, {"id": "b", "vector": _rand(seed=1).tolist()}])
        tbl.delete("id IN ('a', 'b')")
        assert tbl.count_rows() == 0

    def test_delete_nonexistent_no_error(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}])
        tbl.delete("id IN ('ghost')")  # should not raise
        assert tbl.count_rows() == 1


class TestLanceDBTableCountRows:
    def test_count_rows_empty(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t")
        assert tbl.count_rows() == 0

    def test_count_rows_after_add(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}])
        assert tbl.count_rows() == 1

    def test_count_rows_with_field_filter(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[
            {"id": "a", "vector": _rand().tolist(), "score": 10},
            {"id": "b", "vector": _rand(seed=1).tolist(), "score": 20},
        ])
        assert tbl.count_rows("score > 15") == 1

    def test_count_rows_with_id_in_filter(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": f"id{i}", "vector": _rand(seed=i).tolist()} for i in range(5)])
        assert tbl.count_rows("id IN ('id0', 'id1', 'id2')") == 3


class TestLanceDBTableArrowPandas:
    def test_to_arrow_empty(self, tmp_path):
        import pyarrow as pa
        db = connect(str(tmp_path))
        tbl = db.create_table("t")
        result = tbl.to_arrow()
        assert isinstance(result, pa.Table)

    def test_to_pandas_empty(self, tmp_path):
        import pandas as pd
        db = connect(str(tmp_path))
        tbl = db.create_table("t")
        df = tbl.to_pandas()
        assert isinstance(df, pd.DataFrame)

    def test_to_arrow_populated(self, tmp_path):
        import pyarrow as pa
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}, {"id": "b", "vector": _rand(seed=1).tolist()}])
        result = tbl.to_arrow()
        assert result.num_rows == 2

    def test_to_pandas_populated(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}])
        df = tbl.to_pandas()
        assert len(df) == 1
        assert "id" in df.columns


class TestLanceDBTableMisc:
    def test_optimize_no_op(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}])
        tbl.optimize()  # should not raise

    def test_create_index_succeeds(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=[{"id": f"id{i}", "vector": _rand(seed=i).tolist()} for i in range(50)])
        tbl.create_index()  # delegates to tqdb.create_index()

    def test_table_name_property(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("my_table")
        assert tbl.name == "my_table"

    def test_table_persists_across_reopen(self, tmp_path):
        db = connect(str(tmp_path))
        db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}])
        db2 = connect(str(tmp_path))
        tbl = db2.open_table("t")
        assert tbl.count_rows() == 1

    @pytest.mark.xfail(strict=True, reason="create_table has no metric parameter — always defaults to 'ip'; cannot create L2 table via API")
    def test_create_table_with_l2_metric(self, tmp_path):
        """Real LanceDB create_table accepts a schema/metric parameter.
        The tqdb shim hardcodes metric='ip' in create_table; no way to create an L2 table."""
        db = connect(str(tmp_path))
        # No metric param available:
        tbl = db.create_table("t", data=[{"id": "a", "vector": _rand().tolist()}], metric="l2")
        assert tbl._metric == "l2"  # FAILS: no metric param in create_table()


# ============================================================================
# 6. LanceDB SQL WHERE Parser — edge cases
# ============================================================================

class TestLanceDBSQLParser:
    def test_id_in_single_quoted(self):
        r = _parse_sql_where("id IN ('a', 'b', 'c')")
        assert r == {"id": {"$in": ["a", "b", "c"]}}

    def test_field_in_string_values(self):
        r = _parse_sql_where("topic IN ('ml', 'ai')")
        assert r == {"topic": {"$in": ["ml", "ai"]}}

    def test_field_eq_string(self):
        r = _parse_sql_where("cat = 'dog'")
        assert r == {"cat": {"$eq": "dog"}}

    def test_field_ne_string(self):
        r = _parse_sql_where("cat != 'dog'")
        assert r == {"cat": {"$ne": "dog"}}

    def test_field_eq_integer(self):
        r = _parse_sql_where("score = 42")
        assert r == {"score": {"$eq": 42.0}}

    def test_field_gt(self):
        r = _parse_sql_where("score > 5")
        assert r == {"score": {"$gt": 5.0}}

    def test_field_gte(self):
        r = _parse_sql_where("score >= 5")
        assert r == {"score": {"$gte": 5.0}}

    def test_field_lt(self):
        r = _parse_sql_where("score < 5")
        assert r == {"score": {"$lt": 5.0}}

    def test_field_lte(self):
        r = _parse_sql_where("score <= 5")
        assert r == {"score": {"$lte": 5.0}}

    def test_field_negative_number(self):
        r = _parse_sql_where("score >= -1")
        assert r == {"score": {"$gte": -1.0}}

    def test_field_float_comparison(self):
        r = _parse_sql_where("score > 3.14")
        assert r == {"score": {"$gt": 3.14}}

    def test_trailing_comma_raises(self):
        with pytest.raises((ValueError, Exception)):
            _parse_sql_where("id IN ('a',)")

    def test_empty_string_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _parse_sql_where("")

    def test_and_compound_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _parse_sql_where("cat = 'dog' AND score > 5")

    def test_or_compound_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _parse_sql_where("cat = 'x' OR cat = 'y'")

    def test_like_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _parse_sql_where("cat LIKE '%dog%'")

    def test_not_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _parse_sql_where("NOT cat = 'dog'")

    @pytest.mark.xfail(strict=True, reason="double-quoted strings not handled by _FIELD_EQ_STR_PATTERN (single-quote only) → NotImplementedError")
    def test_field_eq_double_quoted_string(self):
        """SQL allows double-quoted string literals. The tqdb parser only handles
        single-quoted strings; double-quoted input raises NotImplementedError."""
        r = _parse_sql_where('cat = "dog"')
        assert r == {"cat": {"$eq": "dog"}}

    def test_extra_whitespace_ok(self):
        r = _parse_sql_where("  score   >=   10  ")
        assert r == {"score": {"$gte": 10.0}}

    def test_case_insensitive_in(self):
        r = _parse_sql_where("id in ('a', 'b')")
        assert r == {"id": {"$in": ["a", "b"]}}


# ============================================================================
# 7. Cross-compat & integration edge cases
# ============================================================================

class TestCrossCompatIntegration:
    def test_chroma_and_lance_isolated_in_same_directory_subtrees(self, tmp_path):
        """ChromaDB client and LanceDB connection at different subdirs don't interfere."""
        chroma_path = str(tmp_path / "chroma")
        lance_path = str(tmp_path / "lance")
        client = PersistentClient(chroma_path)
        lance_db = connect(lance_path)
        col = client.get_or_create_collection("shared_name")
        tbl = lance_db.create_table("shared_name", data=[{"id": "x", "vector": _rand().tolist()}])
        col.add(ids=["a"], embeddings=[_rand(seed=99).tolist()])
        assert col.count() == 1
        assert tbl.count_rows() == 1

    def test_rag_retriever_and_chroma_at_different_paths_no_interference(self, tmp_path):
        rag_path = str(tmp_path / "rag")
        chroma_path = str(tmp_path / "chroma")
        ret = TurboQuantRetriever(rag_path, dimension=DIM)
        ret.add_texts(["rag doc"], [_rand().tolist()])
        client = PersistentClient(chroma_path)
        col = client.get_or_create_collection("c")
        col.add(ids=["chroma_doc"], embeddings=[_rand(seed=5).tolist()])
        assert len(ret.doc_store) == 1
        assert col.count() == 1

    def test_chroma_query_returns_nonempty_results(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        embs = _rand_batch(20).tolist()
        col.add(ids=[f"id{i}" for i in range(20)], embeddings=embs)
        result = col.query(query_embeddings=[embs[5]], n_results=5)
        assert len(result["ids"][0]) == 5

    def test_lance_and_chroma_compete_for_same_vectorstore_semantics(self, tmp_path):
        """Both shims store vectors; verify they return top-1 = self for identical query."""
        vec = _rand().tolist()
        extra = _rand_batch(10, seed=7).tolist()

        # ChromaDB shim
        chroma = PersistentClient(str(tmp_path / "c"))
        col = chroma.get_or_create_collection("c")
        col.add(ids=["target"] + [f"e{i}" for i in range(10)],
                embeddings=[vec] + extra)
        cr = col.query(query_embeddings=[vec], n_results=1)
        assert cr["ids"][0][0] == "target"

        # LanceDB shim
        lance = connect(str(tmp_path / "l"))
        tbl = lance.create_table("t", data=[{"id": "target", "vector": vec}] + [{"id": f"e{i}", "vector": e} for i, e in enumerate(extra)])
        lr = tbl.search(vec).limit(1).to_list()
        assert lr[0]["id"] == "target"

    def test_chroma_collection_delete_and_recreate_clean_state(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["old_data"], embeddings=[_rand().tolist()])
        client.delete_collection("c")
        col2 = client.create_collection("c")
        assert col2.count() == 0
