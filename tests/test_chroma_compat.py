"""Tests for the ChromaDB compatibility shim (tqdb.chroma_compat)."""

import numpy as np
import pytest

from tqdb.chroma_compat import PersistentClient


DIM = 16


def rand_vecs(n: int, d: int = DIM) -> list[list[float]]:
    rng = np.random.default_rng(42)
    return rng.random((n, d)).tolist()


# ---------------------------------------------------------------------------
# Client / collection management
# ---------------------------------------------------------------------------

class TestClientCollectionManagement:
    def test_persistent_client_creates_dir(self, tmp_path):
        client = PersistentClient(path=str(tmp_path / "db"))
        assert (tmp_path / "db").exists()

    def test_get_or_create_creates_collection(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("my_col")
        assert col.name == "my_col"

    def test_get_or_create_idempotent(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col1 = client.get_or_create_collection("col")
        col2 = client.get_or_create_collection("col")
        assert col1.name == col2.name

    def test_create_collection_duplicate_raises(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        client.create_collection("col")
        with pytest.raises(ValueError, match="already exists"):
            client.create_collection("col")

    def test_get_collection_missing_raises(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        with pytest.raises(ValueError, match="not found"):
            client.get_collection("ghost")

    def test_delete_collection(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        client.create_collection("col")
        client.delete_collection("col")
        assert "col" not in client.list_collections()

    def test_list_collections(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        client.create_collection("a")
        client.create_collection("b")
        assert sorted(client.list_collections()) == ["a", "b"]

    def test_count_collections(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        client.create_collection("x")
        client.create_collection("y")
        assert client.count_collections() == 2

    def test_reset_removes_all(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        client.create_collection("a")
        client.create_collection("b")
        client.reset()
        assert client.list_collections() == []


# ---------------------------------------------------------------------------
# Metric parsing from hnsw:space metadata
# ---------------------------------------------------------------------------

class TestMetricFromMetadata:
    def test_cosine_metric(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c", metadata={"hnsw:space": "cosine"})
        assert col._metric == "cosine"

    def test_default_metric_is_ip(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        assert col._metric == "ip"

    def test_l2_metric(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c", metadata={"hnsw:space": "l2"})
        assert col._metric == "l2"


# ---------------------------------------------------------------------------
# add / count / peek
# ---------------------------------------------------------------------------

class TestAddCountPeek:
    def test_add_and_count(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(ids=["a", "b"], embeddings=rand_vecs(2))
        assert col.count() == 2

    def test_count_empty(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        assert col.count() == 0

    def test_add_with_metadata_and_documents(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["x"],
            embeddings=rand_vecs(1),
            metadatas=[{"src": "test"}],
            documents=["hello world"],
        )
        result = col.get(ids=["x"])
        assert result["ids"] == ["x"]
        assert result["metadatas"][0]["src"] == "test"
        assert result["documents"][0] == "hello world"

    def test_peek_returns_up_to_limit(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(ids=[str(i) for i in range(5)], embeddings=rand_vecs(5))
        p = col.peek(3)
        assert len(p["ids"]) == 3

    def test_no_embedding_without_function_raises(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        with pytest.raises(ValueError, match="No embedding_function"):
            col.add(ids=["a"], documents=["some text"])


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_by_ids(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(ids=["a", "b", "c"], embeddings=rand_vecs(3))
        res = col.get(ids=["a", "c"])
        assert sorted(res["ids"]) == ["a", "c"]

    def test_get_with_where_filter(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["x", "y"],
            embeddings=rand_vecs(2),
            metadatas=[{"tag": "foo"}, {"tag": "bar"}],
        )
        res = col.get(where={"tag": {"$eq": "foo"}})
        assert res["ids"] == ["x"]

    def test_get_with_limit_and_offset(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(ids=["a", "b", "c", "d"], embeddings=rand_vecs(4))
        res = col.get(limit=2, offset=1)
        assert len(res["ids"]) == 2

    def test_get_include_param(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(ids=["a"], embeddings=rand_vecs(1), metadatas=[{"k": "v"}])
        res = col.get(ids=["a"], include=["metadatas"])
        assert res["metadatas"] is not None
        assert res["documents"] is None


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_returns_nested_lists(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        vecs = rand_vecs(10)
        col.add(ids=[str(i) for i in range(10)], embeddings=vecs)
        res = col.query(query_embeddings=[vecs[0]], n_results=3)
        assert len(res["ids"]) == 1       # one query → one result list
        assert len(res["ids"][0]) == 3    # 3 results

    def test_query_distances_present(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        vecs = rand_vecs(5)
        col.add(ids=[str(i) for i in range(5)], embeddings=vecs)
        res = col.query(query_embeddings=[vecs[0]], n_results=2)
        assert res["distances"] is not None
        assert len(res["distances"][0]) == 2

    def test_query_with_where_filter(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["a", "b"],
            embeddings=rand_vecs(2),
            metadatas=[{"group": "1"}, {"group": "2"}],
        )
        res = col.query(
            query_embeddings=[rand_vecs(1)[0]],
            n_results=5,
            where={"group": {"$eq": "1"}},
        )
        assert all(r == "a" for r in res["ids"][0])

    def test_where_document_raises(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(ids=["a"], embeddings=rand_vecs(1))
        with pytest.raises(NotImplementedError):
            col.query(query_embeddings=[rand_vecs(1)[0]], where_document={"$contains": "hi"})


# ---------------------------------------------------------------------------
# delete / upsert / update
# ---------------------------------------------------------------------------

class TestMutations:
    def test_delete_by_ids(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(ids=["a", "b", "c"], embeddings=rand_vecs(3))
        col.delete(ids=["a"])
        assert col.count() == 2

    def test_delete_by_where(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["a", "b"],
            embeddings=rand_vecs(2),
            metadatas=[{"tag": "del"}, {"tag": "keep"}],
        )
        col.delete(where={"tag": {"$eq": "del"}})
        res = col.get()
        assert "a" not in res["ids"]
        assert "b" in res["ids"]

    def test_upsert_inserts_new(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.upsert(ids=["new"], embeddings=rand_vecs(1))
        assert col.count() == 1

    def test_upsert_updates_existing(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(ids=["a"], embeddings=rand_vecs(1))
        col.upsert(ids=["a"], embeddings=rand_vecs(1), metadatas=[{"updated": True}])
        res = col.get(ids=["a"])
        assert res["metadatas"][0].get("updated") is True

    def test_embedding_function_used_for_add(self, tmp_path):
        def embed_fn(texts):
            return np.ones((len(texts), DIM), dtype=np.float32).tolist()

        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col", embedding_function=embed_fn)
        col.add(ids=["a"], documents=["hello"])
        assert col.count() == 1


# ---------------------------------------------------------------------------
# dtype — vectors must be float32
# ---------------------------------------------------------------------------

class TestDtypes:
    def test_add_uses_float32(self, tmp_path):
        """The numpy array built inside add() must have dtype float32."""
        import tqdb.chroma_compat as mod
        from unittest.mock import patch

        captured = {}
        orig_asarray = mod.np.asarray

        def spy(arr, dtype=None, **kw):
            result = orig_asarray(arr, dtype=dtype, **kw)
            if dtype is not None:
                captured["dtype"] = result.dtype
            return result

        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        with patch.object(mod.np, "asarray", side_effect=spy):
            col.add(ids=["a"], embeddings=rand_vecs(1))

        assert captured.get("dtype") == np.float32, (
            f"Expected float32, got {captured.get('dtype')}"
        )


# ---------------------------------------------------------------------------
# update — batching and metadata-only paths
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_batch_embeddings(self, tmp_path):
        """Updating multiple items with new embeddings works in one call."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["a", "b", "c"],
            embeddings=rand_vecs(3),
            metadatas=[{"v": 0}, {"v": 0}, {"v": 0}],
        )
        new_vecs = rand_vecs(3)
        col.update(
            ids=["a", "b", "c"],
            embeddings=new_vecs,
            metadatas=[{"v": 1}, {"v": 2}, {"v": 3}],
        )
        res = col.get(ids=["a", "b", "c"])
        meta_by_id = {
            id_: meta
            for id_, meta in zip(res["ids"], res["metadatas"])
        }
        assert meta_by_id["a"]["v"] == 1
        assert meta_by_id["b"]["v"] == 2
        assert meta_by_id["c"]["v"] == 3

    def test_update_metadata_only(self, tmp_path):
        """Updating metadata without embeddings calls update_metadata, not insert_batch."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(ids=["a"], embeddings=rand_vecs(1), documents=["original"])
        col.update(ids=["a"], metadatas=[{"updated": True}])
        res = col.get(ids=["a"])
        assert res["metadatas"][0].get("updated") is True
        # document should remain unchanged (update_metadata preserves it if doc=None)
        assert res["documents"][0] == "original"


# ---------------------------------------------------------------------------
# get / delete — list_ids fast path
# ---------------------------------------------------------------------------

class TestGetOptimized:
    def test_get_where_returns_only_matching(self, tmp_path):
        """get(where=...) without ids should return only records matching the filter."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["x", "y", "z"],
            embeddings=rand_vecs(3),
            metadatas=[{"tag": "a"}, {"tag": "b"}, {"tag": "a"}],
        )
        res = col.get(where={"tag": {"$eq": "a"}})
        assert sorted(res["ids"]) == ["x", "z"]
        assert "y" not in res["ids"]

    def test_get_where_with_ids_intersection(self, tmp_path):
        """get(ids=..., where=...) should return the intersection."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["a", "b", "c"],
            embeddings=rand_vecs(3),
            metadatas=[{"tag": "x"}, {"tag": "x"}, {"tag": "y"}],
        )
        res = col.get(ids=["a", "b"], where={"tag": {"$eq": "x"}})
        assert sorted(res["ids"]) == ["a", "b"]


class TestDeleteOptimized:
    def test_delete_where_only_removes_matching(self, tmp_path):
        """delete(where=...) removes exactly the records matching the filter."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["a", "b", "c"],
            embeddings=rand_vecs(3),
            metadatas=[{"tag": "del"}, {"tag": "del"}, {"tag": "keep"}],
        )
        col.delete(where={"tag": {"$eq": "del"}})
        assert col.count() == 1
        res = col.get()
        assert res["ids"] == ["c"]


# ---------------------------------------------------------------------------
# $exists and $contains filter operators
# ---------------------------------------------------------------------------

class TestFilters:
    def test_filter_exists_true(self, tmp_path):
        """`$exists: True` keeps only records where the field is present."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["has", "missing"],
            embeddings=rand_vecs(2),
            metadatas=[{"rare": "yes"}, {}],
        )
        res = col.get(where={"rare": {"$exists": True}})
        assert res["ids"] == ["has"]

    def test_filter_exists_false(self, tmp_path):
        """`$exists: False` keeps only records where the field is absent."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["has", "missing"],
            embeddings=rand_vecs(2),
            metadatas=[{"rare": "yes"}, {}],
        )
        res = col.get(where={"rare": {"$exists": False}})
        assert res["ids"] == ["missing"]

    def test_filter_contains(self, tmp_path):
        """`$contains` filters by substring match."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        col.add(
            ids=["match", "no_match"],
            embeddings=rand_vecs(2),
            metadatas=[{"text": "hello world"}, {"text": "goodbye"}],
        )
        res = col.get(where={"text": {"$contains": "world"}})
        assert res["ids"] == ["match"]


# ---------------------------------------------------------------------------
# Auto-embed: no documents + no embed_fn must raise, not embed empty strings
# ---------------------------------------------------------------------------

class TestAutoEmbed:
    def test_add_no_docs_no_embeddings_raises(self, tmp_path):
        """add() with no embeddings and no documents must raise ValueError."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        with pytest.raises(ValueError):
            col.add(ids=["a"])

    def test_add_docs_with_embed_fn_receives_real_docs(self, tmp_path):
        """The embedding_function receives actual document strings, not empty strings."""
        received = {}

        def embed_fn(texts):
            received["texts"] = list(texts)
            return np.ones((len(texts), DIM), dtype=np.float32).tolist()

        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col", embedding_function=embed_fn)
        col.add(ids=["a"], documents=["real document"])
        assert received["texts"] == ["real document"]

    def test_upsert_no_docs_no_embeddings_raises(self, tmp_path):
        """upsert() with no embeddings and no documents must raise ValueError."""
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("col")
        with pytest.raises(ValueError):
            col.upsert(ids=["a"])
