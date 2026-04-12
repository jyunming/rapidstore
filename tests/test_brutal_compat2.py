"""Round 10 — Brutal Compat Layer Gap Tests (Part 2)

Systematically probes every critical gap between the TurboQuantDB compat shims
(ChromaDB, LanceDB, LangChain/RAG) and the real vendor APIs.

Strategy:
  - xfail(strict=True)  → known gaps where real API differs from compat shim
                           (test defines *correct* behaviour; it fails because
                           the shim is incomplete → documents the defect)
  - Regular assertions  → edge-case and correctness tests that must pass TODAY

Gap inventory tags: COMPAT-GAP-Cn (Chroma), COMPAT-GAP-Ln (LanceDB), COMPAT-GAP-Rn (RAG)
"""

from __future__ import annotations

import tempfile
import threading
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chroma_client(tmp_path):
    from tqdb.chroma_compat import PersistentClient
    return PersistentClient(str(tmp_path))


def _make_chroma_col(tmp_path, name="t", dim=4, metric="ip"):
    c = _chroma_client(tmp_path)
    col = c.get_or_create_collection(name, metadata={"hnsw:space": metric})
    return c, col


def _vec(dim=4, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-9)
    return v.tolist()


def _vecs(n, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    return (m / norms).tolist()


def _lance_db(tmp_path):
    from tqdb.lancedb_compat import connect
    return connect(str(tmp_path))


def _lance_table(tmp_path, n=6, dim=4, name="t"):
    db = _lance_db(tmp_path)
    rows = [{"id": str(i), "vector": _vec(dim, seed=i)} for i in range(n)]
    tbl = db.create_table(name, data=rows)
    return db, tbl


def _rag(tmp_path, dim=4, metric="ip"):
    from tqdb.rag import TurboQuantRetriever
    return TurboQuantRetriever(str(tmp_path), dimension=dim, metric=metric)


# ===========================================================================
# A  ChromaDB — API attribute / method gaps
# ===========================================================================

class TestChromaAPIGaps:
    """Real chromadb exposes attributes/methods not yet implemented in compat."""

    def test_collection_id_attribute(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(ids=["a"], embeddings=[_vec()])
        # real ChromaDB: col.id is a UUID string
        assert col.id is not None

    def test_collection_metadata_attribute(self, tmp_path):
        c = _chroma_client(tmp_path)
        col = c.get_or_create_collection("t", metadata={"hnsw:space": "ip", "custom": "val"})
        # real ChromaDB: col.metadata reflects creation metadata
        assert col.metadata is not None
        assert col.metadata.get("custom") == "val"

    def test_client_heartbeat(self, tmp_path):
        c = _chroma_client(tmp_path)
        # real ChromaDB PersistentClient.heartbeat() returns nanoseconds
        result = c.heartbeat()
        assert isinstance(result, int) and result > 0

    def test_get_include_embeddings_returns_actual_vectors(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        emb = _vec()
        col.add(ids=["a"], embeddings=[emb])
        r = col.get(include=["embeddings"])
        # real ChromaDB: r["embeddings"] is a list of lists
        assert r["embeddings"] is not None
        assert len(r["embeddings"]) == 1
        assert abs(r["embeddings"][0][0] - emb[0]) < 1e-4

    def test_query_include_embeddings_returns_actual_vectors(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        emb = _vec()
        col.add(ids=["a"], embeddings=[emb])
        r = col.query(query_embeddings=[emb], n_results=1, include=["embeddings"])
        assert r["embeddings"] is not None
        assert len(r["embeddings"][0]) == 1
        assert abs(r["embeddings"][0][0][0] - emb[0]) < 1e-4

    def test_list_collections_returns_sorted_strings(self, tmp_path):
        c = _chroma_client(tmp_path)
        c.get_or_create_collection("alpha")
        c.get_or_create_collection("beta")
        cols = c.list_collections()
        # chromadb ≥ 1.5: list_collections() returns sorted list of name strings
        assert cols == ["alpha", "beta"]
        assert all(isinstance(n, str) for n in cols)


# ===========================================================================
# B  ChromaDB — edge-case correctness (must PASS today)
# ===========================================================================

class TestChromaEdgeCases:

    def test_get_empty_ids_list_returns_empty(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(ids=["x"], embeddings=[_vec()])
        r = col.get(ids=[])
        assert r["ids"] == []

    def test_get_where_empty_dict_returns_all(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(ids=["a", "b"], embeddings=[_vec(seed=0), _vec(seed=1)])
        r = col.get(where={})
        assert set(r["ids"]) == {"a", "b"}

    def test_query_n_results_larger_than_corpus_capped(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(ids=["a", "b"], embeddings=[_vec(seed=0), _vec(seed=1)])
        r = col.query(query_embeddings=[_vec()], n_results=100)
        assert len(r["ids"][0]) == 2

    def test_query_n_results_zero_raises(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(ids=["a"], embeddings=[_vec()])
        with pytest.raises((ValueError, Exception)):
            col.query(query_embeddings=[_vec()], n_results=0)

    def test_add_empty_ids_and_embeddings_raises(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        with pytest.raises((ValueError, Exception)):
            col.add(ids=[], embeddings=[])

    def test_add_no_embeddings_no_ef_raises_value_error(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        with pytest.raises(ValueError):
            col.add(ids=["z"])

    def test_modify_rename_then_list_shows_new_name(self, tmp_path):
        c = _chroma_client(tmp_path)
        col = c.get_or_create_collection("old_name")
        col.modify(name="new_name")
        cols = c.list_collections()
        col_names = list(cols)
        assert "new_name" in col_names
        assert "old_name" not in col_names

    def test_multiple_collections_isolated_counts(self, tmp_path):
        c = _chroma_client(tmp_path)
        c1 = c.get_or_create_collection("c1")
        c2 = c.get_or_create_collection("c2")
        c1.add(ids=["a", "b", "c"], embeddings=_vecs(3))
        c2.add(ids=["x"], embeddings=[_vec()])
        assert c1.count() == 3
        assert c2.count() == 1

    def test_multiple_collections_query_no_cross_pollution(self, tmp_path):
        c = _chroma_client(tmp_path)
        c1 = c.get_or_create_collection("c1")
        c2 = c.get_or_create_collection("c2")
        emb_a = _vec(seed=10)
        emb_b = _vec(seed=20)
        c1.add(ids=["only_in_c1"], embeddings=[emb_a])
        c2.add(ids=["only_in_c2"], embeddings=[emb_b])
        r1 = c1.query(query_embeddings=[emb_a], n_results=1)
        r2 = c2.query(query_embeddings=[emb_b], n_results=1)
        assert r1["ids"][0] == ["only_in_c1"]
        assert r2["ids"][0] == ["only_in_c2"]

    def test_upsert_same_id_many_times_count_stays_1(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        for i in range(20):
            col.upsert(ids=["stable"], embeddings=[_vec(seed=i)])
        assert col.count() == 1

    def test_delete_all_ids_then_query_returns_empty(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(ids=["a", "b", "c"], embeddings=_vecs(3))
        col.delete(ids=["a", "b", "c"])
        r = col.query(query_embeddings=[_vec()], n_results=5)
        assert r["ids"][0] == []

    def test_where_and_filter_with_empty_op_returns_matching(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(
            ids=["hot", "cold"],
            embeddings=[_vec(seed=0), _vec(seed=1)],
            metadatas=[{"temp": "hot"}, {"temp": "cold"}],
        )
        r = col.get(where={"temp": {"$eq": "hot"}})
        assert r["ids"] == ["hot"]

    def test_where_ne_filter_excludes_matching(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(
            ids=["a", "b", "c"],
            embeddings=_vecs(3),
            metadatas=[{"k": "x"}, {"k": "y"}, {"k": "x"}],
        )
        r = col.get(where={"k": {"$ne": "x"}})
        assert r["ids"] == ["b"]

    def test_where_in_filter(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(
            ids=["a", "b", "c", "d"],
            embeddings=_vecs(4),
            metadatas=[{"n": i} for i in range(4)],
        )
        r = col.get(where={"n": {"$in": [1, 3]}})
        assert set(r["ids"]) == {"b", "d"}

    def test_add_then_update_then_get_returns_updated_metadata(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(ids=["a"], embeddings=[_vec()], metadatas=[{"v": 1}])
        col.update(ids=["a"], metadatas=[{"v": 99}])
        r = col.get(ids=["a"])
        assert r["metadatas"][0]["v"] == 99

    def test_concurrent_add_different_ids_final_count_correct(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        errors = []

        def worker(batch_start):
            try:
                n = 5
                ids = [f"id_{batch_start}_{i}" for i in range(n)]
                embs = _vecs(n, seed=batch_start)
                col.add(ids=ids, embeddings=embs)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(b * 5,)) for b in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"concurrent add errors: {errors}"
        assert col.count() == 40

    def test_get_offset_limit_paging_correct(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        ids = [f"id_{i}" for i in range(10)]
        col.add(ids=ids, embeddings=_vecs(10))
        page1 = col.get(limit=4, offset=0)
        page2 = col.get(limit=4, offset=4)
        all_ids = col.get()["ids"]
        assert len(page1["ids"]) == 4
        assert len(page2["ids"]) == 4
        # pages should be non-overlapping slices of all_ids
        assert set(page1["ids"]) | set(page2["ids"]) <= set(all_ids)
        assert set(page1["ids"]).isdisjoint(set(page2["ids"]))

    def test_peek_returns_at_most_n(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(ids=[f"x{i}" for i in range(20)], embeddings=_vecs(20))
        r = col.peek(limit=5)  # real ChromaDB uses `limit=`, not `n=`
        assert len(r["ids"]) <= 5

    def test_query_multiple_embeddings_returns_per_query_results(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(ids=["a", "b", "c"], embeddings=_vecs(3))
        r = col.query(query_embeddings=[_vec(seed=0), _vec(seed=1)], n_results=2)
        # 2 queries → 2 result lists
        assert len(r["ids"]) == 2
        assert len(r["ids"][0]) <= 2
        assert len(r["ids"][1]) <= 2

    def test_include_both_metadatas_and_documents(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(
            ids=["a"],
            embeddings=[_vec()],
            metadatas=[{"tag": "qa"}],
            documents=["hello world"],
        )
        r = col.get(include=["metadatas", "documents"])
        assert r["metadatas"][0]["tag"] == "qa"
        assert r["documents"][0] == "hello world"

    def test_get_include_only_ids_no_metadatas_no_documents(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(ids=["a"], embeddings=[_vec()], metadatas=[{"k": "v"}])
        r = col.get(include=["ids"])
        assert "a" in r["ids"]
        # When only ids requested, metadatas/documents should not be in result
        # (or should be None/empty)
        assert r.get("metadatas") is None or r.get("metadatas") == [] or r.get("metadatas") == [None]

    def test_chroma_add_large_batch_1000(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        n = 1000
        ids = [f"doc_{i}" for i in range(n)]
        embs = _vecs(n, dim=16)
        # Need to recreate with correct dim
        c = _chroma_client(tmp_path)
        col2 = c.get_or_create_collection("large16", metadata={"hnsw:space": "ip"})
        # Rebuild compat col with dim=16
        from tqdb.chroma_compat import PersistentClient as PC
        with tempfile.TemporaryDirectory() as td:
            client = PC(td)
            bigcol = client.get_or_create_collection("big")
            bigcol.add(ids=ids, embeddings=embs)
            assert bigcol.count() == n

    def test_query_where_filter_narrows_results(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(
            ids=[f"v{i}" for i in range(20)],
            embeddings=_vecs(20),
            metadatas=[{"group": "A" if i < 10 else "B"} for i in range(20)],
        )
        r = col.query(
            query_embeddings=[_vec()],
            n_results=5,
            where={"group": {"$eq": "A"}},
        )
        for doc_id in r["ids"][0]:
            idx = int(doc_id[1:])
            assert idx < 10, f"B-group doc leaked through filter: {doc_id}"

    def test_collection_survives_client_reconnect(self, tmp_path):
        from tqdb.chroma_compat import PersistentClient as PC
        p = str(tmp_path)
        c1 = PC(p)
        col1 = c1.get_or_create_collection("persist_test")
        col1.add(ids=["a"], embeddings=[_vec()], documents=["test doc"])
        del c1, col1  # close

        c2 = PC(p)
        col2 = c2.get_collection("persist_test")
        r = col2.get(ids=["a"])
        assert r["ids"] == ["a"]
        assert r["documents"][0] == "test doc"

    def test_delete_with_where_removes_correct_subset(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(
            ids=["a", "b", "c"],
            embeddings=_vecs(3),
            metadatas=[{"cat": "x"}, {"cat": "y"}, {"cat": "x"}],
        )
        col.delete(where={"cat": {"$eq": "x"}})
        remaining = col.get()
        assert set(remaining["ids"]) == {"b"}

    def test_add_sanitizes_list_metadata_value_to_string(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(ids=["a"], embeddings=[_vec()], metadatas=[{"tags": ["foo", "bar"]}])
        r = col.get(ids=["a"])
        # list should be coerced to string so it doesn't crash the DB
        meta = r["metadatas"][0]
        assert isinstance(meta.get("tags"), str)

    def test_add_sanitizes_none_metadata_value(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(ids=["a"], embeddings=[_vec()], metadatas=[{"k": None}])
        r = col.get(ids=["a"])
        meta = r["metadatas"][0]
        assert isinstance(meta.get("k"), str)  # coerced to "None"

    def test_reset_after_multiple_collections_all_gone(self, tmp_path):
        c = _chroma_client(tmp_path)
        for name in ["a", "b", "c", "d"]:
            col = c.get_or_create_collection(name)
            col.add(ids=["x"], embeddings=[_vec()])
        c.reset()
        assert c.list_collections() == []
        assert c.count_collections() == 0


# ===========================================================================
# C  LanceDB — API attribute / method gaps
# ===========================================================================

class TestLanceDBAPIGaps:
    """Real lancedb exposes attributes/methods not yet implemented in compat."""

    def test_len_table(self, tmp_path):
        _, tbl = _lance_table(tmp_path, n=5)
        assert len(tbl) == 5

    def test_schema_attribute(self, tmp_path):
        _, tbl = _lance_table(tmp_path, n=3)
        schema = tbl.schema
        assert schema is not None
        assert "vector" in schema.names

    def test_head_method(self, tmp_path):
        _, tbl = _lance_table(tmp_path, n=10)
        rows = tbl.head(3)
        assert len(rows) == 3

    def test_search_none_full_scan(self, tmp_path):
        _, tbl = _lance_table(tmp_path, n=5)
        # real LanceDB: tbl.search() or tbl.search(None) returns all rows
        rows = tbl.search(None).to_list()
        assert len(rows) == 5

    def test_update_method(self, tmp_path):
        db, tbl = _lance_table(tmp_path, n=3)
        tbl.add([{"id": "0", "vector": _vec(seed=0), "cat": "original"}], mode="overwrite")
        # real LanceDB: tbl.update(where="id = '0'", values={"cat": "updated"})
        tbl.update(where="id = '0'", values={"cat": "updated"})
        rows = tbl.to_list()
        cats = {r["id"]: r.get("cat") for r in rows}
        assert cats.get("0") == "updated"

    def test_merge_insert_method(self, tmp_path):
        _, tbl = _lance_table(tmp_path, n=3)
        new_rows = [{"id": "0", "vector": _vec(seed=99)}]  # existing id → update
        tbl.merge_insert("id").when_matched_update_all().execute(new_rows)
        rows = {r["id"]: r for r in tbl.to_list()}
        # vector for id "0" should have changed
        assert rows["0"]["vector"] != _vec(seed=0)


# ===========================================================================
# D  LanceDB — SQL / filter edge-case correctness (must PASS today)
# ===========================================================================

class TestLanceDBSQLEdgeCases:

    def test_double_quoted_field_in_where_raises_not_implemented(self, tmp_path):
        """SQL with double-quoted identifiers is not in the regex parser."""
        db = _lance_db(tmp_path)
        rows = [{"id": "q0", "cat": "a", "vector": _vec()}]
        tbl = db.create_table("t", data=rows)
        where = chr(34) + "cat" + chr(34) + " = " + chr(39) + "a" + chr(39)
        with pytest.raises(NotImplementedError):
            tbl.search(_vec()).where(where).to_list()

    def test_is_not_null_where_raises_not_implemented(self, tmp_path):
        _, tbl = _lance_table(tmp_path)
        with pytest.raises(NotImplementedError):
            tbl.search(_vec()).where("id IS NOT NULL").to_list()

    def test_empty_in_list_raises_not_implemented(self, tmp_path):
        """IN () with empty list is not valid SQL and not supported."""
        _, tbl = _lance_table(tmp_path)
        with pytest.raises((NotImplementedError, ValueError, Exception)):
            tbl.search(_vec()).where("id IN ()").to_list()

    def test_and_compound_where_raises_not_implemented(self, tmp_path):
        db = _lance_db(tmp_path)
        rows = [{"id": "q0", "cat": "a", "vector": _vec()}]
        tbl = db.create_table("t", data=rows)
        with pytest.raises(NotImplementedError):
            tbl.search(_vec()).where("id = 'q0' AND cat = 'a'").to_list()

    def test_or_compound_where_raises_not_implemented(self, tmp_path):
        _, tbl = _lance_table(tmp_path)
        with pytest.raises(NotImplementedError):
            tbl.search(_vec()).where("id = '0' OR id = '1'").to_list()

    def test_like_where_raises_not_implemented(self, tmp_path):
        _, tbl = _lance_table(tmp_path)
        with pytest.raises(NotImplementedError):
            tbl.search(_vec()).where("id LIKE '0%'").to_list()

    def test_select_nonexistent_column_silently_returns_rows(self, tmp_path):
        """Selecting a nonexistent column should NOT crash (compat ignores bad columns)."""
        _, tbl = _lance_table(tmp_path, n=3)
        rows = tbl.search(_vec()).select(["nonexistent_col"]).to_list()
        # compat returns rows anyway (ignores column filter silently)
        assert len(rows) > 0

    def test_where_field_gt_float(self, tmp_path):
        db = _lance_db(tmp_path)
        rows = [{"id": str(i), "score": float(i) * 0.1, "vector": _vec(seed=i)} for i in range(10)]
        tbl = db.create_table("t", data=rows)
        results = tbl.search(_vec()).where("score > 0.5").to_list()
        for r in results:
            assert r["score"] > 0.5, f"filter failed: score={r['score']}"

    def test_where_field_lte_integer(self, tmp_path):
        db = _lance_db(tmp_path)
        rows = [{"id": str(i), "rank": i, "vector": _vec(seed=i)} for i in range(10)]
        tbl = db.create_table("t", data=rows)
        results = tbl.search(_vec()).where("rank <= 3").to_list()
        for r in results:
            assert r["rank"] <= 3

    def test_delete_by_id_in_then_count_decremented(self, tmp_path):
        _, tbl = _lance_table(tmp_path, n=6)
        before = tbl.count_rows()
        tbl.delete("id IN ('0', '1', '2')")
        after = tbl.count_rows()
        assert after == before - 3

    def test_create_table_overwrite_replaces_all_rows(self, tmp_path):
        db = _lance_db(tmp_path)
        rows_a = [{"id": "a", "vector": _vec(seed=0)}]
        rows_b = [{"id": "b", "vector": _vec(seed=1)}, {"id": "c", "vector": _vec(seed=2)}]
        tbl = db.create_table("t", data=rows_a)
        assert tbl.count_rows() == 1
        tbl2 = db.create_table("t", data=rows_b, mode="overwrite")
        assert tbl2.count_rows() == 2

    def test_add_mode_append_then_search_sees_all(self, tmp_path):
        _, tbl = _lance_table(tmp_path, n=3)
        new_rows = [{"id": "new0", "vector": _vec(seed=99)}]
        tbl.add(new_rows, mode="append")
        r = tbl.search(_vec(seed=99)).limit(10).to_list()
        ids = [x["id"] for x in r]
        assert "new0" in ids

    def test_search_limit_respected(self, tmp_path):
        _, tbl = _lance_table(tmp_path, n=20)
        r = tbl.search(_vec()).limit(3).to_list()
        assert len(r) == 3

    def test_search_self_is_top1(self, tmp_path):
        db = _lance_db(tmp_path)
        v = _vec(seed=42)
        rows = [{"id": "target", "vector": v}] + [
            {"id": f"noise_{i}", "vector": _vec(seed=i)} for i in range(30)
        ]
        tbl = db.create_table("t", data=rows)
        results = tbl.search(v).limit(1).to_list()
        assert results[0]["id"] == "target"

    def test_to_pandas_has_expected_columns(self, tmp_path):
        db = _lance_db(tmp_path)
        rows = [{"id": "a", "label": "qa", "vector": _vec()}]
        tbl = db.create_table("t", data=rows)
        df = tbl.to_pandas()
        assert "id" in df.columns
        assert "vector" in df.columns  # gap: returns only id + metadata blob, not raw vector

    def test_to_arrow_has_expected_rows(self, tmp_path):
        _, tbl = _lance_table(tmp_path, n=7)
        at = tbl.to_arrow()
        assert at.num_rows == 7

    def test_persist_across_reconnect(self, tmp_path):
        from tqdb.lancedb_compat import connect
        p = str(tmp_path)
        db1 = connect(p)
        rows = [{"id": "x", "vector": _vec(seed=5)}]
        db1.create_table("persist_t", data=rows)
        del db1

        db2 = connect(p)
        tbl = db2.open_table("persist_t")
        assert tbl.count_rows() == 1

    def test_table_names_grows_with_tables(self, tmp_path):
        db = _lance_db(tmp_path)
        assert db.table_names() == []
        db.create_table("a", data=[{"id": "1", "vector": _vec()}])
        db.create_table("b", data=[{"id": "2", "vector": _vec()}])
        names = db.table_names()
        assert "a" in names and "b" in names

    def test_drop_table_removes_from_table_names(self, tmp_path):
        db = _lance_db(tmp_path)
        db.create_table("gone", data=[{"id": "1", "vector": _vec()}])
        db.drop_table("gone")
        assert "gone" not in db.table_names()

    def test_count_rows_with_field_in_filter(self, tmp_path):
        db = _lance_db(tmp_path)
        rows = [{"id": str(i), "cat": "x" if i % 2 == 0 else "y", "vector": _vec(seed=i)} for i in range(10)]
        tbl = db.create_table("t", data=rows)
        # id IN (0,2,4)
        count = tbl.count_rows(filter="id IN ('0', '2', '4')")
        assert count == 3

    def test_add_pyarrow_table(self, tmp_path):
        import pyarrow as pa
        db = _lance_db(tmp_path)
        schema = pa.schema([("id", pa.string()), ("vector", pa.list_(pa.float32(), 4))])
        arr_id = pa.array(["r0", "r1", "r2"])
        arr_vec = pa.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]], type=pa.list_(pa.float32(), 4))
        at = pa.table({"id": arr_id, "vector": arr_vec})
        tbl = db.create_table("t", data=[{"id": "init", "vector": [0.0]*4}])
        tbl.add(at, mode="append")
        assert tbl.count_rows() == 4


# ===========================================================================
# E  LangChain RAG — API gap tests
# ===========================================================================

class TestRAGAPIGaps:
    """Real LangChain VectorStore interface methods missing from TurboQuantRetriever."""

    def test_get_relevant_documents_missing(self, tmp_path):
        r = _rag(tmp_path)
        r.add_texts(["hello", "world"], embeddings=_vecs(2))
        docs = r.get_relevant_documents("hello")
        assert isinstance(docs, list)

    def test_invoke_missing(self, tmp_path):
        r = _rag(tmp_path)
        r.add_texts(["hello", "world"], embeddings=_vecs(2))
        docs = r.invoke("hello")
        assert isinstance(docs, list)

    def test_similarity_search_with_score_missing(self, tmp_path):
        r = _rag(tmp_path)
        r.add_texts(["hello"], embeddings=[_vec()])
        results = r.similarity_search_with_score(_vec(), k=1)
        assert isinstance(results, list)
        doc, score = results[0]
        assert isinstance(score, float)

    def test_similarity_search_filter_kwarg(self, tmp_path):
        r = _rag(tmp_path)
        r.add_texts(["hello", "world"], embeddings=_vecs(2),
                    metadatas=[{"tag": "a"}, {"tag": "b"}])
        results = r.similarity_search(_vec(), k=1, filter={"tag": {"$eq": "a"}})
        assert len(results) == 1
        assert results[0].metadata["tag"] == "a"

    def test_from_texts_classmethod_missing(self, tmp_path):
        from tqdb.rag import TurboQuantRetriever
        r = TurboQuantRetriever.from_texts(
            texts=["hello", "world"],
            embeddings=_vecs(2),
            db_path=str(tmp_path),
            dimension=4,
        )
        assert r is not None
        results = r.similarity_search(_vec(), k=1)
        assert len(results) == 1

    def test_delete_method_missing(self, tmp_path):
        r = _rag(tmp_path)
        r.add_texts(["hello"], embeddings=[_vec()])
        # add_texts returns ids via doc_store keys
        doc_id = list(r.doc_store.keys())[0]
        r.delete([doc_id])
        assert doc_id not in r.doc_store

    def test_as_retriever_missing(self, tmp_path):
        r = _rag(tmp_path)
        chain_retriever = r.as_retriever()
        assert chain_retriever is not None

    def test_add_documents_missing(self, tmp_path):
        r = _rag(tmp_path)
        # Simulate Document-like objects (duck-typing)
        class FakeDoc:
            def __init__(self, text, meta):
                self.page_content = text
                self.metadata = meta
        docs = [FakeDoc("hello", {"src": "x"}), FakeDoc("world", {"src": "y"})]
        # Need embeddings since we have no EF
        r.add_documents(docs, embeddings=_vecs(2))
        assert len(r.doc_store) == 2

    def test_similarity_search_returns_langchain_documents(self, tmp_path):
        r = _rag(tmp_path)
        r.add_texts(["hello"], embeddings=[_vec()])
        results = r.similarity_search(_vec(), k=1)
        doc = results[0]
        # real LangChain Document has .page_content and .metadata attributes
        assert hasattr(doc, "page_content")
        assert hasattr(doc, "metadata")


# ===========================================================================
# F  LangChain RAG — correctness (must PASS today)
# ===========================================================================

class TestRAGCorrectness:

    def test_add_texts_then_similarity_search_returns_non_empty(self, tmp_path):
        r = _rag(tmp_path)
        r.add_texts(["alpha", "beta", "gamma"], embeddings=_vecs(3))
        results = r.similarity_search(_vec(seed=0), k=2)
        assert len(results) > 0

    def test_similarity_search_result_has_required_keys(self, tmp_path):
        r = _rag(tmp_path)
        r.add_texts(["hello world"], embeddings=[_vec()])
        results = r.similarity_search(_vec(), k=1)
        assert len(results) == 1
        # similarity_search() returns Document objects (COMPAT-GAP-R9 fix)
        assert hasattr(results[0], "page_content")
        assert hasattr(results[0], "metadata")

    def test_similarity_search_scores_are_float(self, tmp_path):
        r = _rag(tmp_path)
        r.add_texts(["x", "y", "z"], embeddings=_vecs(3))
        # scores are accessible via similarity_search_with_score()
        results = r.similarity_search_with_score(_vec(), k=3)
        for doc, score in results:
            assert isinstance(score, float)

    def test_doc_store_keys_unique_across_multiple_add_rounds(self, tmp_path):
        r = _rag(tmp_path)
        r.add_texts(["a", "b"], embeddings=_vecs(2, seed=0))
        r.add_texts(["c", "d"], embeddings=_vecs(2, seed=2))
        assert len(r.doc_store) == 4
        assert len(set(r.doc_store.keys())) == 4

    def test_documents_accessible_after_reopen(self, tmp_path):
        """Document text is stored in tqdb (persistent), so results survive reopen."""
        p = str(tmp_path)
        r1 = _rag(tmp_path)
        r1.add_texts(["hello"], embeddings=[_vec()])
        del r1  # simulate process restart
        r2 = _rag(tmp_path)
        results = r2.similarity_search(_vec(), k=5)
        # tqdb persists the document field, so text is still accessible after reopen
        assert len(results) >= 1
        assert hasattr(results[0], "page_content")

    def test_similarity_search_k_greater_than_stored_capped(self, tmp_path):
        r = _rag(tmp_path)
        r.add_texts(["only one"], embeddings=[_vec()])
        results = r.similarity_search(_vec(), k=100)
        assert len(results) <= 1

    def test_similarity_search_empty_db_returns_empty(self, tmp_path):
        r = _rag(tmp_path)
        results = r.similarity_search(_vec(), k=5)
        assert results == []

    def test_metadatas_preserved_in_results(self, tmp_path):
        r = _rag(tmp_path)
        r.add_texts(
            ["doc with meta"],
            embeddings=[_vec()],
            metadatas=[{"author": "tqdb", "version": 42}],
        )
        results = r.similarity_search(_vec(), k=1)
        assert len(results) == 1
        assert results[0].metadata.get("author") == "tqdb"
        assert results[0].metadata.get("version") == 42

    def test_k_equals_1_returns_exactly_one(self, tmp_path):
        r = _rag(tmp_path)
        r.add_texts(["a", "b", "c"], embeddings=_vecs(3))
        results = r.similarity_search(_vec(), k=1)
        assert len(results) == 1

    def test_add_texts_dimension_mismatch_raises(self, tmp_path):
        r = _rag(tmp_path, dim=4)
        with pytest.raises((ValueError, RuntimeError, Exception)):
            r.add_texts(["text"], embeddings=[[0.1, 0.2]])  # dim=2 vs expected 4

    def test_add_texts_count_mismatch_raises(self, tmp_path):
        r = _rag(tmp_path)
        with pytest.raises((ValueError, Exception)):
            r.add_texts(["a", "b", "c"], embeddings=_vecs(2))  # 3 texts, 2 embeddings

    def test_l2_metric_retriever_works(self, tmp_path):
        r = _rag(tmp_path, metric="l2")
        r.add_texts(["hello", "world"], embeddings=_vecs(2))
        results = r.similarity_search(_vec(), k=2)
        assert len(results) > 0

    def test_numpy_embeddings_accepted(self, tmp_path):
        r = _rag(tmp_path)
        embs = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
        r.add_texts(["a", "b", "c"], embeddings=embs)
        results = r.similarity_search(embs[0].tolist(), k=1)
        assert len(results) == 1

    def test_large_batch_500_texts_search_returns_results(self, tmp_path):
        r = _rag(tmp_path, dim=16)
        n = 500
        embs = _vecs(n, dim=16, seed=7)
        texts = [f"document_{i}" for i in range(n)]
        r.add_texts(texts, embeddings=embs)
        results = r.similarity_search(embs[0], k=10)
        assert len(results) > 0
        assert results[0].page_content == "document_0"

    def test_concurrent_add_texts_no_crash(self, tmp_path):
        r = _rag(tmp_path)
        errors = []

        def worker(seed):
            try:
                r.add_texts([f"doc_{seed}"], embeddings=[_vec(seed=seed)])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"concurrent add_texts errors: {errors}"

    def test_all_zero_embeddings_does_not_crash(self, tmp_path):
        r = _rag(tmp_path)
        r.add_texts(["zero"], embeddings=[[0.0, 0.0, 0.0, 0.0]])
        # search may return empty or a result — just must not crash
        results = r.similarity_search([0.0, 0.0, 0.0, 0.0], k=1)
        assert isinstance(results, list)

    def test_single_char_texts(self, tmp_path):
        r = _rag(tmp_path)
        r.add_texts(["a"], embeddings=[_vec()])
        r.add_texts(["b"], embeddings=[_vec(seed=1)])
        results = r.similarity_search(_vec(), k=2)
        assert len(results) >= 1

    def test_unicode_texts_stored_correctly(self, tmp_path):
        r = _rag(tmp_path)
        texts = ["日本語テキスト", "中文文本", "한국어 텍스트", "العربية"]
        embs = _vecs(len(texts))
        r.add_texts(texts, embeddings=embs)
        results = r.similarity_search(_vec(seed=0), k=4)
        found_texts = {res.page_content for res in results}
        # at least some unicode texts should be retrievable
        assert len(found_texts) > 0
        assert found_texts <= set(texts)


# ===========================================================================
# G  Cross-compat stress / interaction tests
# ===========================================================================

class TestCrossCompatStress:

    def test_chroma_and_lance_same_base_dir_no_conflict(self, tmp_path):
        """Both compat layers can coexist in sibling directories under the same parent."""
        import os
        chroma_path = str(tmp_path / "chroma_data")
        lance_path = str(tmp_path / "lance_data")
        os.makedirs(chroma_path, exist_ok=True)

        from tqdb.chroma_compat import PersistentClient as PC
        from tqdb.lancedb_compat import connect

        c = PC(chroma_path)
        col = c.get_or_create_collection("shared_test")
        col.add(ids=["c0"], embeddings=[_vec()])

        db = connect(lance_path)
        tbl = db.create_table("shared_test", data=[{"id": "l0", "vector": _vec(seed=1)}])

        assert col.count() == 1
        assert tbl.count_rows() == 1

    def test_rag_and_chroma_different_paths_no_interference(self, tmp_path):
        import os
        rag_path = str(tmp_path / "rag")
        chroma_path = str(tmp_path / "chroma")
        os.makedirs(rag_path, exist_ok=True)
        os.makedirs(chroma_path, exist_ok=True)

        from tqdb.chroma_compat import PersistentClient as PC

        r = _rag(tmp_path / "rag")
        r.add_texts(["rag doc"], embeddings=[_vec(seed=0)])

        c = PC(chroma_path)
        col = c.get_or_create_collection("chroma_col")
        col.add(ids=["chroma_doc"], embeddings=[_vec(seed=1)])

        rag_results = r.similarity_search(_vec(seed=0), k=1)
        chroma_results = col.query(query_embeddings=[_vec(seed=1)], n_results=1)

        assert rag_results[0].page_content == "rag doc"
        assert chroma_results["ids"][0][0] == "chroma_doc"

    def test_chroma_high_cardinality_metadata_filter_performance(self, tmp_path):
        """Filter over 1000 docs with unique metadata keys should not crash."""
        _, col = _make_chroma_col(tmp_path, dim=8)
        n = 1000
        from tqdb.chroma_compat import PersistentClient as PC
        with tempfile.TemporaryDirectory() as td:
            c = PC(td)
            big_col = c.get_or_create_collection("bigmeta")
            big_col.add(
                ids=[f"d{i}" for i in range(n)],
                embeddings=_vecs(n, dim=8),
                metadatas=[{"bucket": i % 10, "rank": i} for i in range(n)],
            )
            r = big_col.query(
                query_embeddings=[_vec(dim=8)],
                n_results=5,
                where={"bucket": {"$eq": 3}},
            )
            for doc_id in r["ids"][0]:
                idx = int(doc_id[1:])
                assert idx % 10 == 3, f"filter miss: {doc_id}"

    def test_lance_large_batch_add_then_search_correctness(self, tmp_path):
        """Target vector must rank in top-3 among 500 candidates at dim=32."""
        db = _lance_db(tmp_path)
        dim = 32
        target_vec = _vec(dim=dim, seed=999)
        rows = [{"id": "target", "vector": target_vec}]
        rows += [{"id": f"noise_{i}", "vector": _vec(dim=dim, seed=i)} for i in range(499)]
        tbl = db.create_table("t", data=rows)
        results = tbl.search(target_vec).limit(3).to_list()
        ids_in_top3 = [r["id"] for r in results]
        assert "target" in ids_in_top3, f"target not in top-3; got: {ids_in_top3}"

    def test_rag_retriever_search_after_multiple_add_rounds_correct(self, tmp_path):
        r = _rag(tmp_path, dim=8)
        target = _vec(dim=8, seed=777)
        r.add_texts(["noise1", "noise2"], embeddings=_vecs(2, dim=8, seed=1))
        r.add_texts(["target doc"], embeddings=[target])
        r.add_texts(["noise3", "noise4"], embeddings=_vecs(2, dim=8, seed=3))
        results = r.similarity_search(target, k=1)
        assert results[0].page_content == "target doc"

    def test_chroma_collection_add_query_delete_query_cycle(self, tmp_path):
        _, col = _make_chroma_col(tmp_path)
        col.add(ids=["a", "b", "c"], embeddings=_vecs(3), metadatas=[{"x": i} for i in range(3)])
        r1 = col.query(query_embeddings=[_vec()], n_results=3)
        assert len(r1["ids"][0]) == 3
        col.delete(ids=["b"])
        r2 = col.query(query_embeddings=[_vec()], n_results=3)
        assert len(r2["ids"][0]) == 2
        assert "b" not in r2["ids"][0]

    def test_lance_add_delete_re_add_same_id_search_finds_new(self, tmp_path):
        """LanceDB allows duplicate ids in append mode; after delete the new row is top."""
        db = _lance_db(tmp_path)
        original_vec = _vec(seed=0)
        new_vec = _vec(seed=100)
        tbl = db.create_table("t", data=[{"id": "x", "vector": original_vec}])
        tbl.delete("id IN ('x')")
        tbl.add([{"id": "x", "vector": new_vec}], mode="append")
        results = tbl.search(new_vec).limit(1).to_list()
        assert results[0]["id"] == "x"
