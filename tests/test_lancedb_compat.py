"""Tests for the LanceDB compatibility shim (tqdb.lancedb_compat)."""

import numpy as np
import pytest

from tqdb.lancedb_compat import connect

pyarrow = pytest.importorskip("pyarrow", reason="pyarrow required for LanceDB shim tests")
import pyarrow as pa


DIM = 16


def rand_vecs(n: int, d: int = DIM) -> np.ndarray:
    return np.random.default_rng(7).random((n, d)).astype(np.float32)


def make_pa_table(n: int, d: int = DIM) -> pa.Table:
    vecs = rand_vecs(n, d)
    return pa.table({
        "id": pa.array([f"id_{i}" for i in range(n)]),
        "vector": pa.array(vecs.tolist(), type=pa.list_(pa.float32(), d)),
    })


def make_rows(n: int, d: int = DIM) -> list[dict]:
    vecs = rand_vecs(n, d)
    return [{"id": f"id_{i}", "vector": vecs[i].tolist()} for i in range(n)]


# ---------------------------------------------------------------------------
# connect / create_table / open_table
# ---------------------------------------------------------------------------

class TestConnection:
    def test_connect_creates_dir(self, tmp_path):
        db = connect(str(tmp_path / "db"))
        assert (tmp_path / "db").exists()

    def test_cloud_uri_raises(self, tmp_path):
        with pytest.raises(NotImplementedError):
            connect("s3://mybucket/prefix")

    def test_create_table_returns_table(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("docs", data=make_pa_table(5))
        assert tbl.name == "docs"

    def test_open_table_missing_raises(self, tmp_path):
        db = connect(str(tmp_path))
        with pytest.raises(ValueError, match="not found"):
            db.open_table("ghost")

    def test_open_existing_table(self, tmp_path):
        db = connect(str(tmp_path))
        db.create_table("t", data=make_pa_table(3))
        tbl = db.open_table("t")
        assert tbl.name == "t"

    def test_drop_table(self, tmp_path):
        db = connect(str(tmp_path))
        db.create_table("t", data=make_pa_table(2))
        db.drop_table("t")
        assert "t" not in db.table_names()

    def test_table_names(self, tmp_path):
        db = connect(str(tmp_path))
        db.create_table("a", data=make_pa_table(1))
        db.create_table("b", data=make_pa_table(1))
        assert sorted(db.table_names()) == ["a", "b"]

    def test_mode_overwrite_wipes_existing(self, tmp_path):
        db = connect(str(tmp_path))
        db.create_table("t", data=make_pa_table(5))
        db.create_table("t", data=make_pa_table(2), mode="overwrite")
        tbl = db.open_table("t")
        assert tbl.count_rows() == 2


# ---------------------------------------------------------------------------
# add — PyArrow Table and list[dict]
# ---------------------------------------------------------------------------

class TestAdd:
    def test_add_pyarrow_table(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=make_pa_table(4))
        assert tbl.count_rows() == 4

    def test_add_list_of_dicts(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=make_rows(3))
        assert tbl.count_rows() == 3

    def test_add_append(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=make_rows(2))
        extra = [{"id": f"extra_{i}", "vector": rand_vecs(1)[i % 1].tolist()} for i in range(3)]
        tbl.add(extra)
        assert tbl.count_rows() == 5

    def test_add_overwrite_resets(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=make_rows(5))
        tbl.add(make_rows(2), mode="overwrite")
        assert tbl.count_rows() == 2


# ---------------------------------------------------------------------------
# search + fluent query builder
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_returns_results(self, tmp_path):
        db = connect(str(tmp_path))
        rows = make_rows(10)
        tbl = db.create_table("t", data=rows)
        q = np.asarray(rows[0]["vector"], dtype=np.float64)
        results = tbl.search(q).limit(5).to_list()
        assert len(results) == 5

    def test_search_distance_in_result(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=make_rows(5))
        q = rand_vecs(1)[0].astype(np.float64)
        results = tbl.search(q).limit(3).to_list()
        assert all("_distance" in r for r in results)

    def test_search_metric_dot(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=make_rows(5))
        q = rand_vecs(1)[0].astype(np.float64)
        results = tbl.search(q).metric("dot").limit(3).to_list()
        assert len(results) == 3

    def test_search_metric_cosine(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=make_rows(5))
        q = rand_vecs(1)[0].astype(np.float64)
        results = tbl.search(q).metric("cosine").limit(3).to_list()
        assert len(results) == 3

    def test_fluent_nprobes_no_error(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=make_rows(5))
        q = rand_vecs(1)[0].astype(np.float64)
        results = tbl.search(q).nprobes(8).refine_factor(2).limit(3).to_list()
        assert len(results) == 3

    def test_search_to_arrow(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=make_rows(5))
        q = rand_vecs(1)[0].astype(np.float64)
        arrow_tbl = tbl.search(q).limit(2).to_arrow()
        assert arrow_tbl.num_rows == 2


# ---------------------------------------------------------------------------
# where filter
# ---------------------------------------------------------------------------

class TestWhereFilter:
    def test_search_where_id_in(self, tmp_path):
        db = connect(str(tmp_path))
        rows = make_rows(5)
        tbl = db.create_table("t", data=rows)
        q = rand_vecs(1)[0].astype(np.float64)
        results = tbl.search(q).where("id IN ('id_0', 'id_1')").limit(10).to_list()
        result_ids = {r["id"] for r in results}
        assert result_ids <= {"id_0", "id_1"}

    def test_complex_sql_raises(self, tmp_path):
        from tqdb.lancedb_compat import _parse_sql_where
        with pytest.raises(NotImplementedError):
            _parse_sql_where("id = 'a' AND score > 0.5")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_by_id_in(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=make_rows(5))
        tbl.delete("id IN ('id_0', 'id_1')")
        assert tbl.count_rows() == 3

    def test_delete_field_eq(self, tmp_path):
        db = connect(str(tmp_path))
        rows = [
            {"id": "a", "vector": rand_vecs(1)[0].tolist(), "tag": "del"},
            {"id": "b", "vector": rand_vecs(1)[0].tolist(), "tag": "keep"},
        ]
        tbl = db.create_table("t", data=rows)
        tbl.delete("tag = 'del'")
        assert tbl.count_rows() == 1


# ---------------------------------------------------------------------------
# count_rows / optimize
# ---------------------------------------------------------------------------

class TestCountAndOptimize:
    def test_count_rows_empty(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t")
        assert tbl.count_rows() == 0

    def test_count_rows_after_add(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=make_rows(7))
        assert tbl.count_rows() == 7

    def test_optimize_no_error(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=make_rows(3))
        tbl.optimize()  # should be a no-op without raising


# ---------------------------------------------------------------------------
# to_arrow / to_pandas — must return real records, not bare ID strings
# ---------------------------------------------------------------------------

class TestToArrow:
    def test_to_arrow_returns_records(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=make_rows(4))
        arrow_tbl = tbl.to_arrow()
        assert arrow_tbl.num_rows == 4
        assert "id" in arrow_tbl.schema.names

    def test_to_arrow_empty(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t")
        arrow_tbl = tbl.to_arrow()
        assert arrow_tbl.num_rows == 0

    def test_to_pandas_returns_records(self, tmp_path):
        pd = pytest.importorskip("pandas")
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=make_rows(3))
        df = tbl.to_pandas()
        assert len(df) == 3
        assert "id" in df.columns


# ---------------------------------------------------------------------------
# count_rows with id-based filter
# ---------------------------------------------------------------------------

class TestCountRowsFilter:
    def test_count_rows_id_in_filter(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=make_rows(5))
        assert tbl.count_rows("id IN ('id_0', 'id_1')") == 2

    def test_count_rows_id_in_single(self, tmp_path):
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=make_rows(5))
        assert tbl.count_rows("id IN ('id_0')") == 1


# ---------------------------------------------------------------------------
# SQL WHERE parser extensions
# ---------------------------------------------------------------------------

class TestSQLParser:
    def test_field_in_non_id(self):
        from tqdb.lancedb_compat import _parse_sql_where
        result = _parse_sql_where("tag IN ('a', 'b')")
        assert result == {"tag": {"$in": ["a", "b"]}}

    def test_field_neq(self):
        from tqdb.lancedb_compat import _parse_sql_where
        result = _parse_sql_where("status != 'deleted'")
        assert result == {"status": {"$ne": "deleted"}}

    def test_field_eq_integer(self):
        from tqdb.lancedb_compat import _parse_sql_where
        result = _parse_sql_where("count = 42")
        assert result == {"count": {"$eq": 42.0}}

    def test_field_gt_numeric(self):
        from tqdb.lancedb_compat import _parse_sql_where
        result = _parse_sql_where("score > 0.5")
        assert result == {"score": {"$gt": 0.5}}

    def test_field_gte_numeric(self):
        from tqdb.lancedb_compat import _parse_sql_where
        result = _parse_sql_where("score >= 1.0")
        assert result == {"score": {"$gte": 1.0}}

    def test_field_lt_numeric(self):
        from tqdb.lancedb_compat import _parse_sql_where
        result = _parse_sql_where("rank < 10")
        assert result == {"rank": {"$lt": 10.0}}

    def test_field_lte_numeric(self):
        from tqdb.lancedb_compat import _parse_sql_where
        result = _parse_sql_where("rank <= 5")
        assert result == {"rank": {"$lte": 5.0}}

    def test_field_in_used_in_delete(self, tmp_path):
        """field IN (...) for non-id field works end-to-end in delete()."""
        rows = [
            {"id": "a", "vector": rand_vecs(1)[0].tolist(), "tag": "del"},
            {"id": "b", "vector": rand_vecs(1)[0].tolist(), "tag": "del"},
            {"id": "c", "vector": rand_vecs(1)[0].tolist(), "tag": "keep"},
        ]
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=rows)
        tbl.delete("tag IN ('del')")
        assert tbl.count_rows() == 1


# ---------------------------------------------------------------------------
# Metric override warning
# ---------------------------------------------------------------------------

class TestMetricWarn:
    def test_metric_override_warns(self, tmp_path):
        """search().metric('cosine') on an ip-table should emit a warning."""
        import warnings
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=make_rows(5))  # default metric=ip
        q = rand_vecs(1)[0].astype(np.float32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tbl.search(q).metric("cosine").limit(3).to_list()
        assert any("cosine" in str(warning.message).lower() for warning in w), (
            "Expected a warning about metric mismatch"
        )

    def test_same_metric_no_warn(self, tmp_path):
        """No warning when the requested metric matches the table metric."""
        import warnings
        db = connect(str(tmp_path))
        tbl = db.create_table("t", data=make_rows(5))  # metric=ip
        q = rand_vecs(1)[0].astype(np.float32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tbl.search(q).metric("ip").limit(3).to_list()
        metric_warns = [x for x in w if "metric" in str(x.message).lower()]
        assert len(metric_warns) == 0


# ---------------------------------------------------------------------------
# Dim recovery from manifest.json when _lance_meta.json is absent
# ---------------------------------------------------------------------------

class TestDimRecovery:
    def test_dim_from_manifest_fallback(self, tmp_path):
        """open_table() still works after _lance_meta.json is deleted."""
        import os
        db = connect(str(tmp_path))
        db.create_table("t", data=make_rows(3))

        # Remove the shim sidecar — only tqdb's manifest.json remains
        lance_meta = tmp_path / "t" / "_lance_meta.json"
        if lance_meta.exists():
            lance_meta.unlink()

        tbl = db.open_table("t")
        q = rand_vecs(1)[0].astype(np.float32)
        results = tbl.search(q).limit(2).to_list()
        assert len(results) == 2
