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
