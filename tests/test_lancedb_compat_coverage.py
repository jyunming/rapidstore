"""
Coverage-gap tests for tqdb.lancedb_compat.

Targets lines identified as missing in the 91% coverage report:
  61, 158, 191, 215, 237, 239, 241, 249, 253-254,
  291, 316, 321, 374, 390-391, 400, 407, 462, 486

Lines 42-43 and 154-155 (ImportError fallbacks) are intentionally excluded —
they require the extension to be absent, which is impossible in this venv.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import pyarrow as pa

from tqdb.lancedb_compat import (
    CompatTable,
    CompatQuery,
    _map_metric,
    _extract_rows,
    connect,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DIM = 8
_rng = np.random.default_rng(99)


def rand_vec(d: int = DIM) -> np.ndarray:
    return _rng.random(d).astype(np.float32)


def rand_vecs(n: int, d: int = DIM) -> np.ndarray:
    return _rng.random((n, d)).astype(np.float32)


def list_rows(n: int, d: int = DIM, *, meta: bool = False, doc: bool = False) -> list[dict]:
    """Build n rows with plain Python list vectors, optional metadata and document."""
    rows = []
    for i in range(n):
        r: dict = {"id": f"id_{i}", "vector": rand_vec(d).tolist()}
        if meta:
            r["category"] = "alpha" if i % 2 == 0 else "beta"
        if doc:
            r["document"] = f"document text {i}"
        rows.append(r)
    return rows


def make_tbl(tmp_path, n: int = 5, **kw) -> CompatTable:
    conn = connect(str(tmp_path / "db"))
    return conn.create_table("t", data=list_rows(n, **kw))


# ===========================================================================
# _map_metric  (line 61)
# ===========================================================================

class TestMapMetric:
    def test_unknown_metric_raises_value_error(self):
        """Line 61: ValueError for unrecognised metric name."""
        with pytest.raises(ValueError, match="Unsupported metric 'manhattan'"):
            _map_metric("manhattan")

    def test_unknown_metric_case_insensitive_check(self):
        """Confirm that only truly unrecognised strings raise — not just odd casing."""
        with pytest.raises(ValueError):
            _map_metric("MANHATTAN")

    def test_known_metrics_do_not_raise(self):
        """Sanity: recognised metrics must not raise."""
        for m in ("dot", "ip", "cosine", "l2", "euclidean", "DOT", "L2"):
            _map_metric(m)  # must not raise


# ===========================================================================
# _extract_rows  (lines 154-155 skipped — ImportError path; line 158)
# ===========================================================================

class TestExtractRows:
    def test_pyarrow_table_returns_list_of_dicts(self):
        """Lines 152-153: pyarrow Table is unwrapped to list[dict]."""
        vecs = rand_vecs(3)
        tbl = pa.table({
            "id": pa.array(["a", "b", "c"]),
            "vector": pa.array(vecs.tolist(), type=pa.list_(pa.float32(), DIM)),
        })
        rows = _extract_rows(tbl)
        assert isinstance(rows, list)
        assert len(rows) == 3
        assert rows[0]["id"] == "a"

    def test_invalid_type_raises_type_error(self):
        """Line 158: non-Table, non-list input raises TypeError."""
        with pytest.raises(TypeError, match="data must be a PyArrow Table or list"):
            _extract_rows(42)

    def test_invalid_type_string_raises(self):
        """Line 158: string is also rejected."""
        with pytest.raises(TypeError):
            _extract_rows("not a table")


# ===========================================================================
# CompatQuery.where()  (lines 190-191)
# ===========================================================================

class TestCompatQueryWhere:
    def test_where_sets_filter_and_returns_self(self, tmp_path):
        """Lines 190-191: where() stores the filter string and returns self."""
        tbl = make_tbl(tmp_path)
        q = tbl.search(rand_vec().astype(np.float64))
        result = q.where("id IN ('id_0')")
        # Returns self for method chaining
        assert result is q
        assert q._where == "id IN ('id_0')"

    def test_where_with_prefilter_kwarg(self, tmp_path):
        """prefilter kwarg is accepted without error (it's silently ignored)."""
        tbl = make_tbl(tmp_path)
        q = tbl.search(rand_vec().astype(np.float64))
        result = q.where("id IN ('id_0')", prefilter=True)
        assert result is q


# ===========================================================================
# CompatQuery.to_list() — non-id tqdb_filter path  (line 215)
# ===========================================================================

class TestCompatQueryNonIdFilter:
    def test_metadata_eq_filter_hits_line_215(self, tmp_path):
        """
        Line 215: when WHERE clause is a field equality (not an id IN),
        the parsed dict is stored as tqdb_filter and forwarded to db.search().
        """
        conn = connect(str(tmp_path / "db"))
        rows = list_rows(6, meta=True)
        tbl = conn.create_table("t", data=rows)

        q_vec = rand_vec().astype(np.float64)
        results = tbl.search(q_vec).where("category = 'alpha'").limit(10).to_list()

        # All returned rows should match the filter
        assert all(r.get("category") == "alpha" for r in results)

    def test_metadata_neq_filter(self, tmp_path):
        """Line 215: != filter also routes through tqdb_filter."""
        conn = connect(str(tmp_path / "db"))
        rows = list_rows(4, meta=True)
        tbl = conn.create_table("t", data=rows)

        q_vec = rand_vec().astype(np.float64)
        results = tbl.search(q_vec).where("category != 'beta'").limit(10).to_list()
        assert all(r.get("category") != "beta" for r in results)


# ===========================================================================
# CompatQuery.to_list() — metadata fields + document (lines 237, 239)
# ===========================================================================

class TestCompatQueryRowFields:
    def test_metadata_fields_merged_into_row(self, tmp_path):
        """Line 237: metadata key-value pairs are expanded into the row dict."""
        tbl = make_tbl(tmp_path, meta=True)
        q_vec = rand_vec().astype(np.float64)
        results = tbl.search(q_vec).limit(5).to_list()
        # Every row should carry the 'category' metadata field
        assert all("category" in r for r in results)
        assert all(r["category"] in ("alpha", "beta") for r in results)

    def test_document_field_present_when_ingested(self, tmp_path):
        """Line 239: document is included in result when the row has one."""
        tbl = make_tbl(tmp_path, doc=True)
        q_vec = rand_vec().astype(np.float64)
        results = tbl.search(q_vec).limit(5).to_list()
        assert all("document" in r for r in results)
        assert all(r["document"].startswith("document text") for r in results)

    def test_both_metadata_and_document(self, tmp_path):
        """Lines 237 & 239 together: row with both metadata and a document."""
        tbl = make_tbl(tmp_path, meta=True, doc=True)
        q_vec = rand_vec().astype(np.float64)
        results = tbl.search(q_vec).limit(3).to_list()
        for r in results:
            assert "category" in r
            assert "document" in r


# ===========================================================================
# CompatQuery.to_list() — select projection  (line 241)
# ===========================================================================

class TestCompatQuerySelect:
    def test_select_limits_keys(self, tmp_path):
        """Line 241: .select() restricts result keys to the requested columns."""
        tbl = make_tbl(tmp_path, meta=True, doc=True)
        q_vec = rand_vec().astype(np.float64)
        results = (
            tbl.search(q_vec)
            .limit(5)
            .select(["id", "document"])
            .to_list()
        )
        for r in results:
            assert set(r.keys()) <= {"id", "document"}
            # 'category' and '_distance' must be absent
            assert "category" not in r
            assert "_distance" not in r

    def test_select_missing_column_silently_skipped(self, tmp_path):
        """Line 241: columns not present in the row are silently omitted."""
        tbl = make_tbl(tmp_path)
        q_vec = rand_vec().astype(np.float64)
        results = (
            tbl.search(q_vec)
            .limit(3)
            .select(["id", "nonexistent_col"])
            .to_list()
        )
        for r in results:
            assert "id" in r
            assert "nonexistent_col" not in r


# ===========================================================================
# CompatQuery.to_arrow() — empty result  (line 249)
# ===========================================================================

class TestCompatQueryToArrowEmpty:
    def test_to_arrow_returns_empty_table_when_no_results(self, tmp_path):
        """Line 249: to_arrow() returns an empty pa.Table when to_list() is empty."""
        tbl = make_tbl(tmp_path)
        q_vec = rand_vec().astype(np.float64)
        # Use a nonexistent id so the id_allowset filter yields no matches
        arr = (
            tbl.search(q_vec)
            .where("id IN ('__no_such_id__')")
            .to_arrow()
        )
        assert isinstance(arr, pa.Table)
        assert arr.num_rows == 0


# ===========================================================================
# CompatQuery.to_pandas()  (lines 253-254)
# ===========================================================================

class TestCompatQueryToPandas:
    def test_to_pandas_returns_dataframe_with_results(self, tmp_path):
        """Lines 253-254: to_pandas() wraps to_list() in a DataFrame."""
        import pandas as pd

        tbl = make_tbl(tmp_path, n=4)
        q_vec = rand_vec().astype(np.float64)
        df = tbl.search(q_vec).limit(4).to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "_distance" in df.columns

    def test_to_pandas_empty_when_no_results(self, tmp_path):
        """Lines 253-254: returns empty DataFrame when search has zero matches."""
        import pandas as pd

        tbl = make_tbl(tmp_path)
        q_vec = rand_vec().astype(np.float64)
        df = (
            tbl.search(q_vec)
            .where("id IN ('__nothing__')")
            .to_pandas()
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ===========================================================================
# CompatTable._open_db() — dimension unknown  (line 291)
# ===========================================================================

class TestCompatTableOpenDbNoDim:
    def test_open_db_raises_when_dim_unknown(self, tmp_path):
        """
        Line 291: _open_db() raises RuntimeError when no dimension has been
        inferred (no _lance_meta.json and no manifest.json in the table dir).
        """
        tbl_dir = str(tmp_path / "empty_tbl")
        os.makedirs(tbl_dir)
        tbl = CompatTable(tbl_dir, "empty_tbl", "ip")
        assert tbl._dim is None  # precondition
        with pytest.raises(RuntimeError, match="dimension unknown"):
            tbl._open_db()


# ===========================================================================
# CompatTable.add() — early return on empty rows (line 316) +
#                      numpy-array vector path (line 321)
# ===========================================================================

class TestCompatTableAddPaths:
    def test_add_empty_list_is_noop(self, tmp_path):
        """Line 316: add([]) returns immediately without touching the DB."""
        conn = connect(str(tmp_path / "db"))
        tbl = conn.create_table("t")  # no initial data — no manifest yet
        tbl.add([])                   # must not raise
        # count_rows returns 0 because manifest never got written
        assert tbl.count_rows() == 0

    def test_add_empty_pyarrow_table_is_noop(self, tmp_path):
        """Line 316: adding an empty pa.Table also short-circuits."""
        conn = connect(str(tmp_path / "db"))
        tbl = conn.create_table("t")
        empty_pa = pa.table({
            "id": pa.array([], type=pa.string()),
            "vector": pa.array([], type=pa.list_(pa.float32(), DIM)),
        })
        tbl.add(empty_pa)  # must not raise
        assert tbl.count_rows() == 0

    def test_add_numpy_array_vectors_uses_stack_path(self, tmp_path):
        """
        Line 321: when vectors are numpy arrays (not list/tuple), np.stack()
        is used instead of np.asarray().
        """
        conn = connect(str(tmp_path / "db"))
        tbl = conn.create_table("t")
        vecs = rand_vecs(4)
        # Pass numpy arrays directly — triggers the else branch at line 320-321
        rows = [{"id": f"np_{i}", "vector": vecs[i]} for i in range(4)]
        tbl.add(rows)
        assert tbl.count_rows() == 4

    def test_add_mixed_then_numpy(self, tmp_path):
        """Line 321: second add with numpy arrays still works after a list-add."""
        conn = connect(str(tmp_path / "db"))
        tbl = conn.create_table("t", data=list_rows(2))  # list vectors
        vecs = rand_vecs(3)
        rows = [{"id": f"np_{i}", "vector": vecs[i]} for i in range(3)]
        tbl.add(rows)  # numpy vectors — line 321
        assert tbl.count_rows() == 5


# ===========================================================================
# CompatTable.count_rows() — id IN filter  (line 374)
# ===========================================================================

class TestCompatTableCountRowsIdFilter:
    def test_count_rows_id_in_filter(self, tmp_path):
        """Line 374: count_rows with id IN (...) uses list_all() + set intersection."""
        tbl = make_tbl(tmp_path, n=6)
        count = tbl.count_rows(filter="id IN ('id_0', 'id_2', 'id_4')")
        assert count == 3

    def test_count_rows_id_in_single_id(self, tmp_path):
        """Line 374: single id in the IN list."""
        tbl = make_tbl(tmp_path, n=4)
        assert tbl.count_rows(filter="id IN ('id_1')") == 1

    def test_count_rows_id_in_no_match(self, tmp_path):
        """Line 374: id IN filter with no matching ids returns 0."""
        tbl = make_tbl(tmp_path, n=3)
        assert tbl.count_rows(filter="id IN ('__ghost__')") == 0

    def test_count_rows_non_id_filter(self, tmp_path):
        """Line 374: non-id field filter delegates to db.count(filter=...)."""
        tbl = make_tbl(tmp_path, n=6, meta=True)
        # 3 of 6 rows have category='alpha' (even indices)
        count = tbl.count_rows(filter="category = 'alpha'")
        assert count == 3

    def test_count_rows_non_id_filter_no_match(self, tmp_path):
        """Line 374: non-id field filter with no matches returns 0."""
        tbl = make_tbl(tmp_path, n=4, meta=True)
        assert tbl.count_rows(filter="category = 'gamma'") == 0


# ===========================================================================
# CompatTable.create_index()  (lines 390-391)
# ===========================================================================

class TestCompatTableCreateIndex:
    def test_create_index_delegates_to_db(self, tmp_path):
        """Lines 390-391: create_index() opens the DB and calls db.create_index()."""
        tbl = make_tbl(tmp_path, n=5)
        tbl.create_index()  # must not raise

    def test_create_index_with_kwargs_no_error(self, tmp_path):
        """create_index() accepts LanceDB keyword args without error."""
        tbl = make_tbl(tmp_path, n=5)
        tbl.create_index(metric="L2", index_type="IVF_PQ", num_partitions=4)


# ===========================================================================
# CompatTable.to_arrow() — empty-ids path (line 400)
# CompatTable.to_pandas() — no-manifest path (line 407)
# ===========================================================================

class TestCompatTableEmptyPaths:
    def test_to_arrow_returns_empty_table_when_all_rows_deleted(self, tmp_path):
        """
        Line 400: to_arrow() returns pa.table({}) when manifest.json exists
        but list_all() returns an empty list (all records were deleted).
        """
        conn = connect(str(tmp_path / "db"))
        tbl = conn.create_table("t", data=list_rows(3))
        # Delete every row so list_all() returns []
        tbl.delete("id IN ('id_0', 'id_1', 'id_2')")
        arr = tbl.to_arrow()
        assert isinstance(arr, pa.Table)
        assert arr.num_rows == 0

    def test_to_pandas_returns_empty_df_when_no_manifest(self, tmp_path):
        """
        Line 407: to_pandas() returns pd.DataFrame() when manifest.json
        has never been written (table created but no data ever added).
        """
        import pandas as pd

        conn = connect(str(tmp_path / "db"))
        tbl = conn.create_table("t")  # no data → no manifest.json
        df = tbl.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ===========================================================================
# CompatConnection.create_table() — invalid mode  (line 462)
# CompatConnection.drop_table()   — nonexistent table  (line 486)
# ===========================================================================

class TestCompatConnectionEdgeCases:
    def test_create_table_invalid_mode_raises(self, tmp_path):
        """Line 462: mode other than 'create'/'overwrite' raises ValueError."""
        conn = connect(str(tmp_path))
        with pytest.raises(ValueError, match="Unsupported mode 'upsert'"):
            conn.create_table("t", mode="upsert")

    def test_create_table_invalid_mode_message(self, tmp_path):
        """Line 462: error message mentions valid modes."""
        conn = connect(str(tmp_path))
        with pytest.raises(ValueError, match="'create' or 'overwrite'"):
            conn.create_table("t", mode="update")

    def test_drop_table_nonexistent_raises(self, tmp_path):
        """Line 486: drop_table() raises ValueError for a table that doesn't exist."""
        conn = connect(str(tmp_path))
        with pytest.raises(ValueError, match="not found"):
            conn.drop_table("ghost_table")

    def test_drop_table_already_dropped_raises(self, tmp_path):
        """Line 486: dropping a table twice raises on the second call."""
        conn = connect(str(tmp_path))
        conn.create_table("t", data=list_rows(2))
        conn.drop_table("t")
        with pytest.raises(ValueError, match="not found"):
            conn.drop_table("t")
