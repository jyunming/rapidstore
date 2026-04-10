import numpy as np
import pytest

from tqdb import Database


def _safe_error(exc_info: pytest.ExceptionInfo[BaseException]) -> bool:
    """Reject Rust panic surfacing through PyO3 as acceptable API behavior."""
    return "panic" not in exc_info.type.__name__.lower()


class TestDimensionSafety:
    def test_insert_dimension_mismatch_is_python_error_not_panic(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        with pytest.raises(BaseException) as exc_info:
            db.insert("bad", np.ones(7, dtype=np.float32))
        assert _safe_error(exc_info), f"unexpected panic type: {exc_info.type.__name__}"

    def test_insert_batch_dimension_mismatch_is_python_error_not_panic(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        with pytest.raises(BaseException) as exc_info:
            db.insert_batch(["a", "b"], np.ones((2, 7), dtype=np.float32))
        assert _safe_error(exc_info), f"unexpected panic type: {exc_info.type.__name__}"

    def test_search_dimension_mismatch_is_python_error_not_panic(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        db.insert("x", np.ones(8, dtype=np.float32))
        with pytest.raises(BaseException) as exc_info:
            db.search(np.ones(7, dtype=np.float32), top_k=1)
        assert _safe_error(exc_info), f"unexpected panic type: {exc_info.type.__name__}"

    def test_query_dimension_mismatch_is_python_error_not_panic(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        db.insert("x", np.ones(8, dtype=np.float32))
        with pytest.raises(BaseException) as exc_info:
            db.query(np.ones((2, 7), dtype=np.float32), n_results=1)
        assert _safe_error(exc_info), f"unexpected panic type: {exc_info.type.__name__}"


class TestInputHardening:
    def test_reject_nan_insert(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        with pytest.raises(Exception):
            db.insert("nan", np.full(8, np.nan, dtype=np.float32))

    def test_reject_nan_query(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        db.insert("x", np.ones(8, dtype=np.float32))
        with pytest.raises(Exception):
            db.search(np.full(8, np.nan, dtype=np.float32), top_k=1)

    def test_invalid_include_field_raises(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        db.insert("x", np.ones(8, dtype=np.float32))
        with pytest.raises(Exception):
            db.search(np.ones(8, dtype=np.float32), top_k=1, include=["not_a_field"])

    def test_reject_inf_insert(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        with pytest.raises(Exception):
            db.insert("inf", np.full(8, np.inf, dtype=np.float32))

    def test_reject_inf_query(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        db.insert("x", np.ones(8, dtype=np.float32))
        with pytest.raises(Exception):
            db.search(np.full(8, np.inf, dtype=np.float32), top_k=1)

    def test_search_never_returns_nan_score(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        db.insert("x", np.ones(8, dtype=np.float32))
        db.insert("y", np.zeros(8, dtype=np.float32))
        # inf vectors are rejected at insert time; verify normal search returns finite scores
        with pytest.raises(Exception):
            db.insert("inf", np.full(8, np.inf, dtype=np.float32))
        out = db.search(np.ones(8, dtype=np.float32), top_k=2)
        scores = [r.get("score") for r in out]
        assert all(np.isfinite(s) for s in scores), scores


class TestValidationStrictness:
    def test_open_rejects_bits_below_2_without_panic(self, tmp_path):
        with pytest.raises(BaseException) as exc_info:
            Database.open(str(tmp_path / "db"), dimension=8, bits=1, metric="ip")
        assert _safe_error(exc_info), f"unexpected panic type: {exc_info.type.__name__}"

    def test_filter_unknown_operator_raises_count(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        db.insert("x", np.ones(8, dtype=np.float32), metadata={"year": 2024})
        with pytest.raises(Exception):
            db.count(filter={"year": {"$unknown": 1}})

    def test_filter_unknown_operator_raises_search(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        db.insert("x", np.ones(8, dtype=np.float32), metadata={"year": 2024})
        with pytest.raises(Exception):
            db.search(np.ones(8, dtype=np.float32), top_k=1, filter={"year": {"$unknown": 1}})

    def test_filter_unknown_operator_raises_list_ids(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        db.insert("x", np.ones(8, dtype=np.float32), metadata={"year": 2024})
        with pytest.raises(Exception):
            db.list_ids(where_filter={"year": {"$unknown": 1}})

    def test_filter_unknown_operator_raises_query(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        db.insert("x", np.ones(8, dtype=np.float32), metadata={"year": 2024})
        with pytest.raises(Exception):
            db.query(
                np.ones((1, 8), dtype=np.float32),
                n_results=1,
                where_filter={"year": {"$unknown": 1}},
            )

    def test_filter_unknown_operator_raises_delete_batch(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        db.insert("x", np.ones(8, dtype=np.float32), metadata={"year": 2024})
        with pytest.raises(Exception):
            db.delete_batch(where_filter={"year": {"$unknown": 1}})

    def test_search_negative_topk_is_validation_error(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        db.insert("x", np.ones(8, dtype=np.float32))
        with pytest.raises(ValueError):
            db.search(np.ones(8, dtype=np.float32), top_k=-1)

    def test_query_negative_n_results_is_validation_error(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        db.insert("x", np.ones(8, dtype=np.float32))
        with pytest.raises(ValueError):
            db.query(np.ones((1, 8), dtype=np.float32), n_results=-1)

    def test_list_ids_negative_offset_is_validation_error(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        db.insert("x", np.ones(8, dtype=np.float32))
        with pytest.raises(ValueError):
            db.list_ids(offset=-1)

    def test_list_ids_negative_limit_is_validation_error(self, tmp_path):
        db = Database.open(str(tmp_path / "db"), dimension=8, bits=4, metric="ip")
        db.insert("x", np.ones(8, dtype=np.float32))
        with pytest.raises(ValueError):
            db.list_ids(limit=-1)


class TestPathSafety:
    def test_collection_disallows_path_traversal(self, tmp_path):
        base = tmp_path / "base"
        outside = tmp_path / "outside"
        base.mkdir()
        outside.mkdir()

        escaped = "..\\outside\\escaped_collection"
        with pytest.raises(Exception):
            Database.open(str(base), dimension=8, bits=4, collection=escaped)

        escaped_manifest = outside / "escaped_collection" / "manifest.json"
        assert not escaped_manifest.exists()
