import pytest
import threading

from tqdb.chroma_compat import PersistentClient, _apply_filter
from tqdb.lancedb_compat import _parse_sql_where, connect


class TestChromaCompatFilterValidation:
    def test_unknown_operator_rejected(self):
        records = [{"id": "a", "metadata": {"year": 2024}}]
        with pytest.raises(Exception):
            _apply_filter(records, {"year": {"$unknown": 1}})

    def test_unknown_operator_rejected_inside_and(self):
        records = [{"id": "a", "metadata": {"year": 2024}}]
        with pytest.raises(Exception):
            _apply_filter(records, {"$and": [{"year": {"$unknown": 1}}]})


class TestLanceCompatSqlParsing:
    def test_trailing_comma_in_clause_is_rejected(self):
        with pytest.raises(Exception):
            _parse_sql_where("topic IN ('ml',)")

    def test_negative_number_comparison_is_supported(self):
        parsed = _parse_sql_where("score >= -1")
        assert parsed == {"score": {"$gte": -1.0}}


class TestLanceCompatQueryValidation:
    def test_negative_limit_raises_clean_validation_error(self, tmp_path):
        db = connect(str(tmp_path))
        table = db.create_table(
            "docs",
            data=[{"id": "a", "vector": [1.0] * 8}, {"id": "b", "vector": [0.1] * 8}],
        )
        with pytest.raises(ValueError):
            table.search([1.0] * 8).limit(-1).to_list()

    def test_add_rejects_unknown_mode(self, tmp_path):
        db = connect(str(tmp_path))
        table = db.create_table("docs", data=[{"id": "a", "vector": [1.0] * 8}])
        with pytest.raises(ValueError):
            table.add([{"id": "b", "vector": [1.0] * 8}], mode="invalid-mode")


class TestPathSafetyCompat:
    def test_chroma_rejects_collection_name_traversal(self, tmp_path):
        client = PersistentClient(str(tmp_path / "root"))
        with pytest.raises(Exception):
            client.get_or_create_collection("..\\escape_col")

    def test_lance_rejects_table_name_traversal(self, tmp_path):
        db = connect(str(tmp_path / "root"))
        with pytest.raises(Exception):
            db.create_table("..\\escape_tbl", data=[{"id": "a", "vector": [1.0] * 8}])


class TestChromaIncludeValidation:
    def test_get_rejects_unknown_include_field(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[[1.0] * 8])
        with pytest.raises(Exception):
            col.get(ids=["a"], include=["bad_field"])

    def test_query_rejects_unknown_include_field(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        col = client.get_or_create_collection("c")
        col.add(ids=["a"], embeddings=[[1.0] * 8])
        with pytest.raises(Exception):
            col.query(query_embeddings=[[1.0] * 8], n_results=1, include=["bad_field"])


class TestCompatConcurrency:
    def test_chroma_create_collection_concurrent_single_winner(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        outcomes = []

        def _worker():
            try:
                client.create_collection("race")
                outcomes.append("ok")
            except Exception as e:
                outcomes.append(type(e).__name__)

        threads = [threading.Thread(target=_worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert outcomes.count("ok") == 1, outcomes
        assert all(x in ("ok", "ValueError") for x in outcomes), outcomes

    def test_chroma_get_or_create_concurrent_no_json_corruption(self, tmp_path):
        client = PersistentClient(str(tmp_path))
        outcomes = []

        def _worker():
            try:
                client.get_or_create_collection("race")
                outcomes.append("ok")
            except Exception as e:
                outcomes.append(type(e).__name__)

        threads = [threading.Thread(target=_worker) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(x == "ok" for x in outcomes), outcomes

    def test_lance_create_table_concurrent_no_metadata_corruption(self, tmp_path):
        db = connect(str(tmp_path))
        outcomes = []

        def _worker():
            try:
                db.create_table("race", data=[{"id": "a", "vector": [1.0] * 8}])
                outcomes.append("ok")
            except Exception as e:
                outcomes.append(type(e).__name__)

        threads = [threading.Thread(target=_worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert outcomes.count("ok") == 1, outcomes
        assert all(x in ("ok", "ValueError") for x in outcomes), outcomes
