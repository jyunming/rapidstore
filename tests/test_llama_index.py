"""Tests for the LlamaIndex VectorStore at ``tqdb.llama_index``."""

from __future__ import annotations

import gc
import shutil
import tempfile
from contextlib import contextmanager

import numpy as np
import pytest


llama_index = pytest.importorskip("llama_index.core")
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)


@contextmanager
def _tempdir():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        gc.collect()
        shutil.rmtree(path, ignore_errors=True)


def _node(id_: str, vec: np.ndarray, text: str = "", **metadata) -> TextNode:
    n = TextNode(id_=id_, text=text, metadata=dict(metadata))
    n.embedding = vec.tolist()
    return n


def _make_unit(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d).astype(np.float32)
    return v / max(np.linalg.norm(v), 1e-9)


# ── add / query / delete ────────────────────────────────────────────────


def test_add_query_round_trip():
    from tqdb.llama_index import TurboQuantVectorStore

    d = 16
    with _tempdir() as path:
        store = TurboQuantVectorStore.open(path, dimension=d, bits=4, metric="cosine")
        try:
            nodes = [
                _node("a", _make_unit(d, 0), text="alpha"),
                _node("b", _make_unit(d, 1), text="beta"),
                _node("c", _make_unit(d, 2), text="gamma"),
            ]
            store.add(nodes)

            q = VectorStoreQuery(
                query_embedding=_make_unit(d, 0).tolist(),
                similarity_top_k=2,
            )
            res = store.query(q)
            assert "a" in res.ids
            assert all(isinstance(s, float) for s in res.similarities)
            assert res.nodes[0].text in ("alpha", "beta", "gamma")
        finally:
            store._db.close()


def test_delete_by_ref_doc_id():
    from tqdb.llama_index import TurboQuantVectorStore

    d = 8
    with _tempdir() as path:
        store = TurboQuantVectorStore.open(path, dimension=d, bits=2, metric="cosine")
        try:
            store.add([
                _node("x", _make_unit(d, 0), text="x"),
                _node("y", _make_unit(d, 1), text="y"),
            ])
            store.delete("x")
            res = store.query(VectorStoreQuery(
                query_embedding=_make_unit(d, 0).tolist(), similarity_top_k=10
            ))
            assert "x" not in res.ids
            assert "y" in res.ids
        finally:
            store._db.close()


def test_delete_nodes_by_filter():
    from tqdb.llama_index import TurboQuantVectorStore

    d = 8
    with _tempdir() as path:
        store = TurboQuantVectorStore.open(path, dimension=d, bits=2, metric="cosine")
        try:
            store.add([
                _node("a", _make_unit(d, 0), text="a", category="x"),
                _node("b", _make_unit(d, 1), text="b", category="x"),
                _node("c", _make_unit(d, 2), text="c", category="y"),
            ])
            store.delete_nodes(filters=MetadataFilters(filters=[
                MetadataFilter(key="category", value="x")
            ]))
            res = store.query(VectorStoreQuery(
                query_embedding=_make_unit(d, 0).tolist(), similarity_top_k=10
            ))
            assert set(res.ids) == {"c"}
        finally:
            store._db.close()


# ── filter translation ─────────────────────────────────────────────────


def test_query_with_metadata_filters():
    from tqdb.llama_index import TurboQuantVectorStore

    d = 16
    with _tempdir() as path:
        store = TurboQuantVectorStore.open(path, dimension=d, bits=4, metric="cosine")
        try:
            store.add([
                _node("a", _make_unit(d, 0), text="a", year=2023),
                _node("b", _make_unit(d, 1), text="b", year=2022),
                _node("c", _make_unit(d, 2), text="c", year=2021),
            ])
            filters = MetadataFilters(filters=[
                MetadataFilter(key="year", value=2023, operator=FilterOperator.GTE)
            ])
            q = VectorStoreQuery(
                query_embedding=_make_unit(d, 0).tolist(),
                similarity_top_k=10,
                filters=filters,
            )
            res = store.query(q)
            assert set(res.ids) == {"a"}
        finally:
            store._db.close()


def test_query_with_compound_filter():
    from tqdb.llama_index import TurboQuantVectorStore

    d = 8
    with _tempdir() as path:
        store = TurboQuantVectorStore.open(path, dimension=d, bits=2, metric="cosine")
        try:
            store.add([
                _node("a", _make_unit(d, 0), text="a", year=2023, lang="en"),
                _node("b", _make_unit(d, 1), text="b", year=2023, lang="fr"),
                _node("c", _make_unit(d, 2), text="c", year=2022, lang="en"),
            ])
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="year", value=2023),
                    MetadataFilter(key="lang", value="en"),
                ],
                condition=FilterCondition.AND,
            )
            q = VectorStoreQuery(
                query_embedding=_make_unit(d, 0).tolist(),
                similarity_top_k=10,
                filters=filters,
            )
            res = store.query(q)
            assert set(res.ids) == {"a"}
        finally:
            store._db.close()


# ── error paths ─────────────────────────────────────────────────────────


def test_add_without_embedding_raises():
    from tqdb.llama_index import TurboQuantVectorStore

    with _tempdir() as path:
        store = TurboQuantVectorStore.open(path, dimension=8, bits=2, metric="cosine")
        try:
            n = TextNode(id_="x", text="text", metadata={})
            # n.embedding is None
            with pytest.raises(ValueError, match="embedding"):
                store.add([n])
        finally:
            store._db.close()


def test_query_without_embedding_raises():
    from tqdb.llama_index import TurboQuantVectorStore

    with _tempdir() as path:
        store = TurboQuantVectorStore.open(path, dimension=8, bits=2, metric="cosine")
        try:
            with pytest.raises(ValueError, match="query_embedding"):
                store.query(VectorStoreQuery(query_embedding=None))
        finally:
            store._db.close()


def test_persist_calls_checkpoint():
    from tqdb.llama_index import TurboQuantVectorStore

    d = 8
    with _tempdir() as path:
        store = TurboQuantVectorStore.open(path, dimension=d, bits=2, metric="cosine")
        try:
            store.add([_node("x", _make_unit(d, 0), text="x")])
            # persist must not raise.
            store.persist()
        finally:
            store._db.close()


def test_clear_removes_all_nodes():
    from tqdb.llama_index import TurboQuantVectorStore

    d = 8
    with _tempdir() as path:
        store = TurboQuantVectorStore.open(path, dimension=d, bits=2, metric="cosine")
        try:
            store.add([
                _node("a", _make_unit(d, 0), text="a"),
                _node("b", _make_unit(d, 1), text="b"),
            ])
            store.clear()
            res = store.query(VectorStoreQuery(
                query_embedding=_make_unit(d, 0).tolist(), similarity_top_k=10
            ))
            assert res.ids == []
        finally:
            store._db.close()
