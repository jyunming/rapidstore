"""Tests for ``tqdb.vectorstore.TurboQuantVectorStore`` against LangChain v2.

The integration is minimal — TQDB's job is just storage + ANN. These tests
exercise the public ABC surface and confirm the docs returned look correct.
"""

from __future__ import annotations

import gc
import shutil
import tempfile
from contextlib import contextmanager
from typing import List

import numpy as np
import pytest


# Skip the whole module if langchain-core isn't installed in CI.
langchain_core = pytest.importorskip("langchain_core")
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


@contextmanager
def _tempdir():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        gc.collect()
        shutil.rmtree(path, ignore_errors=True)


class _FakeEmbedder(Embeddings):
    """Deterministic 'embedder' mapping each string to a fixed sparse-ish vector.

    We can't pull a real embedding model into CI, so we hash each token to a
    coordinate index and accumulate. Distinct strings → distinct vectors;
    similar substrings → similar vectors.
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> List[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            idx = sum(ord(c) for c in tok) % self.dim
            v[idx] += 1.0
        n = float(np.linalg.norm(v))
        if n > 1e-9:
            v /= n
        return v.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)


# ── core ABC ────────────────────────────────────────────────────────────


def test_from_texts_round_trip():
    from tqdb.vectorstore import TurboQuantVectorStore

    embed = _FakeEmbedder(dim=32)
    with _tempdir() as path:
        store = TurboQuantVectorStore.from_texts(
            ["alpha bravo", "charlie delta", "echo foxtrot"],
            embedding=embed,
            path=path,
        )
        try:
            results = store.similarity_search("alpha bravo", k=2)
            assert len(results) <= 2
            assert isinstance(results[0], Document)
            # The exact-match doc must surface.
            assert any("alpha" in r.page_content for r in results)
        finally:
            store._db.close()


def test_add_documents_and_delete():
    from tqdb.vectorstore import TurboQuantVectorStore

    embed = _FakeEmbedder(dim=32)
    with _tempdir() as path:
        store = TurboQuantVectorStore.open(
            path, embedding=embed, dimension=32, bits=4, metric="cosine"
        )
        try:
            ids = ["a", "b", "c"]
            docs = [
                Document(page_content="alpha", metadata={"k": 1}),
                Document(page_content="bravo", metadata={"k": 2}),
                Document(page_content="charlie", metadata={"k": 3}),
            ]
            returned = store.add_documents(docs, ids=ids)
            assert returned == ids

            fetched = store.get_by_ids(["a", "c"])
            contents = {d.page_content for d in fetched}
            assert contents == {"alpha", "charlie"}

            assert store.delete(["b"]) is True
            remaining = store.get_by_ids(["a", "b", "c"])
            assert {d.id for d in remaining} == {"a", "c"}
        finally:
            store._db.close()


def test_similarity_search_with_score_returns_floats():
    from tqdb.vectorstore import TurboQuantVectorStore

    embed = _FakeEmbedder(dim=32)
    with _tempdir() as path:
        store = TurboQuantVectorStore.from_texts(
            ["one fish", "two fish", "red fish", "blue fish"],
            embedding=embed,
            path=path,
        )
        try:
            r = store.similarity_search_with_score("blue fish", k=3)
            assert len(r) == 3
            for doc, score in r:
                assert isinstance(doc, Document)
                assert isinstance(score, float)
        finally:
            store._db.close()


def test_similarity_search_by_vector():
    from tqdb.vectorstore import TurboQuantVectorStore

    embed = _FakeEmbedder(dim=32)
    with _tempdir() as path:
        store = TurboQuantVectorStore.from_texts(
            ["alpha", "beta", "gamma"], embedding=embed, path=path
        )
        try:
            qvec = embed.embed_query("alpha")
            r = store.similarity_search_by_vector(qvec, k=1)
            assert len(r) == 1
            assert "alpha" in r[0].page_content
        finally:
            store._db.close()


def test_filter_passthrough_uses_metadata():
    from tqdb.vectorstore import TurboQuantVectorStore

    embed = _FakeEmbedder(dim=32)
    with _tempdir() as path:
        store = TurboQuantVectorStore.open(
            path, embedding=embed, dimension=32, bits=4, metric="cosine"
        )
        try:
            store.add_documents(
                [
                    Document(page_content="cat", metadata={"animal": "feline"}),
                    Document(page_content="dog", metadata={"animal": "canine"}),
                ],
                ids=["c", "d"],
            )
            r = store.similarity_search("cat", k=5, filter={"animal": "feline"})
            assert {x.id for x in r} == {"c"}
        finally:
            store._db.close()


def test_relevance_score_fn_returns_callable():
    from tqdb.vectorstore import TurboQuantVectorStore

    embed = _FakeEmbedder(dim=32)
    with _tempdir() as path:
        store = TurboQuantVectorStore.open(
            path, embedding=embed, dimension=32, metric="cosine"
        )
        try:
            fn = store._select_relevance_score_fn()
            assert callable(fn)
            # Cosine score 1.0 → relevance ~ 1.0
            assert fn(1.0) == pytest.approx(1.0, abs=1e-3)
            assert fn(-1.0) == pytest.approx(0.0, abs=1e-3)
        finally:
            store._db.close()


def test_as_retriever_works():
    """as_retriever is provided by the LangChain VectorStore base class — just
    confirm it returns a Runnable that can invoke a query end-to-end."""
    from tqdb.vectorstore import TurboQuantVectorStore

    embed = _FakeEmbedder(dim=32)
    with _tempdir() as path:
        store = TurboQuantVectorStore.from_texts(
            ["alpha", "beta", "gamma"], embedding=embed, path=path
        )
        try:
            retriever = store.as_retriever(search_kwargs={"k": 2})
            results = retriever.invoke("alpha")
            assert len(results) <= 2
            assert all(isinstance(r, Document) for r in results)
        finally:
            store._db.close()


# ── error paths ─────────────────────────────────────────────────────────


def test_no_embedding_required_for_by_vector():
    from tqdb.vectorstore import TurboQuantVectorStore

    with _tempdir() as path:
        store = TurboQuantVectorStore.open(
            path, embedding=None, dimension=8, metric="cosine"
        )
        try:
            # Insert directly via the underlying engine: by_vector path doesn't
            # need an embedding function.
            qvec = np.zeros(8, dtype=np.float32)
            qvec[0] = 1.0
            store._db.insert("only", qvec)
            r = store.similarity_search_by_vector([1.0] + [0.0] * 7, k=1)
            assert len(r) == 1 and r[0].id == "only"
        finally:
            store._db.close()


def test_text_search_without_embedding_raises():
    from tqdb.vectorstore import TurboQuantVectorStore

    with _tempdir() as path:
        store = TurboQuantVectorStore.open(
            path, embedding=None, dimension=8, metric="cosine"
        )
        try:
            with pytest.raises(ValueError, match="embedding"):
                store.similarity_search("query", k=1)
        finally:
            store._db.close()
