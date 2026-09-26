"""
Step 15 - The Qdrant retriever must give the SAME results as the NumPy retriever it replaced.
Uses Qdrant's local in-memory mode (":memory:"), so no Qdrant container is needed.
"""
import numpy as np
import pytest

from bm25 import BM25, Tokenizer
from hybrid import Hybrid
from pipeline import with_header
from qdrant_store import QdrantDense, fingerprint

QUESTIONS = ["Wer genehmigt den PACS-Zugang?", "Der PACS ist nich ereichbar", "VPN Zugang beantragen"]


class NumpyDense:
    """What DenseRetriever (Step 5) computes: normalised passage matrix @ query vector."""
    def __init__(self, model, texts):
        self.model = model
        self.matrix = model.encode(["passage: " + t for t in texts])

    def scores(self, query):
        return self.matrix @ self.model.encode("query: " + query)


@pytest.fixture
def texts(chunks):
    return [with_header(c) for c in chunks]


@pytest.fixture
def qdrant(fake_embedder, texts, chunks):
    return QdrantDense(fake_embedder, "fake-e5", texts, chunks, ":memory:")


@pytest.mark.parametrize("question", QUESTIONS)
def test_same_scores_as_numpy(qdrant, fake_embedder, texts, question):
    numpy_scores = NumpyDense(fake_embedder, texts).scores(question)
    assert qdrant.scores(question) == pytest.approx(numpy_scores, abs=1e-5)


@pytest.mark.parametrize("question", QUESTIONS)
def test_same_hybrid_candidates(qdrant, fake_embedder, texts, question):
    bm25 = BM25(texts, Tokenizer(stopwords=True, stemming=True))
    a = Hybrid(bm25, NumpyDense(fake_embedder, texts), "weighted", 0.7).search(question, k=20)
    b = Hybrid(bm25, qdrant, "weighted", 0.7).search(question, k=20)
    assert set(a) == set(b)       # same 20 candidates for the reranker (ties may swap order)


def test_existing_collection_is_not_encoded_again(qdrant, fake_embedder):
    calls = fake_embedder.calls
    qdrant._ensure_index()                     # what happens on the next container start
    assert fake_embedder.calls == calls


def test_fingerprint_changes_when_data_changes(texts):
    assert fingerprint(texts, "e5", "passage: ") == fingerprint(list(texts), "e5", "passage: ")
    assert fingerprint(texts, "e5", "passage: ") != fingerprint(texts[:-1], "e5", "passage: ")
    assert fingerprint(texts, "e5", "passage: ") != fingerprint(texts, "other-model", "passage: ")


def test_top_n_fills_missing_chunks_with_floor(qdrant, texts):
    s = qdrant.scores("PACS Zugang", top_n=10)
    assert len(s) == len(texts)
    assert np.sum(s > s.min()) <= 10
