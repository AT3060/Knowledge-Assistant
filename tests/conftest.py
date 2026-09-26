"""
Step 15 - Shared test setup (pytest loads this file automatically before any test).

Fixtures are reusable test ingredients: a test function that has a parameter named `chunks`
receives the value of the `chunks` fixture below. scope="session" = computed once per test run.

None of the tests download a model. Where a model is needed we use a small FAKE:
the tests check OUR logic (fusion, citations, guardrail, API, Qdrant), not the quality of e5
or Qwen - that is what the evaluation in eval/ is for. This keeps the whole suite fast (seconds)
and runnable on a free GitHub machine without a GPU.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src", ROOT / "eval"):
    sys.path.insert(0, str(p))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="session")
def chunks():
    return load_jsonl(ROOT / "data" / "chunks.jsonl")


@pytest.fixture(scope="session")
def questions():
    return load_jsonl(ROOT / "data" / "questions.jsonl")


class FakeEmbedder:
    """Stands in for the e5 SentenceTransformer: same encode() signature, but the 'embedding'
    is just a normalised bag of hashed words. Deterministic, instant, no download."""
    dim = 64

    def __init__(self):
        self.calls = 0

    def _one(self, text):
        v = np.zeros(self.dim)
        for w in text.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        return v / (np.linalg.norm(v) or 1.0)

    def encode(self, x, normalize_embeddings=True, batch_size=32):
        self.calls += 1
        if isinstance(x, list):
            return np.array([self._one(t) for t in x], dtype=np.float32)
        return self._one(x).astype(np.float32)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
