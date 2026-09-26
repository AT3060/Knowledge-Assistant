"""
Step 15 - API behaviour (api/main.py) with a fake pipeline: no models, answers under our control.

FastAPI's TestClient sends real HTTP requests to the app in memory - no server, no port.
`with TestClient(app)` also runs the lifespan (startup), exactly like uvicorn would.
"""
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from generate import NO_ANSWER
from pipeline import RAGPipeline

QUESTION = {"question": "Der PACS ist nich ereichbar"}


class FakeGenerator:
    def __init__(self, answer):
        self.answer = answer

    def generate(self, messages):
        return self.answer


class FakePipeline(RAGPipeline):
    """The real RAGPipeline.answer() logic (citation mapping, abstention, timing),
    but retrieve() returns the first 5 chunks with fixed scores instead of running models."""
    def __init__(self, chunks, answer=None, scores=(0.9, 0.5, 0.2, 0.1, 0.05)):
        self.chunks, self.top_k, self.scores = chunks, 5, scores
        self.generator = FakeGenerator(answer) if answer is not None else None

    def retrieve(self, question):
        return list(zip(self.chunks[:5], self.scores))


def client_for(chunks, answer=None, scores=(0.9, 0.5, 0.2, 0.1, 0.05)):
    return TestClient(api_main.create_app(FakePipeline(chunks, answer, scores)))


@pytest.fixture(autouse=True)
def large_reranker(monkeypatch):
    """Default for every test: behave as if RERANKER=large (the calibrated one)."""
    monkeypatch.setattr(api_main, "RERANKER", "large")


def test_health(chunks):
    with client_for(chunks, answer="x [1]") as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["n_chunks"] == len(chunks)


def test_grounded_answer_passes_through(chunks):
    with client_for(chunks, answer="Das PACS ist ausgefallen [1].") as c:
        body = c.post("/ask", json=QUESTION).json()
    assert body["grounded"] and not body["abstained"] and body["guardrail"] is None
    assert body["citations"][0]["doc"] == chunks[0]["doc"]
    assert body["sources"] == [f"📄 {chunks[0]['doc']} — Seite {chunks[0]['page']}"]


@pytest.mark.parametrize("answer", ["Nein, der PACS ist nicht verfügbar.",   # no citation (0.5B case)
                                    "Siehe Handbuch [9]."])                    # only invalid citation
def test_guardrail_blocks_ungrounded_answer(chunks, answer):
    with client_for(chunks, answer=answer) as c:
        body = c.post("/ask", json=QUESTION).json()
    assert body["answer"] == NO_ANSWER
    assert body["abstained"] and not body["grounded"]
    assert body["guardrail"] == "no_valid_citation"


def test_model_abstention_is_not_a_guardrail_event(chunks):
    with client_for(chunks, answer=NO_ANSWER) as c:
        body = c.post("/ask", json=QUESTION).json()
    assert body["abstained"] and body["guardrail"] is None


def test_ask_without_llm_returns_503(chunks):
    with client_for(chunks, answer=None) as c:
        assert c.post("/ask", json=QUESTION).status_code == 503
        assert c.post("/retrieve", json=QUESTION).status_code == 200     # retrieval still works


@pytest.mark.parametrize("payload", [{"question": "  "}, {"question": "x" * 501}, {}, {"q": "PACS"}])
def test_invalid_input_returns_422(chunks, payload):
    with client_for(chunks, answer="x [1]") as c:
        assert c.post("/ask", json=payload).status_code == 422


@pytest.mark.parametrize("score, level", [(0.9, "high"), (0.3, "medium"), (0.05, "low")])
def test_confidence_levels_for_large_reranker(chunks, score, level):
    with client_for(chunks, scores=(score, 0.01, 0.01, 0.01, 0.01)) as c:
        body = c.post("/retrieve", json=QUESTION).json()
    assert body["confidence"] == pytest.approx(score) and body["confidence_level"] == level


def test_small_reranker_logits_become_probabilities(chunks, monkeypatch):
    monkeypatch.setattr(api_main, "RERANKER", "small")
    with client_for(chunks, scores=(8.9, 0.0, -1.3, -2.3, -2.9)) as c:
        body = c.post("/retrieve", json=QUESTION).json()
    scores = [p["score"] for p in body["passages"]]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[1] == pytest.approx(0.5)                    # sigmoid(0) = 0.5
    assert body["confidence_level"] == "uncalibrated"         # thresholds belong to the large model
