"""
Step 16 - Monitoring: /metrics content, outcome counters, and the review queue.

Prometheus metrics are global counters that only go up, so each test compares the value
BEFORE and AFTER its request instead of expecting an absolute number.
"""
import json

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import api.main as api_main
from generate import NO_ANSWER
from test_api import QUESTION, FakePipeline


def value(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


def make_client(chunks, answer, scores=(0.9, 0.5, 0.2, 0.1, 0.05)):
    return TestClient(api_main.create_app(FakePipeline(chunks, answer, scores)))


def test_metrics_endpoint_lists_our_metrics(chunks, monkeypatch):
    monkeypatch.setattr(api_main, "RERANKER", "large")
    with make_client(chunks, "Antwort [1].") as c:
        c.post("/ask", json=QUESTION)
        r = c.get("/metrics")
    assert r.status_code == 200
    for name in ("copilot_http_requests_total", "copilot_ask_outcomes_total",
                 "copilot_stage_seconds_bucket", "copilot_retrieval_confidence_bucket",
                 "copilot_pipeline_info"):
        assert name in r.text


def test_outcomes_are_counted(chunks, monkeypatch):
    monkeypatch.setattr(api_main, "RERANKER", "large")
    cases = {"answered": "Antwort [1].", "abstained": NO_ANSWER, "guardrail": "Antwort ohne Quelle."}
    for outcome, answer in cases.items():
        before = value("copilot_ask_outcomes_total", outcome=outcome)
        with make_client(chunks, answer) as c:
            c.post("/ask", json=QUESTION)
        assert value("copilot_ask_outcomes_total", outcome=outcome) == before + 1


def test_rejected_requests_are_counted_too(chunks):
    before = value("copilot_http_requests_total", path="/ask", status="422")
    with make_client(chunks, "Antwort [1].") as c:
        c.post("/ask", json={"question": ""})
    assert value("copilot_http_requests_total", path="/ask", status="422") == before + 1


def test_unknown_paths_share_one_label(chunks):
    before = value("copilot_http_requests_total", path="other", status="404")
    with make_client(chunks, "Antwort [1].") as c:
        c.get("/wp-admin")
        c.get("/random-scanner-url-12345")
    assert value("copilot_http_requests_total", path="other", status="404") == before + 2


def test_review_queue_gets_bad_answers_only(chunks, monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "RERANKER", "large")
    queue = tmp_path / "review" / "queue.jsonl"
    monkeypatch.setattr(api_main, "REVIEW_LOG", str(queue))
    with make_client(chunks, "Antwort [1].") as c:                              # good answer
        c.post("/ask", json=QUESTION)
    assert not queue.exists()
    with make_client(chunks, NO_ANSWER) as c:                                   # abstention
        c.post("/ask", json=QUESTION)
    with make_client(chunks, "Antwort [1].", scores=(0.05, 0.01, 0.01, 0.01, 0.01)) as c:  # low conf.
        c.post("/ask", json=QUESTION)
    lines = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
    assert [x["outcome"] for x in lines] == ["abstained", "answered"]
    assert lines[0]["question"] == QUESTION["question"]
    assert lines[0]["top_passages"][0] == f"{chunks[0]['doc']}#{chunks[0]['page']}"


def test_review_queue_off_by_default(chunks, monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "REVIEW_LOG", None)
    monkeypatch.chdir(tmp_path)
    with make_client(chunks, NO_ANSWER) as c:
        assert c.post("/ask", json=QUESTION).status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_unwritable_review_queue_does_not_break_the_api(chunks, monkeypatch, tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")                                     # a FILE where a folder should be
    monkeypatch.setattr(api_main, "REVIEW_LOG", str(blocker / "queue.jsonl"))
    with make_client(chunks, NO_ANSWER) as c:
        assert c.post("/ask", json=QUESTION).status_code == 200
