"""
Step 16 - Monitoring and observability.

Logging tells you what happened in ONE request. Monitoring tells you how the system behaves
over time: Is it getting slower? How often does it say "keine Information"? Which questions
fail? This module produces three signals:

1. METRICS (numbers over time)
   Counters and histograms kept in memory and published at GET /metrics in the Prometheus
   text format. The Prometheus container reads ("scrapes") that page every 15 s and stores
   the history; Grafana draws dashboards from it.
       Counter   - only goes up: requests, answers, abstentions ...
       Histogram - counts observations per bucket (e.g. latency <= 1 s, <= 2 s, ...), from which
                   Prometheus computes percentiles like p95 ("95 % of requests were faster than X").

2. STRUCTURED LOGS
   One JSON line per request instead of free text. Log tools (Loki, Elasticsearch, Azure Monitor)
   can then filter and aggregate: "all requests with outcome=guardrail yesterday".
   No question text here - logs are kept long and read by many people.

3. REVIEW QUEUE
   Questions the system could not answer well (abstained, blocked by the guardrail, or very low
   retrieval confidence) are appended to a JSONL file for a human to review. This is how the
   "Fileserver" vs "Netzlaufwerke" gap would be found in practice.
   Off by default: questions can contain personal data (DSGVO), so storing them is an explicit
   decision (REVIEW_LOG=path), with access control and a retention period in a real deployment.

Rule for all three: monitoring must NEVER break the product. If writing fails, log and continue.
"""
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from prometheus_client import Counter, Histogram, Info

# ---------------------------------------------------------------- 1. metrics
HTTP_REQUESTS = Counter("copilot_http_requests", "HTTP requests by path and status code",
                        ["path", "status"])
HTTP_LATENCY = Histogram("copilot_http_request_duration_seconds", "Time from request to response",
                         ["path"], buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64))
STAGE_SECONDS = Histogram("copilot_stage_seconds", "Time per pipeline stage (retrieve / generate)",
                          ["stage"], buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32))
CONFIDENCE = Histogram("copilot_retrieval_confidence", "Top reranker score on the 0-1 scale",
                       buckets=(0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0))
ASK_OUTCOMES = Counter("copilot_ask_outcomes", "What /ask returned: answered / abstained / guardrail",
                       ["outcome"])
PIPELINE = Info("copilot_pipeline", "Models and vector store this instance runs with")

# Label values must come from a small, fixed set. If we used the raw URL as a label, a scanner
# requesting 10 000 random URLs would create 10 000 time series ("cardinality explosion").
KNOWN_PATHS = {"/ask", "/retrieve", "/health", "/metrics"}


async def metrics_middleware(request, call_next):
    """Runs around EVERY request, also those FastAPI rejects before our code runs (422)."""
    start = time.perf_counter()
    path = request.url.path if request.url.path in KNOWN_PATHS else "other"
    try:
        response = await call_next(request)
    except Exception:
        HTTP_REQUESTS.labels(path, "500").inc()
        raise
    HTTP_REQUESTS.labels(path, str(response.status_code)).inc()
    HTTP_LATENCY.labels(path).observe(time.perf_counter() - start)
    return response


# ---------------------------------------------------------------- 2. structured logs
event_log = logging.getLogger("knowledge-api.events")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))      # the line IS the JSON, no prefix
event_log.addHandler(_handler)
event_log.setLevel(logging.INFO)
event_log.propagate = False


def log_event(**fields):
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **fields}
    event_log.info(json.dumps(record, ensure_ascii=False))


# ---------------------------------------------------------------- 3. review queue
class ReviewQueue:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()                # two requests must not interleave lines

    @property
    def enabled(self):
        return self.path is not None

    def add(self, record):
        if not self.enabled:
            return
        line = json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record},
                          ensure_ascii=False)
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError as e:
            logging.getLogger("knowledge-api").warning("review queue not writable (%s): %s", self.path, e)
