"""
Step 12 - A REST API for the RAG pipeline (FastAPI).

The Gradio Space is an interface for HUMANS. An API is an interface for OTHER PROGRAMS:
an intranet page, a Teams bot, a ticket system ... sends a question as JSON over HTTP and
gets the answer + citations back as JSON. Same pipeline (src/pipeline.py), new front door.

    POST /ask        question -> grounded German answer with citations (retrieval + LLM)
    POST /retrieve   question -> top passages only (no LLM: fast, good for debugging)
    GET  /health     is the service up, which models are loaded?
    GET  /docs       interactive documentation, generated automatically by FastAPI

Step 12b adds two safety checks found by testing the API on a laptop:
    - citation guardrail: an answer without a valid citation is replaced by the "no information" answer
    - confidence on one 0-1 scale for every reranker, with a high/medium/low label only where calibrated

Step 14: if QDRANT_URL is set, the dense vectors live in a Qdrant vector database (src/qdrant_store.py)
instead of a NumPy matrix in RAM. Without QDRANT_URL everything works exactly as in Step 12.

Run from the project root (laptop, CPU, small models):
    LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct RERANKER=small python -m uvicorn api.main:app --port 8000
Retrieval only, no LLM loaded at all (starts much faster):
    LLM_MODEL=none RERANKER=small python -m uvicorn api.main:app --port 8000
Then open http://127.0.0.1:8000/docs in the browser.
"""
import json
import logging
import math
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generate import NO_ANSWER  # noqa: E402
from pipeline import E5, RAGPipeline, format_sources  # noqa: E402

# Configuration comes from environment variables, not from the code. The same code can then run
# with a 0.5B model on a laptop and with the 7B model on a GPU server - only the settings change.
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")   # "none" = retrieval only
RERANKER = os.getenv("RERANKER", "small")                          # key from src/rerank.py MODELS
CHUNKS_PATH = Path(os.getenv("CHUNKS_PATH", str(ROOT / "data" / "chunks.jsonl")))
QDRANT_URL = os.getenv("QDRANT_URL")          # e.g. http://qdrant:6333 ; unset = in-memory NumPy (Step 5)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("knowledge-api")
logging.getLogger("httpx").setLevel(logging.WARNING)   # hide one log line per Hugging Face / Qdrant request


# ---------------------------------------------------------------- request / response schemas
# Pydantic models describe the JSON shape. FastAPI uses them to
#   1. validate incoming JSON (wrong type, too short ... -> automatic 422 error, our code never runs)
#   2. build the response JSON (only the declared fields are sent)
#   3. generate the /docs page
QuestionText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)]


class QuestionIn(BaseModel):
    question: QuestionText = Field(examples=["Wer genehmigt den PACS-Zugang?"])


class Citation(BaseModel):
    n: int = Field(description="Passage number [n] used in the answer")
    doc: str
    page: int


class Passage(BaseModel):
    doc: str
    page: int
    title: str
    text: str
    score: float = Field(description="Reranker relevance score")


class RetrieveOut(BaseModel):
    request_id: str
    confidence: float = Field(description="Top reranker score, always on a 0-1 scale")
    confidence_level: str = Field(description="high / medium / low, or 'uncalibrated'")
    passages: list[Passage]
    t_retrieve_ms: float


class AskOut(BaseModel):
    request_id: str
    answer: str
    abstained: bool = Field(description="True if no grounded answer is given")
    grounded: bool = Field(description="True if the answer cites at least one retrieved passage")
    guardrail: str | None = Field(None, description="Why the model's answer was replaced, if it was")
    citations: list[Citation]
    sources: list[str]
    confidence: float = Field(description="Top reranker score, always on a 0-1 scale")
    confidence_level: str = Field(description="high / medium / low, or 'uncalibrated'")
    passages: list[Passage]
    t_retrieve_ms: float
    t_generate_ms: float


# ---------------------------------------------------------------- Step 12b: confidence on one scale
# Rerankers do not all output the same kind of number:
#   bge-reranker-v2-m3 ("large")        -> already a probability in [0, 1]
#   mmarco-mMiniLMv2 ("small")          -> a raw logit, any real number (we saw 8.9 and -2.9)
# A logit x becomes a probability with the sigmoid  σ(x) = 1 / (1 + e^(-x)).
SCORE_IS_LOGIT = {"small": True, "large": False}
# The high/medium/low thresholds were measured in Step 10 for the LARGE reranker only.
# Calibration belongs to one model: for any other reranker we say so instead of guessing.
CALIBRATED_FOR = "large"


def to_probability(score):
    if SCORE_IS_LOGIT.get(RERANKER, False):
        return 1.0 / (1.0 + math.exp(-score))
    return score


def confidence_level(prob):
    """Thresholds from the Step 10 analysis: above 0.5 about 91% of answers were correct."""
    if RERANKER != CALIBRATED_FOR:
        return "uncalibrated"
    return "high" if prob >= 0.5 else "medium" if prob >= 0.1 else "low"


# ---------------------------------------------------------------- load models once
def load_pipeline():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    generator = None
    if LLM_MODEL.lower() != "none":
        import torch
        from generate import Generator
        use_gpu = torch.cuda.is_available()
        # On CPU use float32: float16 on CPU is slow or unsupported for many operations.
        generator = Generator(LLM_MODEL, device_map="auto" if use_gpu else None,
                              dtype=None if use_gpu else torch.float32)
    if not QDRANT_URL:
        return RAGPipeline(chunks, generator, reranker_key=RERANKER)

    # Step 14: lazy_index=True -> the NumPy retriever loads the e5 model but encodes nothing.
    # Then we swap it for the Qdrant retriever, which reuses the same e5 model for the queries.
    from qdrant_store import QdrantDense
    rag = RAGPipeline(chunks, generator, reranker_key=RERANKER, lazy_index=True)
    rag.hybrid.dense = QdrantDense(rag.hybrid.dense.model, E5, rag.texts, chunks, QDRANT_URL)
    return rag


def create_app(rag=None):
    """rag=None: load the real models at startup. Tests (Step 15) pass a small fake pipeline instead."""

    @asynccontextmanager
    async def lifespan(app):
        # Everything before 'yield' runs ONCE when the server starts, everything after it on shutdown.
        # Loading models takes seconds to minutes, so it must not happen inside a request.
        start = time.perf_counter()
        app.state.rag = rag if rag is not None else load_pipeline()
        # PyTorch models are not guaranteed to be safe when two requests use them at the same time.
        # The lock lets requests queue up and run the pipeline one after another.
        app.state.lock = threading.Lock()
        log.info("pipeline ready in %.1f s (llm=%s, reranker=%s)", time.perf_counter() - start,
                 LLM_MODEL, RERANKER)
        yield

    app = FastAPI(title="Deutscher Enterprise Knowledge Copilot API", version="0.1.0",
                  description="German question answering over hospital IT handbooks, with citations.",
                  lifespan=lifespan)

    # Endpoints are plain 'def', not 'async def': the pipeline is blocking CPU/GPU work.
    # FastAPI runs plain 'def' endpoints in a thread pool, so the server stays responsive meanwhile.

    @app.get("/health")
    def health(request: Request):
        rag_ = request.app.state.rag
        return {"status": "ok",
                "llm": LLM_MODEL if rag_.generator is not None else None,
                "reranker": RERANKER,
                "vector_store": "qdrant" if QDRANT_URL else "in-memory",
                "n_chunks": len(rag_.chunks)}

    @app.post("/retrieve", response_model=RetrieveOut)
    def retrieve(body: QuestionIn, request: Request):
        request_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        with request.app.state.lock:
            hits = request.app.state.rag.retrieve(body.question)
        t_ms = (time.perf_counter() - t0) * 1000
        top = to_probability(hits[0][1]) if hits else 0.0
        log.info("retrieve id=%s top_score=%.3f t=%.0fms", request_id, top, t_ms)
        return RetrieveOut(
            request_id=request_id, confidence=top, confidence_level=confidence_level(top),
            passages=[Passage(doc=c["doc"], page=c["page"], title=c["title"], text=c["text"],
                              score=to_probability(s)) for c, s in hits],
            t_retrieve_ms=round(t_ms, 1))

    @app.post("/ask", response_model=AskOut)
    def ask(body: QuestionIn, request: Request):
        rag_ = request.app.state.rag
        if rag_.generator is None:
            # 503 = "Service Unavailable": the request is fine, but this server can't do it right now.
            raise HTTPException(status_code=503,
                                detail="No LLM loaded (LLM_MODEL=none). Use /retrieve instead.")
        request_id = uuid.uuid4().hex[:12]
        with request.app.state.lock:
            result = rag_.answer(body.question)
        top = to_probability(result["top_score"])

        # Step 12b: citation guardrail. The prompt TELLS the model to cite every statement, but a
        # model can ignore instructions (the 0.5B model did). So we CHECK it: an answer that is not
        # an abstention and cites no retrieved passage is treated as ungrounded and is not shown.
        guardrail = None
        if not result["abstained"] and not result["citations"]:
            guardrail = "no_valid_citation"
            log.warning("guardrail id=%s reason=%s invalid_citations=%s raw_answer=%r", request_id,
                        guardrail, result["invalid_citations"], result["raw_answer"])
            result = {**result, "answer": NO_ANSWER, "abstained": True}
        grounded = bool(result["citations"]) and guardrail is None

        # One log line per request: the starting point for monitoring (Step 16).
        log.info("ask id=%s abstained=%s grounded=%s guardrail=%s n_citations=%d top_score=%.3f "
                 "t_retrieve=%.0fms t_generate=%.0fms", request_id, result["abstained"], grounded,
                 guardrail, len(result["citations"]), top,
                 result["t_retrieve"] * 1000, result["t_generate"] * 1000)
        return AskOut(
            request_id=request_id,
            answer=result["answer"],
            abstained=result["abstained"],
            grounded=grounded,
            guardrail=guardrail,
            citations=[Citation(n=c["n"], doc=c["doc"], page=c["page"]) for c in result["citations"]],
            sources=format_sources(result),
            confidence=top,
            confidence_level=confidence_level(top),
            passages=[Passage(doc=p["doc"], page=p["page"], title=p["title"], text=p["text"],
                              score=to_probability(p["score"])) for p in result["passages"]],
            t_retrieve_ms=round(result["t_retrieve"] * 1000, 1),
            t_generate_ms=round(result["t_generate"] * 1000, 1))

    return app


# 'uvicorn api.main:app' looks for this variable.
app = create_app()
