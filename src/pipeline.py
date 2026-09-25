"""
Step 9 - The complete RAG pipeline, used by the evaluation AND later by the demo app.

question -> BM25 + e5 (weighted hybrid, α=0.7) -> top 20 chunks -> bge-reranker-v2-m3
         -> top 5 passages -> LLM -> German answer with [n] -> citations (document + page)
"""
import time

from bm25 import BM25, Tokenizer
from dense import DenseRetriever
from generate import NO_ANSWER, build_messages, is_abstention, parse_citations
from hybrid import Hybrid
from rerank import Reranker

E5 = "intfloat/multilingual-e5-small"


def with_header(chunk):
    """Same 'contextual header' text format that the retrievers were evaluated with."""
    doc = chunk["doc"].removesuffix(".pdf").replace("_", " ")
    return f"{doc} – {chunk['title']}: {chunk['text']}"


class RAGPipeline:
    def __init__(self, chunks, generator=None, reranker_key="large", alpha=0.7, pool=20, top_k=5):
        self.chunks, self.generator, self.pool, self.top_k = chunks, generator, pool, top_k
        self.texts = [with_header(c) for c in chunks]
        self.hybrid = Hybrid(BM25(self.texts, Tokenizer(stopwords=True, stemming=True)),
                             DenseRetriever(self.texts, E5), "weighted", alpha)
        self.reranker = Reranker.from_key(reranker_key)

    def retrieve(self, question):
        """Top-k chunks after reranking, with their reranker scores."""
        candidates = self.hybrid.search(question, k=self.pool)
        ranked, scores = self.reranker.rerank(question, candidates, self.texts)
        return [(self.chunks[i], float(s)) for i, s in zip(ranked[:self.top_k], scores[:self.top_k])]

    def answer(self, question):
        t0 = time.perf_counter()
        hits = self.retrieve(question)
        t1 = time.perf_counter()
        passages = [c for c, _ in hits]
        answer = self.generator.generate(build_messages(question, passages)) if self.generator else ""
        t2 = time.perf_counter()

        abstained = is_abstention(answer)
        valid, invalid = parse_citations(answer, len(passages))
        citations = [] if abstained else [
            {"n": n, "doc": passages[n - 1]["doc"], "page": passages[n - 1]["page"],
             "page_id": passages[n - 1]["page_id"]} for n in valid]
        return {
            "answer": NO_ANSWER if abstained else answer,
            "raw_answer": answer,
            "abstained": abstained,
            "citations": citations,
            "invalid_citations": invalid,
            "passages": [{"page_id": c["page_id"], "score": s} for c, s in hits],
            "top_score": hits[0][1] if hits else float("nan"),
            "t_retrieve": t1 - t0,
            "t_generate": t2 - t1,
        }


def format_sources(result):
    """'📄 PACS_Handbuch.pdf — Seite 4' lines, one per cited page, for the user."""
    seen, lines = set(), []
    for c in result["citations"]:
        if c["page_id"] not in seen:
            seen.add(c["page_id"])
            lines.append(f"📄 {c['doc']} — Seite {c['page']}")
    return lines
