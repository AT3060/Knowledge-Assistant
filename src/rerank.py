"""
Step 7 - Reranking with a cross-encoder.

Bi-encoder (Step 5):   question -> vector      chunk -> vector      score = cosine
                       Question and chunk never "see" each other. Chunk vectors can be
                       computed in advance -> fast, but the comparison is coarse.

Cross-encoder (here):  [question + chunk] -> one transformer pass -> relevance score
                       Every word of the question can attend to every word of the chunk,
                       so it can tell "ich bin gesperrt" (account) from "Sitzung wird gesperrt".
                       Nothing can be precomputed: one full model pass PER (question, chunk) pair.

That is why rerankers are used in two stages ("retrieve, then rerank"):
    1. a fast retriever picks a candidate pool (e.g. the top 20 chunks out of thousands)
    2. the slow, accurate cross-encoder re-sorts only that pool
The pool size is the trade-off: bigger pool = better chance the right chunk is in it,
but latency grows linearly with the pool size.
"""
import numpy as np

SMALL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"   # 118M params, multilingual (incl. German)
LARGE = "BAAI/bge-reranker-v2-m3"                     # 568M params, stronger, much slower


class Reranker:
    def __init__(self, model_name=SMALL):
        from sentence_transformers import CrossEncoder
        self.name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, texts):
        """Re-sort the candidate chunk indices by cross-encoder score, best first."""
        pairs = [(query, texts[i]) for i in candidates]
        scores = np.asarray(self.model.predict(pairs, batch_size=32), dtype=float)
        order = np.argsort(-scores, kind="stable")
        return [candidates[i] for i in order], scores[order]


class RetrieveAndRerank:
    def __init__(self, retriever, reranker, texts, pool=20):
        self.retriever, self.reranker, self.texts, self.pool = retriever, reranker, texts, pool

    def search(self, query, k=50):
        candidates = self.retriever.search(query, k=k)
        reranked, _ = self.reranker.rerank(query, candidates[:self.pool], self.texts)
        # Keep the retriever's order for everything below the pool, so the list stays k long.
        return reranked + candidates[self.pool:]
