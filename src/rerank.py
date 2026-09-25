"""
Step 7 - Reranking with a cross-encoder.

Bi-encoder (Step 5):   question -> vector      chunk -> vector      score = cosine
                       Question and chunk never "see" each other. Chunk vectors can be
                       computed in advance -> fast, but the comparison is coarse.

Cross-encoder (here):  [question + chunk] -> one transformer pass -> relevance score
                       Every word of the question can attend to every word of the chunk.
                       Nothing can be precomputed: one full model pass PER (question, chunk) pair.

That is why rerankers are used in two stages ("retrieve, then rerank"):
    1. a fast retriever picks a candidate pool (e.g. the top 20 chunks)
    2. the slow, accurate cross-encoder re-sorts only that pool
"""
import numpy as np

# Candidate rerankers, all multilingual (German included).  name -> (Hugging Face id, needs remote code)
MODELS = {
    "small": ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", False),     # ~118M params
    "jina":  ("jinaai/jina-reranker-v2-base-multilingual", True),       # ~278M params
    "gte":   ("Alibaba-NLP/gte-multilingual-reranker-base", True),      # ~306M params
    "large": ("BAAI/bge-reranker-v2-m3", False),                        # ~568M params
}
SMALL, LARGE = MODELS["small"][0], MODELS["large"][0]


class Reranker:
    def __init__(self, model_name=SMALL, trust_remote_code=False, max_length=512):
        # trust_remote_code: some models ship their own Python code on the Hub. Only enable it
        # for publishers you trust, because that code runs on your machine.
        from sentence_transformers import CrossEncoder
        self.name = model_name
        self.model = CrossEncoder(model_name, trust_remote_code=trust_remote_code,
                                  max_length=max_length)

    @classmethod
    def from_key(cls, key):
        model_id, remote = MODELS[key]
        return cls(model_id, trust_remote_code=remote)

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

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
