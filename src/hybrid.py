"""
Step 6 - Hybrid retrieval: combine BM25 and dense rankings.

Problem: the two score types live on different scales.
    BM25 scores are unbounded, e.g. 0 ... 25
    cosine similarities are squeezed together, e.g. 0.78 ... 0.88
Adding them directly would let BM25 dominate. Two standard solutions:

1) Reciprocal Rank Fusion (RRF) - ignore the scores, use only the RANKS:
       rrf(d) = Σ_retrievers  1 / (k + rank(d))          (k = 60 by convention)
   A chunk ranked 1st by one retriever gets 1/61, ranked 10th gets 1/70.
   Being near the top of BOTH lists beats being 1st in only one.

2) Weighted fusion - rescale each score vector to [0, 1] (min-max), then mix:
       fused(d) = α · dense(d) + (1 - α) · bm25(d)
   α = 0 is pure BM25, α = 1 is pure dense. α must be chosen - see evaluate_retrieval.py.
"""
import numpy as np


def ranks_from_scores(scores):
    """rank[i] = position of chunk i when sorted by score, best = 1."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def rrf(bm25_scores, dense_scores, k=60):
    return sum(1.0 / (k + ranks_from_scores(s)) for s in (bm25_scores, dense_scores))


def minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def weighted(bm25_scores, dense_scores, alpha):
    return alpha * minmax(dense_scores) + (1 - alpha) * minmax(bm25_scores)


class Hybrid:
    def __init__(self, bm25, dense, method="rrf", alpha=0.5, rrf_k=60):
        self.bm25, self.dense = bm25, dense
        self.method, self.alpha, self.rrf_k = method, alpha, rrf_k

    def score_pair(self, query):
        """Both raw score vectors for one query (one entry per chunk)."""
        return (np.asarray(self.bm25.scores(query), dtype=float),
                np.asarray(self.dense.scores(query), dtype=float))

    def fuse(self, b, d):
        return rrf(b, d, self.rrf_k) if self.method == "rrf" else weighted(b, d, self.alpha)

    def search(self, query, k=50):
        fused = self.fuse(*self.score_pair(query))
        return list(np.argsort(-fused, kind="stable")[:k])
