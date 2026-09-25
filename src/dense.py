"""
Step 5 - Dense retrieval with embeddings.

An embedding model turns a text into a vector (for multilingual-e5-small: 384 numbers).
It was trained so that texts with similar MEANING get vectors that point in similar
directions, even if they share no words.

Similarity = cosine of the angle between two vectors:

    cos(q, d) = (q · d) / (|q| · |d|)

We normalise every vector to length 1 once (normalize_embeddings=True). Then |q| = |d| = 1,
and the cosine is simply the dot product q · d. Scoring all chunks is one matrix-vector
product:   scores = D @ q,  where D is the (n_chunks x 384) matrix of chunk vectors.
"""
import time

import numpy as np


class DenseRetriever:
    def __init__(self, texts, model_name="intfloat/multilingual-e5-small",
                 query_prefix="query: ", passage_prefix="passage: ", device=None, lazy=False):
        # Imported here so the rest of the project works without installing torch.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = query_prefix
        self.passages = [passage_prefix + t for t in texts]
        self.model_name = model_name
        self.matrix = None
        if not lazy:
            self.build_index()

    def build_index(self):
        """Encode all chunks. With lazy=True this happens on the first search instead of at startup
        (on ZeroGPU, models must not compute anything at startup - only inside @spaces.GPU)."""
        start = time.perf_counter()
        self.matrix = self.model.encode(self.passages, normalize_embeddings=True, batch_size=32)
        if hasattr(self.matrix, "cpu"):
            self.matrix = self.matrix.cpu().numpy()
        self.index_seconds = time.perf_counter() - start
        print(f"  encoded {len(self.passages)} chunks with {self.model_name} -> matrix "
              f"{self.matrix.shape}, {self.index_seconds:.1f} s")

    def scores(self, query):
        if self.matrix is None:
            self.build_index()
        q = self.model.encode(self.query_prefix + query, normalize_embeddings=True)
        if hasattr(q, "cpu"):
            q = q.cpu().numpy()
        return self.matrix @ q                        # cosine similarity with every chunk

    def search(self, query, k=50):
        s = self.scores(query)
        return list(np.argsort(-s)[:k])               # indices of the k most similar chunks
