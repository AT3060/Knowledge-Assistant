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
                 query_prefix="query: ", passage_prefix="passage: ", device=None):
        # Imported here so the rest of the project works without installing torch.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = query_prefix
        start = time.perf_counter()
        self.matrix = self.model.encode([passage_prefix + t for t in texts],
                                        normalize_embeddings=True, batch_size=32)
        self.index_seconds = time.perf_counter() - start
        print(f"  encoded {len(texts)} chunks with {model_name} "
              f"(prefixes: {query_prefix!r}/{passage_prefix!r}) -> matrix {self.matrix.shape}, "
              f"{self.index_seconds:.1f} s")

    def scores(self, query):
        q = self.model.encode(self.query_prefix + query, normalize_embeddings=True)
        return self.matrix @ q                        # cosine similarity with every chunk

    def search(self, query, k=50):
        s = self.scores(query)
        return list(np.argsort(-s)[:k])               # indices of the k most similar chunks
