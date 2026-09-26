"""
Step 14 - Dense retrieval with a vector database (Qdrant) instead of a NumPy matrix.

Until now (Step 5) the chunk vectors lived in RAM as a NumPy matrix, and every start of the
app encoded all chunks again. That is fine for 136 chunks, but not for a real company:
    - 1 million chunks x 384 floats = ~1.5 GB, recomputed on every restart
    - several API containers would each hold their own copy
    - no filtering (e.g. "only PACS documents"), no updates without restarting

A vector database stores the vectors once, persistently, and answers "which vectors are most
similar to this one?" for any number of clients. Qdrant runs as its own container (Step 14
docker-compose.yml); the API talks to it over HTTP.

This class has the SAME interface as DenseRetriever (scores / search), so Hybrid, the reranker
and the whole pipeline work unchanged. The query is still encoded by our e5 model - Qdrant only
stores and searches vectors, it does not compute embeddings.
"""
import hashlib
import logging
import time

import numpy as np

log = logging.getLogger("knowledge-api")


def fingerprint(texts, model_name, passage_prefix):
    """Short hash of everything that determines the vectors. If chunks, model or prefix change,
    the hash changes -> a new collection is built instead of silently using stale vectors."""
    h = hashlib.sha256(model_name.encode())
    h.update(passage_prefix.encode())
    for t in texts:
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:10]


class QdrantDense:
    def __init__(self, model, model_name, texts, chunks, url, collection_prefix="handbook_chunks",
                 query_prefix="query: ", passage_prefix="passage: ", wait_seconds=60):
        from qdrant_client import QdrantClient

        self.model, self.texts, self.chunks = model, texts, chunks
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        self.collection = f"{collection_prefix}_{fingerprint(texts, model_name, passage_prefix)}"
        self.client = QdrantClient(":memory:") if url == ":memory:" else QdrantClient(url=url)
        self._wait_until_ready(wait_seconds)
        self._ensure_index()

    def _wait_until_ready(self, wait_seconds):
        """docker compose starts both containers at the same time, and Qdrant may need a few
        seconds. A robust client retries instead of crashing on the first failed connection."""
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                self.client.get_collections()
                return
            except Exception as e:  # noqa: BLE001  (connection refused, timeout ...)
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Qdrant not reachable after {wait_seconds} s") from e
                log.info("waiting for Qdrant ...")
                time.sleep(2)

    def _ensure_index(self):
        """Build the collection once. On later starts it already exists -> nothing is encoded."""
        from qdrant_client import models

        n = len(self.texts)
        if self.client.collection_exists(self.collection):
            count = self.client.count(self.collection, exact=True).count
            if count == n:
                log.info("Qdrant collection %s already has %d vectors - skipping indexing",
                         self.collection, n)
                return
            self.client.delete_collection(self.collection)     # incomplete (e.g. crash while indexing)

        start = time.perf_counter()
        vectors = self.model.encode([self.passage_prefix + t for t in self.texts],
                                    normalize_embeddings=True, batch_size=32)
        vectors = np.asarray(vectors, dtype=np.float32)
        # COSINE distance: Qdrant returns the cosine similarity as the score, the same number
        # our NumPy version computed as the dot product of normalised vectors (Step 5).
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE))
        # Point id = chunk index, so a hit maps straight back to self.chunks[id].
        # The payload is stored for humans: you can browse it in the Qdrant dashboard.
        self.client.upsert(self.collection, points=[
            models.PointStruct(id=i, vector=v.tolist(),
                               payload={"doc": c["doc"], "page": c["page"], "title": c["title"],
                                        "text": c["text"]})
            for i, (v, c) in enumerate(zip(vectors, self.chunks))])
        log.info("indexed %d chunks into Qdrant collection %s in %.1f s", n, self.collection,
                 time.perf_counter() - start)

    def scores(self, query, top_n=None):
        """One similarity per chunk, like DenseRetriever.scores, so the weighted hybrid fusion
        (min-max over all chunks) gives exactly the same result as before.

        top_n=None asks Qdrant for ALL chunks - fine for 136 chunks. With millions of chunks you
        would ask for e.g. the top 100 only; chunks Qdrant did not return get the lowest score seen.
        """
        q = self.model.encode(self.query_prefix + query, normalize_embeddings=True)
        q = np.asarray(q, dtype=np.float32)
        hits = self.client.query_points(self.collection, query=q.tolist(),
                                        limit=top_n or len(self.texts)).points
        s = np.full(len(self.texts), min((h.score for h in hits), default=0.0), dtype=float)
        for h in hits:
            s[h.id] = h.score
        return s

    def search(self, query, k=50):
        s = self.scores(query)
        return list(np.argsort(-s)[:k])
