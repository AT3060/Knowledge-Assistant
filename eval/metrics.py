"""
Step 1 — Retrieval metrics, written from scratch.

Vocabulary used throughout the project:
  corpus   : {doc_id: text}                 every searchable unit (here: a PDF page)
  queries  : {query_id: question}           the test questions
  qrels    : {query_id: {doc_id: grade}}    "query relevance judgements": which pages answer
                                            which question (grade 1 = relevant, 2 = highly relevant)
  run      : {query_id: [doc_id, ...]}      what a retriever returned, best first
"""
import math


def dedupe(ranked):
    """A retriever returns chunks; several chunks can come from the same page.
    We evaluate at page level, so keep only the first occurrence of each page."""
    seen, out = set(), []
    for d in ranked:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def recall_at_k(ranked, relevant, k):
    """Share of the relevant pages that appear in the top k."""
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(ranked, relevant, k=10):
    """1 / position of the FIRST relevant page (0 if none in the top k)."""
    for pos, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1 / pos
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    """Discounted cumulative gain, normalised by the best possible ranking.
    A relevant page at position p contributes grade / log2(p + 1)."""
    dcg = sum(relevant.get(d, 0) / math.log2(pos + 1)
              for pos, d in enumerate(ranked[:k], start=1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(pos + 1) for pos, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def evaluate(run, qrels, ks=(5, 10)):
    """Average every metric over all queries."""
    scores = {f"Recall@{k}": 0.0 for k in ks} | {"MRR@10": 0.0, "nDCG@10": 0.0}
    for qid, relevant in qrels.items():
        ranked = dedupe(run.get(qid, []))
        for k in ks:
            scores[f"Recall@{k}"] += recall_at_k(ranked, relevant, k)
        scores["MRR@10"] += reciprocal_rank(ranked, relevant, 10)
        scores["nDCG@10"] += ndcg_at_k(ranked, relevant, 10)
    return {m: round(v / len(qrels), 4) for m, v in scores.items()}


if __name__ == "__main__":
    # Toy example: compute these by hand first, then run the file and compare.
    qrels = {
        "q1": {"PACS#4": 1},                          # one relevant page
        "q2": {"Passwort#5": 1},
        "q3": {"KIS#6": 1},
        "q4": {"PACS#4": 2, "Benutzer#4": 1},         # two relevant pages, graded
    }
    run = {
        "q1": ["PACS#4", "PACS#3", "KIS#3"],                                  # hit at rank 1
        "q2": ["Benutzer#3", "Hardware#6", "Passwort#5"],                     # hit at rank 3
        "q3": ["KIS#3", "KIS#4", "Service#3", "Service#5", "KIS#5",
               "Sicher#3", "KIS#6"],                                          # hit at rank 7
        "q4": ["Benutzer#4", "Benutzer#4", "PACS#4"],   # duplicate chunk from the same page
    }
    for qid in qrels:
        ranked = dedupe(run[qid])
        print(qid, ranked[:3], "| RR =", round(reciprocal_rank(ranked, qrels[qid]), 3),
              "| nDCG@10 =", round(ndcg_at_k(ranked, qrels[qid], 10), 3))
    print("\nAverages:", evaluate(run, qrels))
