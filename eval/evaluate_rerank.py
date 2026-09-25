"""
Step 7 - Does a cross-encoder reranker improve the hybrid retriever, and what does it cost?

Run:  python eval/evaluate_rerank.py            (small reranker, ~0.5 GB download)
      python eval/evaluate_rerank.py --large    (also bge-reranker-v2-m3, ~2.3 GB, slow on CPU)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bm25 import BM25, Tokenizer                                         # noqa: E402
from dense import DenseRetriever                                         # noqa: E402
from evaluate_retrieval import (E5, load_jsonl, print_table,             # noqa: E402
                                run_retriever, with_header)
from hybrid import Hybrid                                                # noqa: E402
from metrics import evaluate, recall_at_k                                # noqa: E402
from rerank import LARGE, SMALL, Reranker, RetrieveAndRerank             # noqa: E402

ALPHA = 0.7          # chosen on the dev set in Step 6


def gold_rank(pages, qrels):
    """Position (1-based) of the first relevant page, or None if it is not in the list."""
    return next((i for i, p in enumerate(pages, start=1) if p in qrels), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true", help="also evaluate the large reranker")
    args = ap.parse_args()

    chunks = load_jsonl(ROOT / "data" / "chunks.jsonl")
    test = [q for q in load_jsonl(ROOT / "data" / "questions.jsonl")
            if q["split"] == "test" and q["qrels"]]
    qrels = {q["id"]: q["qrels"] for q in test}
    texts = [with_header(c) for c in chunks]

    print("Building indexes and loading rerankers ...")
    hybrid = Hybrid(BM25(texts, Tokenizer(stopwords=True, stemming=True)),
                    DenseRetriever(texts, E5), "weighted", ALPHA)
    rerankers = [Reranker(SMALL)] + ([Reranker(LARGE)] if args.large else [])

    # ---- How much can a reranker possibly fix?  It only re-sorts the pool, so the
    # retriever's recall at the pool size is a hard ceiling.
    base_run, base_ms = run_retriever(hybrid.search, test, chunks)
    print("\nCeiling: share of test questions with a relevant page inside the hybrid pool")
    for n in (5, 10, 20, 30):
        pool_run, _ = run_retriever(lambda q, n=n: hybrid.search(q, k=n), test, chunks)
        hit = np.mean([gold_rank(pool_run[q["id"]], q["qrels"]) is not None for q in test])
        print(f"  top {n:2d} chunks: {hit:.3f}")

    # ---- Main comparison
    results, runs = {"Hybrid (no reranker)": evaluate(base_run, qrels) | {"ms/query": base_ms}}, {}
    runs["Hybrid (no reranker)"] = base_run
    for rr in rerankers:
        short = "small" if rr.name == SMALL else "large"
        for pool in (10, 20, 30):
            name = f"+ rerank {short}, pool {pool}"
            run, ms = run_retriever(RetrieveAndRerank(hybrid, rr, texts, pool).search, test, chunks)
            runs[name] = run
            results[name] = evaluate(run, qrels) | {"ms/query": ms}
    print("\nTest set:")
    print_table(results)

    # ---- Per question type: no reranker vs. the best reranked setting (by MRR)
    best = max((n for n in results if n.startswith("+")), key=lambda n: results[n]["MRR@10"])
    types = sorted({q["type"] for q in test})
    print(f"\nRecall@5 per question type (best reranked setting: {best}):")
    print_table({n: {t: np.mean([recall_at_k(runs[n][q["id"]], q["qrels"], 5)
                                 for q in test if q["type"] == t]) for t in types}
                 for n in ("Hybrid (no reranker)", best)})

    # ---- The ambiguity cases from Steps 4 and 5: did the reranker fix them?
    print("\nRank of the first correct page for the 'gesperrt' questions (None = not in top 10):")
    for q in test:
        if "gesperrt bin" in q["question"]:
            before = gold_rank(runs["Hybrid (no reranker)"][q["id"]][:10], q["qrels"])
            after = gold_rank(runs[best][q["id"]][:10], q["qrels"])
            print(f"  {before!s:>4} -> {after!s:<4}  {q['question']}")

    # ---- What changed?
    hit = {n: {q["id"] for q in test if recall_at_k(runs[n][q["id"]], q["qrels"], 5) > 0}
           for n in ("Hybrid (no reranker)", best)}
    gained, lost = hit[best] - hit["Hybrid (no reranker)"], hit["Hybrid (no reranker)"] - hit[best]
    by_id = {q["id"]: q for q in test}
    print(f"\nReranker vs no reranker at Recall@5: gained {len(gained)}, lost {len(lost)}")
    for label, ids in (("Gained", gained), ("Lost", lost)):
        for qid in sorted(ids)[:5]:
            q = by_id[qid]
            print(f"  {label}: [{q['type']}] {q['question']}   gold: {', '.join(q['qrels'])}")


if __name__ == "__main__":
    main()
