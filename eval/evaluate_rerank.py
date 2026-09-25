"""
Step 7 - Reranker shoot-out: which reranker gives the best quality for its cost?

Run (Kaggle GPU):  python eval/evaluate_rerank.py --models small,jina,gte,large
Run (laptop CPU):  python eval/evaluate_rerank.py --models small
"""
import argparse
import gc
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
from rerank import MODELS, Reranker, RetrieveAndRerank                   # noqa: E402

ALPHA = 0.7          # chosen on the dev set in Step 6


def gold_rank(pages, qrels):
    """Position (1-based) of the first relevant page, or None if it is not in the list."""
    return next((i for i, p in enumerate(pages, start=1) if p in qrels), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="small,large", help=f"comma-separated, from {list(MODELS)}")
    ap.add_argument("--pools", default="10,20")
    args = ap.parse_args()
    keys = [k.strip() for k in args.models.split(",")]
    pools = [int(p) for p in args.pools.split(",")]

    chunks = load_jsonl(ROOT / "data" / "chunks.jsonl")
    test = [q for q in load_jsonl(ROOT / "data" / "questions.jsonl")
            if q["split"] == "test" and q["qrels"]]
    qrels = {q["id"]: q["qrels"] for q in test}
    texts = [with_header(c) for c in chunks]

    print("Building hybrid retriever ...")
    hybrid = Hybrid(BM25(texts, Tokenizer(stopwords=True, stemming=True)),
                    DenseRetriever(texts, E5), "weighted", ALPHA)

    base_run, base_ms = run_retriever(hybrid.search, test, chunks)
    results = {"Hybrid (no reranker)": evaluate(base_run, qrels) | {"params (M)": 0, "ms/query": base_ms}}
    runs = {"Hybrid (no reranker)": base_run}

    for key in keys:
        print(f"\nLoading reranker '{key}' ({MODELS[key][0]}) ...")
        try:
            rr = Reranker.from_key(key)
        except Exception as e:                                   # noqa: BLE001
            # Models with custom code sometimes break with new library versions: skip, don't crash.
            print(f"  could not load '{key}': {type(e).__name__}: {str(e)[:200]}")
            continue
        params = rr.n_params() / 1e6
        for pool in pools:
            name = f"+ {key}, pool {pool}"
            run, ms = run_retriever(RetrieveAndRerank(hybrid, rr, texts, pool).search, test, chunks)
            runs[name] = run
            results[name] = evaluate(run, qrels) | {"params (M)": params, "ms/query": ms}
            print(f"  {name}: MRR@10 {results[name]['MRR@10']:.3f}, {ms:.0f} ms/query")
        del rr                                                   # free GPU memory before the next model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:                                        # noqa: BLE001
            pass

    print("\nTest set:")
    print_table(results)

    best = max((n for n in results if n.startswith("+")), key=lambda n: results[n]["MRR@10"])
    types = sorted({q["type"] for q in test})
    print(f"\nRecall@5 per question type (best setting by MRR: {best}):")
    print_table({n: {t: np.mean([recall_at_k(runs[n][q["id"]], q["qrels"], 5)
                                 for q in test if q["type"] == t]) for t in types}
                 for n in ("Hybrid (no reranker)", best)})

    hit = {n: {q["id"] for q in test if recall_at_k(runs[n][q["id"]], q["qrels"], 5) > 0}
           for n in ("Hybrid (no reranker)", best)}
    print(f"\n{best} vs no reranker at Recall@5: gained {len(hit[best] - hit['Hybrid (no reranker)'])}, "
          f"lost {len(hit['Hybrid (no reranker)'] - hit[best])}")


if __name__ == "__main__":
    main()
