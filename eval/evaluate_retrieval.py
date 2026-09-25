"""
Steps 4-6 - Evaluate retrievers on the test questions.

Every retriever is just a function:  question -> list of chunk indices, best first.
The harness turns chunk rankings into PAGE rankings (the qrels are pages), then uses
metrics.py from Step 1.

Earlier comparisons (Step 4: tokenizer options, Step 5: prefixes and headers) are in the
Git history; this version compares the best BM25, the best dense, and hybrid fusion.

Run:  python eval/evaluate_retrieval.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))                 # so we can import from src/

from bm25 import BM25, Tokenizer                       # noqa: E402
from dense import DenseRetriever                       # noqa: E402
from hybrid import Hybrid, rrf, weighted               # noqa: E402
from metrics import dedupe, evaluate, recall_at_k      # noqa: E402  (eval/metrics.py)

E5 = "intfloat/multilingual-e5-small"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def with_header(chunk):
    """'Contextual header': document name and section title in front of the chunk text."""
    doc = chunk["doc"].removesuffix(".pdf").replace("_", " ")
    return f"{doc} – {chunk['title']}: {chunk['text']}"


def run_retriever(search, questions, chunks):
    """Returns (run, avg latency in ms). run = {question_id: [page_id, ...]}, best first."""
    run, start = {}, time.perf_counter()
    for q in questions:
        run[q["id"]] = dedupe([chunks[i]["page_id"] for i in search(q["question"])])
    return run, (time.perf_counter() - start) / len(questions) * 1000


def run_from_scores(score_fn, questions, chunks, k=50):
    """Like run_retriever, but from a function that returns fused chunk scores directly."""
    return {q["id"]: dedupe([chunks[i]["page_id"]
                             for i in np.argsort(-score_fn(q), kind="stable")[:k]])
            for q in questions}


def print_table(rows, width=34):
    cols = list(next(iter(rows.values())).keys())
    print(f"{'':{width}s}" + "".join(f"{c:>14s}" for c in cols))
    for name, scores in rows.items():
        print(f"{name:{width}s}" + "".join(f"{scores[c]:14.3f}" for c in cols))


def main():
    chunks = load_jsonl(ROOT / "data" / "chunks.jsonl")
    all_q = load_jsonl(ROOT / "data" / "questions.jsonl")
    test = [q for q in all_q if q["split"] == "test" and q["qrels"]]    # unanswerable: no qrels
    dev = [q for q in all_q if q["split"] == "train"]                   # see "choosing α" below
    print(f"{len(chunks)} chunks | {len(test)} answerable test questions | "
          f"{len(dev)} dev questions (train split, used only to choose α)\n")

    headed = [with_header(c) for c in chunks]
    print("Building indexes ...")
    bm25 = BM25(headed, Tokenizer(stopwords=True, stemming=True))
    dense = DenseRetriever(headed, E5)
    hybrid = Hybrid(bm25, dense)

    # Score every question once with both retrievers and cache the result,
    # so trying many fusion settings doesn't re-run the embedding model.
    cache = {q["id"]: hybrid.score_pair(q["question"]) for q in dev + test}

    # ---- Choosing α: tune on DEV, report on TEST.
    # If we picked the α that looks best on the test set, the test score would be optimistic:
    # we'd have fitted a parameter to the very questions we report on. The train questions
    # are not used for anything yet (fine-tuning comes in Step 8), so they serve as a dev set.
    print("\nWeighted fusion  α·dense + (1-α)·BM25   (α=0: pure BM25, α=1: pure dense)")
    sweep = {}
    for alpha in np.round(np.arange(0, 1.01, 0.1), 1):
        fn = lambda q, a=alpha: weighted(*cache[q["id"]], a)
        dev_s = evaluate(run_from_scores(fn, dev, chunks), {q["id"]: q["qrels"] for q in dev})
        test_s = evaluate(run_from_scores(fn, test, chunks), {q["id"]: q["qrels"] for q in test})
        sweep[f"α = {alpha}"] = {"dev nDCG@10": dev_s["nDCG@10"], "test nDCG@10": test_s["nDCG@10"],
                                 "test R@5": test_s["Recall@5"]}
    print_table(sweep, width=12)
    best_alpha = float(max(sweep, key=lambda r: sweep[r]["dev nDCG@10"]).split("= ")[1])
    print(f"-> α chosen on dev: {best_alpha}")

    # ---- Main comparison on the test set (real end-to-end latency per query)
    retrievers = {
        "BM25 (stop+stem+header)":     bm25.search,
        "Dense e5-small + header":     dense.search,
        "Hybrid RRF (k=60)":           Hybrid(bm25, dense, "rrf").search,
        f"Hybrid weighted (α={best_alpha})": Hybrid(bm25, dense, "weighted", best_alpha).search,
    }
    qrels = {q["id"]: q["qrels"] for q in test}
    results, runs = {}, {}
    for name, search in retrievers.items():
        run, ms = run_retriever(search, test, chunks)
        runs[name] = run
        results[name] = evaluate(run, qrels) | {"ms/query": ms}
    print("\nTest set:")
    print_table(results)

    # ---- Recall@5 per question type
    types = sorted({q["type"] for q in test})
    print("\nRecall@5 per question type:")
    print_table({name: {t: np.mean([recall_at_k(runs[name][q["id"]], q["qrels"], 5)
                                    for q in test if q["type"] == t]) for t in types}
                 for name in retrievers})

    # ---- What did fusion change compared with dense alone?
    by_id = {q["id"]: q for q in test}
    dense_name, rrf_name = "Dense e5-small + header", "Hybrid RRF (k=60)"
    hit = {n: {q["id"] for q in test if recall_at_k(runs[n][q["id"]], q["qrels"], 5) > 0}
           for n in (dense_name, rrf_name)}
    gained, lost = hit[rrf_name] - hit[dense_name], hit[dense_name] - hit[rrf_name]
    print(f"\nRRF vs dense at Recall@5: gained {len(gained)} questions, lost {len(lost)}")
    for label, ids in (("Gained by RRF", gained), ("Lost by RRF", lost)):
        for qid in sorted(ids)[:5]:
            q = by_id[qid]
            print(f"  {label}: [{q['type']}] {q['question']}   gold: {', '.join(q['qrels'])}")


if __name__ == "__main__":
    main()
