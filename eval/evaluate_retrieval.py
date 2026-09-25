"""
Step 4/5 - Evaluate retrievers on the test questions.

Every retriever is just a function:  question -> list of chunk indices, best first.
The harness turns chunk rankings into PAGE rankings (the qrels are pages), then uses
metrics.py from Step 1.

Run:  python eval/evaluate_retrieval.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))                 # so we can import from src/

from bm25 import BM25, Tokenizer                       # noqa: E402
from dense import DenseRetriever                       # noqa: E402
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


def print_table(rows, width=34):
    cols = list(next(iter(rows.values())).keys())
    print(f"{'':{width}s}" + "".join(f"{c:>12s}" for c in cols))
    for name, scores in rows.items():
        print(f"{name:{width}s}" + "".join(f"{scores[c]:12.3f}" for c in cols))


def main():
    chunks = load_jsonl(ROOT / "data" / "chunks.jsonl")
    questions = [q for q in load_jsonl(ROOT / "data" / "questions.jsonl")
                 if q["split"] == "test" and q["qrels"]]          # unanswerable ones have no qrels
    qrels = {q["id"]: q["qrels"] for q in questions}
    print(f"{len(chunks)} chunks, {len(questions)} answerable test questions\n")

    plain = [c["text"] for c in chunks]
    headed = [with_header(c) for c in chunks]

    print("Building indexes ...")
    retrievers = {
        "BM25 (stop+stem+header)":     BM25(headed, Tokenizer(stopwords=True, stemming=True)).search,
        "Dense e5-small, NO prefixes": DenseRetriever(plain, E5, "", "").search,
        "Dense e5-small":              DenseRetriever(plain, E5).search,
        "Dense e5-small + header":     DenseRetriever(headed, E5).search,
    }

    results, runs = {}, {}
    for name, search in retrievers.items():
        run, ms = run_retriever(search, questions, chunks)
        runs[name] = run
        results[name] = evaluate(run, qrels) | {"ms/query": ms}
    print()
    print_table(results)

    # ---- Recall@5 per question type, for every retriever
    types = sorted({q["type"] for q in questions})
    print("\nRecall@5 per question type:")
    rows = {}
    for name in retrievers:
        rows[name] = {f"{t}": sum(recall_at_k(runs[name][q["id"]], q["qrels"], 5)
                                  for q in questions if q["type"] == t)
                      / sum(q["type"] == t for q in questions) for t in types}
    print_table(rows)

    # ---- Do BM25 and dense fail on the SAME questions?  (this motivates hybrid search in Step 6)
    bm25_name, dense_name = "BM25 (stop+stem+header)", "Dense e5-small + header"
    hit = {n: {q["id"] for q in questions if recall_at_k(runs[n][q["id"]], q["qrels"], 5) > 0}
           for n in (bm25_name, dense_name)}
    all_ids = {q["id"] for q in questions}
    only_bm25 = hit[bm25_name] - hit[dense_name]
    only_dense = hit[dense_name] - hit[bm25_name]
    neither = all_ids - hit[bm25_name] - hit[dense_name]
    print(f"\nRecall@5 hits:  both {len(hit[bm25_name] & hit[dense_name])}, "
          f"only BM25 {len(only_bm25)}, only dense {len(only_dense)}, neither {len(neither)}")

    by_id = {q["id"]: q for q in questions}
    for label, ids in (("Found ONLY by dense", only_dense), ("Found ONLY by BM25", only_bm25),
                       ("Found by NEITHER", neither)):
        print(f"\n{label} (first 5):")
        for qid in sorted(ids)[:5]:
            q = by_id[qid]
            print(f"  [{q['type']}] {q['question']}   gold: {', '.join(q['qrels'])}")


if __name__ == "__main__":
    main()
