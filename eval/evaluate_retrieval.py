"""
Step 4 - Evaluate retrievers on the test questions.

Every retriever is just a function:  question -> list of chunk indices, best first.
The harness turns chunk rankings into PAGE rankings (the qrels are pages), then uses
metrics.py from Step 1. Steps 5 and 6 add dense and hybrid retrievers to the same table.

Run:  python eval/evaluate_retrieval.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))                 # so we can import src/bm25.py

from bm25 import BM25, Tokenizer                       # noqa: E402
from metrics import dedupe, evaluate, recall_at_k      # noqa: E402  (eval/metrics.py)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def with_header(chunk):
    """'Contextual header': put document name and section title in front of the chunk text,
    so a chunk like 'Nach der Genehmigung ...' still says WHICH system it belongs to."""
    doc = chunk["doc"].removesuffix(".pdf").replace("_", " ")
    return f"{doc} – {chunk['title']}: {chunk['text']}"


def run_retriever(search, questions, chunks):
    """Returns (run, avg latency in ms). run = {question_id: [page_id, ...]}, best first."""
    run, start = {}, time.perf_counter()
    for q in questions:
        run[q["id"]] = dedupe([chunks[i]["page_id"] for i in search(q["question"])])
    return run, (time.perf_counter() - start) / len(questions) * 1000


def print_table(rows):
    metrics = list(next(iter(rows.values())).keys())
    print(f"{'':32s}" + "".join(f"{m:>11s}" for m in metrics))
    for name, scores in rows.items():
        print(f"{name:32s}" + "".join(f"{scores[m]:11.3f}" for m in metrics))


def main():
    chunks = load_jsonl(ROOT / "data" / "chunks.jsonl")
    questions = [q for q in load_jsonl(ROOT / "data" / "questions.jsonl")
                 if q["split"] == "test" and q["qrels"]]          # unanswerable ones have no qrels
    qrels = {q["id"]: q["qrels"] for q in questions}
    print(f"{len(chunks)} chunks, {len(questions)} answerable test questions\n")

    plain = [c["text"] for c in chunks]
    headed = [with_header(c) for c in chunks]
    retrievers = {
        "BM25 basic":                   BM25(plain, Tokenizer()).search,
        "BM25 + stopwords":             BM25(plain, Tokenizer(stopwords=True)).search,
        "BM25 + stopwords + stemming":  BM25(plain, Tokenizer(stopwords=True, stemming=True)).search,
        "BM25 + stop + stem + header":  BM25(headed, Tokenizer(stopwords=True, stemming=True)).search,
    }

    results, runs = {}, {}
    for name, search in retrievers.items():
        run, ms = run_retriever(search, questions, chunks)
        runs[name] = run
        results[name] = evaluate(run, qrels) | {"ms/query": ms}
    print_table(results)

    # ---- which kinds of questions are hard?  (per question type, last retriever)
    best = list(retrievers)[-1]
    print(f"\nPer question type ({best}):")
    by_type = {}
    for q in questions:
        by_type.setdefault(q["type"], []).append(q)
    print_table({f"{t} (n={len(qs)})": evaluate({q['id']: runs[best][q['id']] for q in qs},
                                                 {q['id']: q['qrels'] for q in qs})
                 for t, qs in by_type.items()})

    # ---- error analysis: questions where the right page is not even in the top 5
    misses = [q for q in questions if recall_at_k(runs[best][q["id"]], q["qrels"], 5) == 0]
    print(f"\nMisses at Recall@5 ({len(misses)} questions), first 8:")
    for q in misses[:8]:
        gold = ", ".join(q["qrels"])
        print(f"  [{q['type']}] {q['question']}\n      gold: {gold}\n      got:  {runs[best][q['id']][:3]}")


if __name__ == "__main__":
    main()
