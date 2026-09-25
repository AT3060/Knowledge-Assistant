"""
Step 9 - Run the full RAG pipeline on the test questions and save every answer.

Quick automatic checks are printed here (abstention, citations, latency).
The systematic answer evaluation (groundedness, relevance) follows in Step 10,
using the saved results/answers_<model>.jsonl files.

Kaggle:  python eval/generate_answers.py --model Qwen/Qwen2.5-7B-Instruct --limit 10   # quick test
         python eval/generate_answers.py --model Qwen/Qwen2.5-7B-Instruct              # all 142
         python eval/generate_answers.py --model Qwen/Qwen2.5-7B-Instruct --load-4bit  # 4-bit variant
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generate import Generator                   # noqa: E402
from pipeline import RAGPipeline                  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (0 = all)")
    args = ap.parse_args()

    chunks = load_jsonl(ROOT / "data" / "chunks.jsonl")
    # ALL test questions, including the 12 unanswerable ones: we want to see the model refuse.
    test = [q for q in load_jsonl(ROOT / "data" / "questions.jsonl") if q["split"] == "test"]
    if args.limit:
        # keep a mix: the first N answerable plus a few unanswerable ones
        answerable = [q for q in test if q["qrels"]][:args.limit]
        unanswerable = [q for q in test if not q["qrels"]][:max(2, args.limit // 5)]
        test = answerable + unanswerable

    print(f"Loading LLM {args.model} ({'4-bit' if args.load_4bit else '16-bit'}) ...")
    generator = Generator(args.model, load_4bit=args.load_4bit)
    print("Loading retrieval pipeline ...")
    rag = RAGPipeline(chunks, generator)

    tag = args.model.split("/")[-1] + ("-4bit" if args.load_4bit else "")
    out_path = ROOT / "results" / f"answers_{tag}.jsonl"
    out_path.parent.mkdir(exist_ok=True)

    rows = []
    with open(out_path, "w", encoding="utf-8") as f:
        for i, q in enumerate(test, start=1):
            r = rag.answer(q["question"])
            row = {"id": q["id"], "question": q["question"], "type": q["type"], "qrels": q["qrels"]} | r
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(test)}] {q['question'][:60]:60s} "
                  f"{'ABSTAINED' if r['abstained'] else str(len(r['citations'])) + ' cit.'}  "
                  f"{r['t_generate']:.1f}s")

    # ---- quick automatic checks
    ans = [r for r in rows if r["qrels"]]
    una = [r for r in rows if not r["qrels"]]
    answered = [r for r in ans if not r["abstained"]]
    with_cit = [r for r in answered if r["citations"]]
    # citation precision: share of cited pages that are labelled relevant (labels are incomplete,
    # so this is a LOWER bound - a cited page can be relevant without being in our labels)
    cited = [c["page_id"] in r["qrels"] for r in with_cit for c in r["citations"]]
    any_correct = [any(c["page_id"] in r["qrels"] for c in r["citations"]) for r in with_cit]

    print(f"\n===== {tag} =====")
    print(f"Answerable questions:   {len(ans)}")
    print(f"  answered (not refused):          {len(answered) / len(ans):.3f}")
    print(f"  answers with >= 1 citation:      {len(with_cit) / max(len(answered), 1):.3f}")
    print(f"  cited pages that are labelled:   {np.mean(cited) if cited else float('nan'):.3f}  (lower bound)")
    print(f"  answers citing a labelled page:  {np.mean(any_correct) if any_correct else float('nan'):.3f}")
    print(f"  invalid citation numbers:        {sum(len(r['invalid_citations']) for r in rows)}")
    if una:
        print(f"Unanswerable questions: {len(una)}")
        print(f"  correctly refused:               {np.mean([r['abstained'] for r in una]):.3f}")
    print(f"Latency: retrieval {np.mean([r['t_retrieve'] for r in rows]):.2f}s, "
          f"generation {np.mean([r['t_generate'] for r in rows]):.2f}s per question")
    print(f"Saved to {out_path.relative_to(ROOT)}")

    print("\nExamples:")
    for r in (answered[:2] + una[:1]):
        print(f"\nQ: {r['question']}\nA: {r['raw_answer']}\n   sources: "
              + ", ".join(f"[{c['n']}] {c['page_id']}" for c in r["citations"]))


if __name__ == "__main__":
    main()
