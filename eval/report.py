"""
Step 10b - The RAG evaluation report (runs on the laptop, no GPU needed).

Combines the saved answers, the judge verdicts and (optionally) your own manual grades.

Run:  python eval/report.py --model Qwen2.5-7B-Instruct
      python eval/report.py --model Qwen2.5-7B-Instruct --export-human 20   # sheet for manual grading
"""
import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def bootstrap_ci(values, n_boot=2000, seed=0):
    """95% confidence interval of the mean: resample the questions with replacement many times."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values)).mean() for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])


def fmt(values, ci=True):
    values = list(values)
    if not values:
        return "n/a"
    m = float(np.mean(values))
    if not ci:
        return f"{m:.3f}"
    lo, hi = bootstrap_ci(values)
    return f"{m:.3f}  [{lo:.3f}, {hi:.3f}]"


def share(values, label):
    values = list(values)
    return sum(v == label for v in values) / len(values) if values else float("nan")


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--export-human", type=int, default=0, help="write N answers to a CSV for manual grading")
    args = ap.parse_args()

    rows = load_jsonl(RESULTS / f"answers_{args.model}.jsonl")
    judged_path = RESULTS / f"judged_{args.model}.jsonl"
    verdicts = ({j["id"]: j["verdict"] for j in load_jsonl(judged_path) if j["verdict"]}
                if judged_path.exists() else {})

    ans = [r for r in rows if r["qrels"]]
    una = [r for r in rows if not r["qrels"]]
    answered = [r for r in ans if not r["abstained"]]
    print(f"===== RAG evaluation: {args.model} =====")
    print(f"{len(ans)} answerable + {len(una)} unanswerable test questions;  "
          f"values: mean  [95% bootstrap CI]\n")

    # ---- 1. refusals
    print("1. Refusal behaviour")
    print(f"   unanswerable correctly refused:   {fmt([r['abstained'] for r in una])}")
    print(f"   answerable wrongly refused:       {fmt([r['abstained'] for r in ans])}")

    # ---- 2. citations (automatic)
    cover = [np.mean(["[" in s for s in sentences(r["raw_answer"])]) for r in answered]
    cites_label = [any(c["page_id"] in r["qrels"] for c in r["citations"]) for r in answered]
    print("\n2. Citations (automatic)")
    print(f"   answers with >= 1 citation:       {fmt([bool(r['citations']) for r in answered], ci=False)}")
    print(f"   sentences carrying a citation:    {fmt(cover)}")
    print(f"   answer cites a labelled page:     {fmt(cites_label)}")
    print(f"   invalid citation numbers:         {sum(len(r['invalid_citations']) for r in rows)}")

    # ---- 3. judge
    if verdicts:
        v = [verdicts[r["id"]] for r in answered if r["id"] in verdicts]
        print(f"\n3. LLM judge ({len(v)} judged answers)")
        for key, labels in (("faithfulness", ("full", "partial", "none")),
                            ("citation_support", ("full", "partial", "none")),
                            ("correctness", ("correct", "partially_correct", "incorrect"))):
            parts = ", ".join(f"{lab} {share([x.get(key) for x in v], lab):.3f}" for lab in labels)
            print(f"   {key:18s} {parts}")
        print(f"   relevance (1-3)    mean {np.mean([x.get('relevance', 0) for x in v]):.2f}")

        # End-to-end: over ALL answerable questions; a refusal counts as not correct.
        e2e = [verdicts.get(r["id"], {}).get("correctness") == "correct" if not r["abstained"] else False
               for r in ans]
        faithful = [x.get("faithfulness") == "full" for x in v]
        print("\n   Headline numbers")
        print(f"   END-TO-END correct (all {len(ans)} answerable):  {fmt(e2e)}")
        print(f"   fully faithful (answered):              {fmt(faithful)}")

        # ---- 4. per question type
        print("\n4. End-to-end correct per question type")
        for t in sorted({r["type"] for r in ans}):
            vals = [ok for ok, r in zip(e2e, ans) if r["type"] == t]
            print(f"   {t:12s} (n={len(vals):3d})  {fmt(vals, ci=False)}")

        # ---- 5. is the reranker score a useful confidence signal?
        print("\n5. Top reranker score as confidence signal")
        bins = [(0.0, 0.1), (0.1, 0.5), (0.5, 1.01)]
        for lo, hi in bins:
            grp = [(ok, r) for ok, r in zip(e2e, ans) if lo <= r["top_score"] < hi]
            if grp:
                print(f"   answerable, score {lo:.1f}-{min(hi, 1):.1f}: n={len(grp):3d}, "
                      f"end-to-end correct {np.mean([ok for ok, _ in grp]):.3f}")
        s = sorted(r["top_score"] for r in una)
        print(f"   unanswerable scores: min {s[0]:.2f}, median {s[len(s) // 2]:.2f}, max {s[-1]:.2f}")

        # ---- 6. judge vs. human
        human_path = RESULTS / f"human_check_{args.model}.csv"
        if human_path.exists():
            with open(human_path, encoding="utf-8-sig") as f:
                graded = [row for row in csv.DictReader(f, delimiter=";") if row["human_correctness"].strip()]
            if graded:
                agree = [row["human_correctness"].strip() == row["judge_correctness"] for row in graded]
                print(f"\n6. Judge vs. your manual grades: agreement {np.mean(agree):.3f} on {len(graded)} answers")
                for row in graded:
                    if row["human_correctness"].strip() != row["judge_correctness"]:
                        print(f"   disagree: {row['question'][:60]}  judge={row['judge_correctness']} "
                              f"you={row['human_correctness'].strip()}")
    else:
        print(f"\n(no judge file {judged_path.name} yet - run eval/judge_answers.py on Kaggle)")

    # ---- 7. latency
    total = [r["t_retrieve"] + r["t_generate"] for r in rows]
    print(f"\n7. Latency per question: mean {np.mean(total):.2f}s, median {np.median(total):.2f}s, "
          f"90th percentile {np.percentile(total, 90):.2f}s  (retrieval {np.mean([r['t_retrieve'] for r in rows]):.2f}s)")

    # ---- export a sheet for manual grading
    if args.export_human:
        path = RESULTS / f"human_check_{args.model}.csv"
        if path.exists():
            print(f"\n{path.name} already exists - not overwriting your grades. Delete it to export a new sheet.")
            return
        random.seed(1)
        sample = random.sample([r for r in answered if r["id"] in verdicts], args.export_human)
        pages = {p["id"]: p for p in load_jsonl(ROOT / "data" / "pages.jsonl")}
        with open(path, "w", encoding="utf-8-sig", newline="") as f:     # utf-8-sig + ';' opens cleanly in German Excel
            w = csv.writer(f, delimiter=";")
            w.writerow(["id", "question", "answer", "cited_pages", "reference_pages", "reference_text",
                        "judge_correctness", "human_correctness"])
            for r in sample:
                w.writerow([r["id"], r["question"], r["raw_answer"],
                            ", ".join(c["page_id"] for c in r["citations"]), ", ".join(r["qrels"]),
                            " | ".join(pages[p]["text"] for p in r["qrels"]),
                            verdicts[r["id"]].get("correctness"), ""])
        print(f"\nWrote {path.relative_to(ROOT)}: fill in 'human_correctness' with "
              f"correct / partially_correct / incorrect, save, and run this report again.")


if __name__ == "__main__":
    main()
