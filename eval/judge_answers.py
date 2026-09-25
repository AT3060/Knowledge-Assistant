"""
Step 10a - LLM-as-a-judge: grade every answer with a second, larger model (run on Kaggle GPU).

For each ANSWERED question the judge sees:
  - the question
  - the numbered passages the answering model saw          -> faithfulness, citation support
  - the answer
  - the reference page(s) we labelled as correct            -> correctness
and returns a JSON verdict. Refusals and unanswerable questions need no judge (checked automatically).

Kaggle:  python eval/judge_answers.py --answers results/answers_Qwen2.5-7B-Instruct.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generate import Generator            # noqa: E402

JUDGE_PROMPT = """You are a strict evaluator of a German question-answering system for hospital IT documentation.

QUESTION:
{question}

PASSAGES shown to the system (numbered):
{passages}

SYSTEM ANSWER:
{answer}

REFERENCE (the documentation page(s) that contain the correct answer):
{reference}

Evaluate the SYSTEM ANSWER. Reply with ONLY a JSON object, no other text:
{{
  "faithfulness": "full" | "partial" | "none",
      // full = every statement is supported by the PASSAGES; partial = some statements are not; none = mostly unsupported
  "citation_support": "full" | "partial" | "none",
      // does each cited passage [n] actually support the statement it is attached to?
  "correctness": "correct" | "partially_correct" | "incorrect",
      // compared with the REFERENCE: does the answer give the right information for the question?
  "relevance": 1 | 2 | 3,
      // 3 = directly answers the question, 2 = partly, 1 = off-topic
  "explanation": "one short sentence"
}}"""


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def page_block(page, n=None):
    head = f"[{n}] " if n is not None else ""
    return f"{head}{page['doc']}, Seite {page['page']} – {page['title']}:\n{page['text']}"


def parse_verdict(text):
    """Take the first {...} block in the reply and parse it; None if impossible."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    raw = re.sub(r"//[^\n]*", "", m.group(0))          # drop comments if the judge copied them
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--judge", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--no-4bit", action="store_true", help="load the judge in 16-bit (needs more memory)")
    args = ap.parse_args()

    pages = {p["id"]: p for p in load_jsonl(ROOT / "data" / "pages.jsonl")}
    rows = load_jsonl(ROOT / args.answers)
    todo = [r for r in rows if r["qrels"] and not r["abstained"]]
    print(f"{len(rows)} answers, {len(todo)} to judge (answered, answerable)")

    print(f"Loading judge {args.judge} ({'16-bit' if args.no_4bit else '4-bit'}) ...")
    judge = Generator(args.judge, load_4bit=not args.no_4bit)

    out_path = ROOT / args.answers.replace("answers_", "judged_")
    failed = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(todo, start=1):
            # The answering model saw chunks; we show the judge the full pages they came from.
            passages = "\n\n".join(page_block(pages[p["page_id"]], n)
                                   for n, p in enumerate(r["passages"], start=1))
            reference = "\n\n".join(page_block(pages[pid]) for pid in r["qrels"])
            prompt = JUDGE_PROMPT.format(question=r["question"], passages=passages,
                                         answer=r["raw_answer"], reference=reference)
            reply = judge.generate([{"role": "user", "content": prompt}], max_new_tokens=200)
            verdict = parse_verdict(reply)
            if verdict is None:
                failed += 1
            f.write(json.dumps({"id": r["id"], "verdict": verdict, "judge_reply": reply,
                                "judge": args.judge}, ensure_ascii=False) + "\n")
            v = verdict or {}
            print(f"[{i}/{len(todo)}] {r['question'][:55]:55s} "
                  f"faith={v.get('faithfulness')} corr={v.get('correctness')} rel={v.get('relevance')}")

    print(f"\nSaved {out_path.relative_to(ROOT)}  (unparseable verdicts: {failed})")


if __name__ == "__main__":
    main()
