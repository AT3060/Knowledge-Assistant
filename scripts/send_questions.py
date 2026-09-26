"""
Step 16 - Send test questions to the running API, so the dashboard has something to show.

Uses only the Python standard library (no extra install). Picks a random mix of test questions,
including the unanswerable ones, plus a few questions with words the handbooks never use.

    python scripts/send_questions.py              # 20 questions to http://127.0.0.1:8000
    python scripts/send_questions.py --n 40 --url http://127.0.0.1:8000
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Words users say, but the handbooks don't ("Fileserver" vs "Netzlaufwerke", Step 13).
VOCABULARY_GAPS = [
    "Die Rechner in der Radiologie erreichen den zentralen Fileserver nicht?",
    "Mein Share auf dem Dateiserver ist weg, was tun?",
    "Wie komme ich von zu Hause ins Firmennetz?",
]


def ask(url, question):
    req = urllib.request.Request(f"{url}/ask", data=json.dumps({"question": question}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(ROOT / "data" / "questions.jsonl", encoding="utf-8") as f:
        test = [json.loads(line) for line in f if '"split": "test"' in line]
    rng = random.Random(args.seed)
    questions = [q["question"] for q in rng.sample(test, min(args.n, len(test)))] + VOCABULARY_GAPS

    print(f"Sending {len(questions)} questions to {args.url}/ask (each takes a few seconds on CPU)\n")
    for i, q in enumerate(questions, 1):
        start = time.perf_counter()
        try:
            r = ask(args.url, q)
            outcome = "guardrail" if r["guardrail"] else "abstained" if r["abstained"] else "answered"
            print(f"{i:3d}. {outcome:9s} conf={r['confidence']:.2f} {time.perf_counter() - start:5.1f}s  {q[:60]}")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"{i:3d}. ERROR {e}  {q[:60]}")


if __name__ == "__main__":
    main()
