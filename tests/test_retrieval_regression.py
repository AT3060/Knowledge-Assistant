"""
Step 15 - Regression test: the BM25 numbers published in the README must not change silently.

BM25 needs no model, so we can run the REAL evaluation (130 test questions) in every CI run.
If someone changes the tokenizer, the stopwords, the chunking or the contextual header, this
test fails and shows the new numbers - then you decide: bug, or improvement (update README + test).
"""
import pytest

from bm25 import BM25, Tokenizer
from metrics import dedupe, evaluate
from pipeline import with_header

README_BM25 = {"Recall@5": 0.760, "Recall@10": 0.849, "MRR@10": 0.640, "nDCG@10": 0.683}


def test_bm25_matches_readme(chunks, questions):
    test = [q for q in questions if q["split"] == "test" and q["qrels"]]
    assert len(test) == 130
    bm25 = BM25([with_header(c) for c in chunks], Tokenizer(stopwords=True, stemming=True))
    run = {q["id"]: dedupe([chunks[i]["page_id"] for i in bm25.search(q["question"], k=50)])
           for q in test}
    scores = evaluate(run, {q["id"]: q["qrels"] for q in test})
    for metric, expected in README_BM25.items():
        assert scores[metric] == pytest.approx(expected, abs=0.001), f"{metric}: {scores}"
