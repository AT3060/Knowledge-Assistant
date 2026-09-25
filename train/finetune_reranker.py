"""
Step 8 - Fine-tune the small reranker on our domain (run on a Kaggle GPU).

Idea: the small reranker (118M) was trained on machine-translated web searches and didn't help.
The large one (568M) helped a lot but is ~5x slower. Can we teach the SMALL one our domain,
so we get close to large-model quality at small-model speed?

Training data = (question, passage, label) pairs, label 1 = relevant, 0 = not relevant.
  - Domain pairs from our TRAIN questions:
        positive: chunks of the labelled page(s)
        negatives: HARD negatives = chunks the hybrid retriever ranks high but that are wrong.
        Random negatives would be too easy ("KIS login" vs "SAP orders"); hard negatives teach
        the fine distinctions we need ("RIS Anmeldung" vs "RIS Zugangsdaten zurücksetzen").
  - GermanDPR pairs (German Wikipedia QA with hard negatives) for general German QA ability,
    so 96 domain questions don't make the model forget everything else.

Model selection: 20% of the train questions are held out as a DEV set. After every epoch we
measure dev MRR with the full pipeline and keep the best epoch. The TEST set is used only once, at the end.

Kaggle:  !python train/finetune_reranker.py --compare-large
         !python train/finetune_reranker.py --push Areftawana3/reranker-klinik-it-mMiniLM
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # one GPU is plenty; avoids multi-GPU quirks

import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from bm25 import BM25, Tokenizer                                          # noqa: E402
from dense import DenseRetriever                                          # noqa: E402
from evaluate_retrieval import E5, load_jsonl, print_table, run_retriever, with_header  # noqa: E402
from hybrid import Hybrid                                                 # noqa: E402
from metrics import evaluate, recall_at_k                                 # noqa: E402
from rerank import LARGE, SMALL, Reranker, RetrieveAndRerank              # noqa: E402

ALPHA = 0.7                  # hybrid weight chosen in Step 6
OUT = ROOT / "models" / "reranker-finetuned"


class InMemoryReranker(Reranker):
    """Same as Reranker, but wraps a model object we already have (the one being trained)."""
    def __init__(self, model, name="fine-tuned"):
        self.model, self.name = model, name


# ------------------------------------------------------------------ training data
def domain_pairs(questions, chunks, texts, hybrid, n_neg=7):
    rows = {"query": [], "passage": [], "label": []}
    for q in questions:
        gold = set(q["qrels"])
        for i, c in enumerate(chunks):                          # positives: all chunks of gold pages
            if c["page_id"] in gold:
                rows["query"].append(q["question"]); rows["passage"].append(texts[i]); rows["label"].append(1.0)
        hard = [i for i in hybrid.search(q["question"], k=20) if chunks[i]["page_id"] not in gold][:n_neg]
        for i in hard:
            rows["query"].append(q["question"]); rows["passage"].append(texts[i]); rows["label"].append(0.0)
    return rows


def load_germandpr():
    """GermanDPR used a loading script, which newer `datasets` versions no longer run.
    Try the auto-converted Parquet version first, then the script, else give up gracefully."""
    from datasets import load_dataset
    for kwargs in ({"revision": "refs/convert/parquet"}, {"trust_remote_code": True}):
        try:
            ds = load_dataset("deepset/germandpr", split="train", **kwargs)
            print(f"  GermanDPR loaded with {kwargs}: {len(ds)} questions")
            return ds
        except Exception as e:                                  # noqa: BLE001
            print(f"  GermanDPR with {kwargs} failed: {type(e).__name__}: {str(e)[:120]}")
    return None


def ctx_list(field):
    """Contexts come either as a dict of lists or as a list of dicts."""
    if isinstance(field, dict):
        return [{"title": t, "text": x} for t, x in zip(field["title"], field["text"])]
    return field or []


def dpr_pairs(ds, max_questions, n_neg=2):
    rows = {"query": [], "passage": [], "label": []}
    for ex in ds.select(range(min(max_questions, len(ds)))):
        pos, neg = ctx_list(ex["positive_ctxs"]), ctx_list(ex["hard_negative_ctxs"])
        if not pos:
            continue
        for ctx, label in [(pos[0], 1.0)] + [(n, 0.0) for n in neg[:n_neg]]:
            rows["query"].append(ex["question"])
            rows["passage"].append(f"{ctx['title']} – {ctx['text']}")
            rows["label"].append(label)
    return rows


# ------------------------------------------------------------------ evaluation helper
def pipeline_scores(reranker, hybrid, texts, questions, chunks, pool):
    run, ms = run_retriever(RetrieveAndRerank(hybrid, reranker, texts, pool).search, questions, chunks)
    return evaluate(run, {q["id"]: q["qrels"] for q in questions}) | {"ms/query": ms}, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--dpr-max", type=int, default=3000, help="GermanDPR questions to use (0 = none)")
    ap.add_argument("--domain-upsample", type=int, default=3, help="repeat domain pairs N times")
    ap.add_argument("--compare-large", action="store_true", help="also evaluate bge-reranker-v2-m3")
    ap.add_argument("--push", help="Hugging Face repo id to upload the best model to")
    args = ap.parse_args()

    from datasets import Dataset
    from sentence_transformers.cross_encoder import (CrossEncoder, CrossEncoderTrainer,
                                                     CrossEncoderTrainingArguments)
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from transformers import TrainerCallback

    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU!)")
    random.seed(42)

    chunks = load_jsonl(ROOT / "data" / "chunks.jsonl")
    texts = [with_header(c) for c in chunks]           # SAME text format as at inference time
    all_q = load_jsonl(ROOT / "data" / "questions.jsonl")
    train_all = [q for q in all_q if q["split"] == "train"]
    test = [q for q in all_q if q["split"] == "test" and q["qrels"]]
    random.shuffle(train_all)
    n_dev = len(train_all) // 5
    dev, train = train_all[:n_dev], train_all[n_dev:]
    print(f"train {len(train)} | dev {len(dev)} | test {len(test)} questions")

    print("Building hybrid retriever ...")
    hybrid = Hybrid(BM25(texts, Tokenizer(stopwords=True, stemming=True)),
                    DenseRetriever(texts, E5), "weighted", ALPHA)

    # ---- training data
    dom = domain_pairs(train, chunks, texts, hybrid)
    print(f"Domain pairs: {len(dom['label'])} ({int(sum(dom['label']))} positive), "
          f"x{args.domain_upsample} upsampled")
    rows = {k: v * args.domain_upsample for k, v in dom.items()}
    if args.dpr_max:
        ds = load_germandpr()
        if ds is not None:
            dpr = dpr_pairs(ds, args.dpr_max)
            print(f"GermanDPR pairs: {len(dpr['label'])}")
            for k in rows:
                rows[k] += dpr[k]
        else:
            print("  -> continuing with domain data only")
    train_ds = Dataset.from_dict(rows).shuffle(seed=42)
    n_pos = sum(rows["label"])
    n_neg = len(rows["label"]) - n_pos
    print(f"Training pairs: {len(train_ds)} ({int(n_pos)} positive, {int(n_neg)} negative)")

    # ---- model, loss, baseline on dev
    model = CrossEncoder(SMALL, num_labels=1)
    # pos_weight: there are ~n_neg/n_pos times more negatives; weight positives up so the model
    # doesn't learn the lazy solution "everything is irrelevant".
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(n_neg / n_pos))
    rr = InMemoryReranker(model)
    base_dev, _ = pipeline_scores(rr, hybrid, texts, dev, chunks, args.pool)
    print(f"\nDev before training: MRR@10 {base_dev['MRR@10']:.3f}, nDCG@10 {base_dev['nDCG@10']:.3f}")
    best = {"mrr": base_dev["MRR@10"], "epoch": 0}
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT))                    # epoch 0 = untouched model, as a fallback

    class DevEval(TrainerCallback):
        """After every epoch: evaluate on DEV with the real pipeline, keep the best model."""
        def on_epoch_end(self, args_, state, control, **kwargs):
            scores, _ = pipeline_scores(rr, hybrid, texts, dev, chunks, args.pool)
            epoch = round(state.epoch)
            mark = ""
            if scores["MRR@10"] > best["mrr"]:
                best.update(mrr=scores["MRR@10"], epoch=epoch)
                model.save_pretrained(str(OUT))
                mark = "  <- best so far, saved"
            print(f"Dev after epoch {epoch}: MRR@10 {scores['MRR@10']:.3f}, "
                  f"nDCG@10 {scores['nDCG@10']:.3f}{mark}")
            model.train()

    targs = CrossEncoderTrainingArguments(
        output_dir=str(ROOT / "models" / "tmp"), num_train_epochs=args.epochs,
        per_device_train_batch_size=32, learning_rate=2e-5, warmup_ratio=0.1,
        fp16=torch.cuda.is_available(), eval_strategy="no", save_strategy="no",
        logging_steps=50, report_to="none", seed=42)
    start = time.time()
    CrossEncoderTrainer(model=model, args=targs, train_dataset=train_ds, loss=loss,
                        callbacks=[DevEval()]).train()
    print(f"Training took {(time.time() - start) / 60:.1f} min; best dev epoch: {best['epoch']}")

    # ---- final comparison on TEST (touched only now)
    print("\nEvaluating on the test set ...")
    finetuned = InMemoryReranker(CrossEncoder(str(OUT)))
    candidates = {"Hybrid (no reranker)": None,
                  f"+ small (original), pool {args.pool}": Reranker(SMALL),
                  f"+ small FINE-TUNED, pool {args.pool}": finetuned,
                  "+ small FINE-TUNED, pool 10": finetuned}
    if args.compare_large:
        candidates[f"+ large, pool {args.pool}"] = Reranker(LARGE)
    results, runs = {}, {}
    for name, reranker in candidates.items():
        if reranker is None:
            run, ms = run_retriever(hybrid.search, test, chunks)
            results[name] = evaluate(run, {q["id"]: q["qrels"] for q in test}) | {"ms/query": ms}
        else:
            pool = 10 if name.endswith("pool 10") else args.pool
            results[name], run = pipeline_scores(reranker, hybrid, texts, test, chunks, pool)
        runs[name] = run
    print_table(results)

    types = sorted({q["type"] for q in test})
    print("\nRecall@5 per question type:")
    print_table({n: {t: np.mean([recall_at_k(runs[n][q["id"]], q["qrels"], 5)
                                 for q in test if q["type"] == t]) for t in types} for n in runs})

    ft, orig = f"+ small FINE-TUNED, pool {args.pool}", f"+ small (original), pool {args.pool}"
    hit = {n: {q["id"] for q in test if recall_at_k(runs[n][q["id"]], q["qrels"], 5) > 0}
           for n in (orig, ft)}
    print(f"\nFine-tuned vs original small at Recall@5: gained {len(hit[ft] - hit[orig])}, "
          f"lost {len(hit[orig] - hit[ft])}")

    if args.push:
        finetuned.model.push_to_hub(args.push)
        print(f"Uploaded to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
