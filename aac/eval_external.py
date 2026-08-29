"""
eval_external.py  —  Block (e): honest external validation

Tests an already-trained model on a PUBLIC dataset it has NEVER seen, to check
whether the in-house (partly templated) gains hold on real-world attacks.

Default set: deepset/prompt-injections  (text, label: 1=injection/attack, 0=legit;
662 rows, Apache-2.0). Deliberately different in style/domain from our seed data,
so this is a real out-of-distribution stress test, not a home-field rematch.

Headline metric is ROC-AUC: it is threshold-free, so it measures ranking ability
without being unfairly punished by the fact that our 0.70 gate was tuned on a
different domain. We also report precision/recall/FPR at 0.70 for context.

Usage:
    python eval_external.py                       # uses artifacts/model.joblib
    python eval_external.py --threshold 0.70
"""
import os, sys, argparse
import numpy as np
sys.path.append(os.path.dirname(__file__))
from src.classifier import AttackClassifier


def load_external(name="deepset"):
    from datasets import load_dataset
    if name == "deepset":
        ds = load_dataset("deepset/prompt-injections")
        rows = []
        for split in ds.keys():
            for r in ds[split]:
                if r.get("text"):
                    rows.append({"text": r["text"], "y": int(r["label"])})
        return rows
    raise ValueError(f"unknown dataset: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/model.joblib")
    ap.add_argument("--dataset", default="deepset")
    ap.add_argument("--threshold", type=float, default=0.70)
    args = ap.parse_args()

    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                 roc_auc_score)
    print(f"Loading model: {args.model}")
    model = AttackClassifier.load(args.model)
    rows = load_external(args.dataset)
    texts = [r["text"] for r in rows]
    y = np.array([r["y"] for r in rows])

    scores = np.array([p["attack_score"] for p in model.predict(texts)])
    yhat = (scores >= args.threshold).astype(int)

    auc = roc_auc_score(y, scores) if len(set(y)) > 1 else float("nan")
    P = precision_score(y, yhat, zero_division=0)
    R = recall_score(y, yhat, zero_division=0)
    F = f1_score(y, yhat, zero_division=0)
    benign = scores[y == 0]
    fpr = float((benign >= args.threshold).mean()) if len(benign) else float("nan")

    print(f"\n===== EXTERNAL VALIDATION ({args.dataset}) =====")
    print(f"  n                       : {len(rows)}  "
          f"(attacks={int(y.sum())}, benign={int((y == 0).sum())})")
    print(f"  roc_auc (threshold-free): {auc:.4f}   <-- headline")
    print(f"  -- at gate threshold {args.threshold} --")
    print(f"  binary_precision        : {P:.4f}")
    print(f"  binary_recall           : {R:.4f}")
    print(f"  binary_f1               : {F:.4f}")
    print(f"  benign_false_pos_rate   : {fpr:.4f}")

    print("\nHow to read this honestly:")
    print("  - AUC near your in-house AUC  -> the detector generalises. Great.")
    print("  - AUC drops a lot (toward 0.5)-> it learned OUR templates, not real intent.")
    print("  - Precision/recall at 0.70 may look off because the gate threshold was")
    print("    tuned on a different domain; that's expected, not a bug. Re-calibrating")
    print("    on a mixed corpus (next work) is the fix.")


if __name__ == "__main__":
    main()
