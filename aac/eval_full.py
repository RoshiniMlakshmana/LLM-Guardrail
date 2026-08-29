"""
eval_full.py  —  consolidated evaluation on a LARGE real held-out test set
(Block 5 calibration on more data + Block 18 eval-harness seed)

Combines every held-out source we have — seed test + deepset test + jackhhao test +
held-out domain benign — into ~550 examples, then reports binary detection metrics,
false-positive rate, AND calibration (ECE / Brier) on that big set. Bigger test set =
trustworthy numbers (ECE on 107 was noisy; this is far steadier).

Usage:  python eval_full.py
"""
import os, sys, json, random
import numpy as np
sys.path.append(os.getcwd())
from src.classifier import AttackClassifier
from src.calibration import expected_calibration_error, brier_score

rng = random.Random(42)


def build_testset():
    rows = []
    seed = [json.loads(l) for l in open("data/seed_dataset.jsonl")]
    for r in seed:
        if r["split"] == "test":
            rows.append({"text": r["text"], "y": 0 if r["label"] == "benign" else 1, "src": "seed"})
    be = [json.loads(l) for l in open("data/benign_expanded.jsonl")]
    rng.shuffle(be)
    for r in be[int(len(be) * 0.8):]:            # held-out domain benign (never trained on)
        rows.append({"text": r["text"], "y": 0, "src": "domain_benign"})
    try:
        from datasets import load_dataset
        d = load_dataset("deepset/prompt-injections")["test"]
        for r in d:
            if r.get("text"):
                rows.append({"text": r["text"], "y": int(r["label"]), "src": "deepset"})
        j = load_dataset("jackhhao/jailbreak-classification")["test"]
        for r in j:
            if r.get("prompt"):
                rows.append({"text": r["prompt"], "y": 1 if r["type"] == "jailbreak" else 0, "src": "jackhhao"})
    except Exception as e:
        print(f"[warn] public sets unavailable ({e}); using seed+domain only")
    return rows


def main(thr=0.55, model_path="artifacts/model_precision.joblib"):
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    rows = build_testset()
    model = AttackClassifier.load(model_path)
    texts = [r["text"] for r in rows]
    y = np.array([r["y"] for r in rows])
    s = np.array([p["attack_score"] for p in model.predict(texts)])
    yhat = (s >= thr).astype(int)

    from collections import Counter
    print(f"consolidated test set: {len(rows)} examples  {dict(Counter(r['src'] for r in rows))}")
    print(f"  attacks={int(y.sum())}  benign={int((y==0).sum())}\n")
    print(f"  ROC-AUC          : {roc_auc_score(y, s):.4f}")
    print(f"  precision @ {thr} : {precision_score(y, yhat, zero_division=0):.4f}")
    print(f"  recall    @ {thr} : {recall_score(y, yhat, zero_division=0):.4f}")
    print(f"  F1        @ {thr} : {f1_score(y, yhat, zero_division=0):.4f}")
    print(f"  benign FPR @ {thr}: {float((s[y==0]>=thr).mean()):.4f}")
    print(f"  ECE (10 bins)    : {expected_calibration_error(y, s, 10):.4f}   <- on {len(rows)} ex, far steadier than 107")
    print(f"  Brier            : {brier_score(y, s):.4f}")

    print("\n  recall by source (attacks only):")
    for src in ["seed", "deepset", "jackhhao"]:
        sub = [(r["text"], r["y"]) for r in rows if r["src"] == src and r["y"] == 1]
        if sub:
            ss = np.array([p["attack_score"] for p in model.predict([t for t, _ in sub])])
            print(f"    {src:10s}: {float((ss>=thr).mean()):.1%}  (n={len(sub)})")
    print("\n  benign FPR by source:")
    for src in ["seed", "domain_benign", "deepset", "jackhhao"]:
        sub = [r["text"] for r in rows if r["src"] == src and r["y"] == 0]
        if sub:
            ss = np.array([p["attack_score"] for p in model.predict(sub)])
            print(f"    {src:14s}: {float((ss>=thr).mean()):.1%}  (n={len(sub)})")


if __name__ == "__main__":
    main()
