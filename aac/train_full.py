"""
train_full.py  —  Step 8: grow the training data to lift the whole trade-off curve.

Step 7 proved threshold tuning is maxed out at external AUC ~0.88. The only way to
get BOTH lower false alarms AND higher catch-rate is a better-separated model = more
and more-diverse data. Here we add a SECOND public source (jackhhao/jailbreak-
classification) on top of seed + deepset, retrain, and measure on each public set's
own held-out TEST split separately.

Honest read: the decisive signal is whether adding jackhhao (source B) lifts AUC on
the deepset (source A) held-out TEST. That is cross-source generalisation, not a
home-field rematch. Each test split is never trained on.

Usage:  python train_full.py --encoder st
"""
import os, sys, json, argparse
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.classifier import AttackClassifier


def load_seed(path="data/seed_dataset.jsonl"):
    rows = [json.loads(l) for l in open(path)]
    return ([r for r in rows if r["split"] == "train"],
            [r for r in rows if r["split"] == "test"])


def load_deepset():
    from datasets import load_dataset
    ds = load_dataset("deepset/prompt-injections")
    def conv(sp):
        return [{"text": r["text"],
                 "label": ("prompt_injection" if int(r["label"]) == 1 else "benign")}
                for r in ds[sp] if r.get("text")]
    return conv("train"), conv("test")


def load_jackhhao():
    from datasets import load_dataset
    ds = load_dataset("jackhhao/jailbreak-classification")
    def conv(sp):
        return [{"text": r["prompt"],
                 "label": ("jailbreak" if r["type"] == "jailbreak" else "benign")}
                for r in ds[sp] if r.get("prompt")]
    return conv("train"), conv("test")


def evalset(model, rows, t):
    from sklearn.metrics import roc_auc_score, recall_score
    y = np.array([0 if r["label"] == "benign" else 1 for r in rows])
    s = np.array([p["attack_score"] for p in model.predict([r["text"] for r in rows])])
    auc = roc_auc_score(y, s) if len(set(y)) > 1 else float("nan")
    rec = recall_score(y, (s >= t).astype(int), zero_division=0) if (y == 1).any() else float("nan")
    fpr = float((s[y == 0] >= t).mean()) if (y == 0).any() else float("nan")
    return auc, rec, fpr, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="st")
    ap.add_argument("--threshold", type=float, default=0.55)
    args = ap.parse_args()

    seed_tr, seed_te = load_seed()
    ds_tr, ds_te = load_deepset()
    jh_tr, jh_te = load_jackhhao()
    combined = ([{"text": r["text"], "label": r["label"]} for r in seed_tr]
                + ds_tr + jh_tr)
    print(f"training rows: seed={len(seed_tr)} + deepset={len(ds_tr)} + jackhhao={len(jh_tr)} = {len(combined)}")

    model = AttackClassifier(encoder_kind=args.encoder)
    model.fit([r["text"] for r in combined], [r["label"] for r in combined], calibrate="sigmoid")

    t = args.threshold
    print(f"\n===== held-out tests (threshold {t}) =====")
    print(f"  {'set':22s} {'AUC':>7} {'recall':>8} {'fpr':>7} {'n':>5}")
    for name, rows in [("deepset TEST (src A)", ds_te),
                       ("jackhhao TEST (src B)", jh_te),
                       ("in-house seed TEST", seed_te)]:
        auc, rec, fpr, n = evalset(model, rows, t)
        print(f"  {name:22s} {auc:>7.3f} {rec:>8.3f} {fpr:>7.3f} {n:>5}")

    print("\nReference — BEFORE adding jackhhao (Step 6/7, deepset-only training):")
    print("  deepset TEST (src A)   AUC ~0.88   <- did adding source B raise this?")

    model.save("artifacts/model_full.joblib")
    print("\nSaved: artifacts/model_full.joblib")


if __name__ == "__main__":
    main()
