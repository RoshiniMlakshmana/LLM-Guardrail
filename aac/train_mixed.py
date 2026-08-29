"""
train_mixed.py  —  Block (e) part 2: fold public data INTO training, then re-test.

Step 5 showed the template-only model collapsed on real attacks (external AUC ~0.56).
Here we mix the PUBLIC training half into our lessons and re-measure on the PUBLIC
test half the model never sees, plus our own in-house test, so we can tell:
  - did external generalisation improve?
  - did we wreck the in-house performance in the process?

Leak guard: deepset/prompt-injections ships its own train/test split. We train only
on its TRAIN rows and evaluate only on its TEST rows. No exam answers leak in.

Public attacks are folded in as label 'prompt_injection' (they are injection-style);
benign public rows become 'benign'. The binary attack-vs-benign score is what the
gate and the external metric use.

Usage:  python train_mixed.py --encoder st
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
    def conv(split):
        out = []
        for r in ds[split]:
            if not r.get("text"):
                continue
            label = "prompt_injection" if int(r["label"]) == 1 else "benign"
            out.append({"text": r["text"], "label": label, "y": int(r["label"])})
        return out
    return conv("train"), conv("test")


def binary_eval(model, rows, threshold=0.70):
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                 roc_auc_score)
    texts = [r["text"] for r in rows]
    y = np.array([0 if r["label"] == "benign" else 1 for r in rows])
    scores = np.array([p["attack_score"] for p in model.predict(texts)])
    yhat = (scores >= threshold).astype(int)
    auc = roc_auc_score(y, scores) if len(set(y)) > 1 else float("nan")
    fpr = float((scores[y == 0] >= threshold).mean()) if (y == 0).any() else float("nan")
    return {
        "n": len(rows), "auc": auc,
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "fpr": fpr,
    }


def show(tag, m):
    print(f"  {tag:28s} AUC={m['auc']:.4f}  P={m['precision']:.3f}  "
          f"R={m['recall']:.3f}  F1={m['f1']:.3f}  FPR={m['fpr']:.3f}  (n={m['n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="st", choices=["tfidf", "st", "auto"])
    args = ap.parse_args()

    seed_tr, seed_te = load_seed()
    pub_tr, pub_te = load_deepset()
    print(f"seed: train={len(seed_tr)} test={len(seed_te)} | "
          f"public: train={len(pub_tr)} test={len(pub_te)}")

    combined = [{"text": r["text"], "label": r["label"]} for r in seed_tr] + \
               [{"text": r["text"], "label": r["label"]} for r in pub_tr]
    print(f"combined training rows: {len(combined)}")

    model = AttackClassifier(encoder_kind=args.encoder)
    model.fit([r["text"] for r in combined], [r["label"] for r in combined])

    print("\n===== AFTER folding public data into training =====")
    show("external (public TEST)", binary_eval(model, pub_te))
    show("in-house (seed TEST)",  binary_eval(model, seed_te))
    print("\nReference — BEFORE (template-only model, Step 5):")
    print("  external (public TEST)       AUC=0.5625  R=0.175   <- the model we are beating")

    model.save("artifacts/model_mixed.joblib")
    print("\nSaved: artifacts/model_mixed.joblib")
    print("Note: public TEST shares deepset's templating with its TRAIN half, so this")
    print("AUC is optimistic vs a brand-new source; still a fair before/after on the SAME exam.")


if __name__ == "__main__":
    main()
