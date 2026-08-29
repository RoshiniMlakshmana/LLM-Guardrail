"""
train_balanced.py  —  Step 9: fix the negative transfer with balanced sampling.

Step 8 showed a big, stylistically-different source (jackhhao, 1044 rows) dominated
the mix and dragged deepset held-out AUC down 0.88 -> 0.76. Fix: CAP each public
source to a comparable size so no single attack style wins, then retrain and re-check
all three held-out AUCs. Seed is small (249) so it is kept whole; deepset and jackhhao
are capped to --cap rows each, stratified by label so attack/benign balance is kept.

Leak guard unchanged: train on each source's TRAIN split, evaluate on its held-out
TEST split.

Usage:  python train_balanced.py --encoder st --cap 500
"""
import os, sys, json, argparse, random
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


def cap_stratified(rows, cap, seed=42):
    if len(rows) <= cap:
        return rows
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(r["label"] != "benign", []).append(r)  # True=attack, False=benign
    out = []
    for grp in by.values():
        rng.shuffle(grp)
        out += grp[: max(1, round(cap * len(grp) / len(rows)))]
    rng.shuffle(out)
    return out


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
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--threshold", type=float, default=0.55)
    args = ap.parse_args()

    seed_tr, seed_te = load_seed()
    ds_tr, ds_te = load_deepset()
    jh_tr, jh_te = load_jackhhao()
    ds_c = cap_stratified(ds_tr, args.cap)
    jh_c = cap_stratified(jh_tr, args.cap)
    combined = [{"text": r["text"], "label": r["label"]} for r in seed_tr] + ds_c + jh_c
    print(f"training rows (capped@{args.cap}): seed={len(seed_tr)} + deepset={len(ds_c)} "
          f"+ jackhhao={len(jh_c)} = {len(combined)}")

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

    print("\nReference AUCs on deepset TEST: 0.88 (deepset-only) -> 0.76 (imbalanced Step 8) -> now?")
    model.save("artifacts/model_balanced.joblib")
    print("Saved: artifacts/model_balanced.joblib")


if __name__ == "__main__":
    main()
