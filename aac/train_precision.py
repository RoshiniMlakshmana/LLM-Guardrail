"""
train_precision.py  —  Block 10 (precision uplift, combined)

Best-of-both retrain: seed attacks + the 247 new DOMAIN benign + capped PUBLIC attacks
(deepset + jackhhao). Domain benign cuts false positives; public attacks keep recall
and external coverage. Calibrated. Honest eval on held-out sets.

Usage:  python train_precision.py --encoder st            (full, with public data)
        python train_precision.py --encoder st --no-public (seed + new benign only)
"""
import os, sys, json, argparse, random
import numpy as np
sys.path.append(os.getcwd())
from src.classifier import AttackClassifier
rng = random.Random(42)


def load_seed(path="data/seed_dataset.jsonl"):
    rows = [json.loads(l) for l in open(path)]
    return ([r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "test"])

def load_benign(path="data/benign_expanded.jsonl"):
    rows = [json.loads(l) for l in open(path)]; rng.shuffle(rows)
    k = int(len(rows) * 0.8); return rows[:k], rows[k:]

def cap_strat(rows, cap):
    if len(rows) <= cap: return rows
    by = {}
    for r in rows: by.setdefault(r["label"] != "benign", []).append(r)
    out = []
    for g in by.values():
        rng.shuffle(g); out += g[:max(1, round(cap * len(g) / len(rows)))]
    return out

def load_public(cap=400):
    from datasets import load_dataset
    ds = load_dataset("deepset/prompt-injections")
    dp = [{"text": r["text"], "label": ("prompt_injection" if int(r["label"]) == 1 else "benign")}
          for sp in ("train",) for r in ds[sp] if r.get("text")]
    dp_te = [{"text": r["text"], "label": ("prompt_injection" if int(r["label"]) == 1 else "benign")}
             for r in ds["test"] if r.get("text")]
    jb = load_dataset("jackhhao/jailbreak-classification")
    jp = [{"text": r["prompt"], "label": ("jailbreak" if r["type"] == "jailbreak" else "benign")}
          for r in jb["train"] if r.get("prompt")]
    jp_te = [{"text": r["prompt"], "label": ("jailbreak" if r["type"] == "jailbreak" else "benign")}
             for r in jb["test"] if r.get("prompt")]
    return cap_strat(dp, cap), cap_strat(jp, cap), dp_te, jp_te

def fpr(model, rows, thr):
    if not rows: return float("nan")
    s = np.array([p["attack_score"] for p in model.predict([r["text"] for r in rows])])
    return float((s >= thr).mean())

def recall(model, rows, thr):
    atk = [r for r in rows if r["label"] != "benign"]
    if not atk: return float("nan")
    s = np.array([p["attack_score"] for p in model.predict([r["text"] for r in atk])])
    return float((s >= thr).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="st"); ap.add_argument("--thr", type=float, default=0.55)
    ap.add_argument("--no-public", action="store_true")
    args = ap.parse_args()

    seed_tr, seed_te = load_seed(); be_tr, be_te = load_benign()
    seed_ben_te = [r for r in seed_te if r["label"] == "benign"]
    combined = [{"text": r["text"], "label": r["label"]} for r in seed_tr] + \
               [{"text": r["text"], "label": "benign"} for r in be_tr]
    try:
        from src.feedback import load_feedback
        fb = load_feedback()
        if fb:
            combined += [{"text": r["text"], "label": r["label"]} for r in fb]
            print(f"folded in {len(fb)} analyst feedback examples")
    except Exception:
        pass
    dp_te = jp_te = []
    if not args.no_public:
        dp, jp, dp_te, jp_te = load_public()
        combined += dp + jp
        print(f"train: seed={len(seed_tr)} + new-benign={len(be_tr)} + deepset={len(dp)} + jackhhao={len(jp)} = {len(combined)}")
    else:
        print(f"train: seed={len(seed_tr)} + new-benign={len(be_tr)} = {len(combined)}")

    new = AttackClassifier(encoder_kind=args.encoder)
    new.fit([r["text"] for r in combined], [r["label"] for r in combined], calibrate="sigmoid")
    old = AttackClassifier.load("artifacts/model_balanced.joblib") if os.path.exists("artifacts/model_balanced.joblib") else None

    print(f"\n===== before vs after (thr {args.thr}) =====")
    print(f"  {'metric':38s} {'OLD':>8s} {'NEW':>8s}")
    def row(name, rows, fn):
        o = fn(old, rows, args.thr) if old else float('nan')
        n = fn(new, rows, args.thr)
        print(f"  {name:38s} {o:>8.1%} {n:>8.1%}")
    row("FP @ new-domain benign", be_te, fpr)
    row("FP @ seed-test benign", seed_ben_te, fpr)
    row("recall @ seed-test attacks", seed_te, recall)
    if dp_te: row("recall @ deepset-test attacks", dp_te, recall)
    if jp_te: row("recall @ jackhhao-test attacks", jp_te, recall)

    new.save("artifacts/model_precision.joblib")
    print("\nSaved: artifacts/model_precision.joblib")

if __name__ == "__main__":
    main()
