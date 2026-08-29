"""
scorecard.py  —  one-shot full detection scorecard across every scan type.
Overall metrics + attack recall by source (in-domain / jailbreak / subtle) +
benign false-positives by source, with the intent raise-to-review column.

Usage:  python scorecard.py            (full, with intent model)
        python scorecard.py --no-intent
"""
import os, sys, json, argparse, random
import numpy as np
sys.path.append(os.getcwd())
from src.classifier import AttackClassifier
from src.calibration import expected_calibration_error, brier_score

GATE = 0.55
INTENT_THR = 0.9


def load():
    rows = []
    for r in (json.loads(l) for l in open("data/seed_dataset.jsonl")):
        if r["split"] == "test":
            rows.append({"text": r["text"], "y": 0 if r["label"] == "benign" else 1, "src": "seed (in-domain)"})
    be = [json.loads(l) for l in open("data/benign_expanded.jsonl")]
    random.Random(42).shuffle(be)
    for r in be[int(len(be) * 0.8):]:
        rows.append({"text": r["text"], "y": 0, "src": "domain benign"})
    try:
        from datasets import load_dataset
        d = load_dataset("deepset/prompt-injections")["test"]
        for r in d:
            if r.get("text"):
                rows.append({"text": r["text"], "y": int(r["label"]), "src": "deepset (subtle)"})
        j = load_dataset("jackhhao/jailbreak-classification")["test"]
        for r in j:
            if r.get("prompt"):
                rows.append({"text": r["prompt"], "y": 1 if r["type"] == "jailbreak" else 0, "src": "jackhhao (jailbreak)"})
    except Exception as e:
        print("[warn] public sets unavailable:", e)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-intent", action="store_true")
    args = ap.parse_args()
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    rows = load(); m = AttackClassifier.load("artifacts/model_precision.joblib")
    texts = [r["text"] for r in rows]; y = np.array([r["y"] for r in rows])
    s = np.array([p["attack_score"] for p in m.predict(texts)])
    intent = None
    if not args.no_intent:
        from src.intent import injection_score
        intent = np.array(injection_score(texts))
    caught = (s >= GATE) | ((intent >= INTENT_THR) if intent is not None else np.zeros(len(s), bool))
    yhat = (s >= GATE).astype(int)

    bar = "=" * 56
    print(bar); print("  DETECTION SCORECARD  (one consolidated test set)"); print(bar)
    print(f"  examples: {len(rows)}   attacks={int(y.sum())}   benign={int((y==0).sum())}\n")
    print("  OVERALL  (classifier alone @0.55)")
    print(f"    ROC-AUC    : {roc_auc_score(y,s):.3f}")
    print(f"    precision  : {precision_score(y,yhat,zero_division=0):.1%}")
    print(f"    recall     : {recall_score(y,yhat,zero_division=0):.1%}")
    print(f"    F1         : {f1_score(y,yhat,zero_division=0):.1%}")
    print(f"    benign FPR : {float((s[y==0]>=GATE).mean()):.1%}")
    print(f"    ECE        : {expected_calibration_error(y,s,10):.3f}")
    print(f"    Brier      : {brier_score(y,s):.3f}")

    print("\n  ATTACK RECALL by type        classifier | +intent->review")
    for src in ["seed (in-domain)", "jackhhao (jailbreak)", "deepset (subtle)"]:
        idx = [i for i, r in enumerate(rows) if r["src"] == src and r["y"] == 1]
        if idx:
            print(f"    {src:24s}  {float((s[idx]>=GATE).mean()):>9.0%} | {float(caught[idx].mean()):>9.0%}")

    print("\n  BENIGN FALSE-POSITIVES by type   blocked | sent-to-review")
    for src in ["seed (in-domain)", "domain benign", "deepset (subtle)", "jackhhao (jailbreak)"]:
        idx = [i for i, r in enumerate(rows) if r["src"] == src and r["y"] == 0]
        if idx:
            print(f"    {src:24s}  {float((s[idx]>=GATE).mean()):>7.0%} | {float(caught[idx].mean()):>13.0%}")
    print(bar)


if __name__ == "__main__":
    main()
