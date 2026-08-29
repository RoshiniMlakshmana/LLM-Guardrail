"""
eval_compare.py  —  full BEFORE vs AFTER scorecard on the consolidated test set.

BEFORE = pre-precision model (artifacts/model_balanced.joblib, the over-flagging one)
AFTER  = current model (artifacts/model_precision.joblib)
AFTER+intent = current model with the intent model routing subtle attacks to review.

Reports overall detection metrics, false-alarms by domain, and attack recall by source,
so you can see exactly how each number improved.

Usage:  python eval_compare.py            (with intent column)
        python eval_compare.py --no-intent
"""
import os, sys, json, random, argparse
import numpy as np
sys.path.append(os.getcwd())
from src.classifier import AttackClassifier
from src.calibration import expected_calibration_error, brier_score

rng = random.Random(42)
GATE = 0.55
INTENT_THR = 0.9


def build_testset():
    rows = []
    for r in (json.loads(l) for l in open("data/seed_dataset.jsonl")):
        if r["split"] == "test":
            rows.append({"t": r["text"], "y": 0 if r["label"] == "benign" else 1, "src": "seed"})
    be = [json.loads(l) for l in open("data/benign_expanded.jsonl")]; rng.shuffle(be)
    for r in be[int(len(be) * 0.8):]:
        rows.append({"t": r["text"], "y": 0, "src": "domain_benign"})
    try:
        from datasets import load_dataset
        for r in load_dataset("deepset/prompt-injections")["test"]:
            if r.get("text"): rows.append({"t": r["text"], "y": int(r["label"]), "src": "deepset"})
        for r in load_dataset("jackhhao/jailbreak-classification")["test"]:
            if r.get("prompt"): rows.append({"t": r["prompt"], "y": 1 if r["type"] == "jailbreak" else 0, "src": "jackhhao"})
    except Exception as e:
        print("[warn] public sets unavailable:", e)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-intent", action="store_true"); args = ap.parse_args()
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    rows = build_testset()
    texts = [r["t"] for r in rows]; y = np.array([r["y"] for r in rows])
    old = AttackClassifier.load("artifacts/model_balanced.joblib")
    new = AttackClassifier.load("artifacts/model_precision.joblib")
    so = np.array([p["attack_score"] for p in old.predict(texts)])
    sn = np.array([p["attack_score"] for p in new.predict(texts)])
    use_intent = not args.no_intent
    si = None
    if use_intent:
        try:
            from src.intent import injection_score
            si = np.array(injection_score(texts))
        except Exception as e:
            print("[warn] intent unavailable:", e); use_intent = False
    # AFTER+intent "flagged" = our score OR intent escalation (-> review)
    flagged = (sn >= GATE) | (si >= INTENT_THR) if use_intent else None

    def f(arr): return f"{arr:>8.1%}"
    print(f"consolidated test set: {len(rows)} (attacks={int(y.sum())}, benign={int((y==0).sum())})\n")
    hdr = f"  {'overall metric':24s} {'BEFORE':>8s} {'AFTER':>8s}" + (f" {'+INTENT':>8s}" if use_intent else "")
    print(hdr)
    def show(name, ob, nb, ib=None):
        line = f"  {name:24s} {f(ob)} {f(nb)}"
        if use_intent and ib is not None: line += f" {f(ib)}"
        print(line)
    show("precision", precision_score(y,(so>=GATE),zero_division=0), precision_score(y,(sn>=GATE),zero_division=0),
         precision_score(y, flagged, zero_division=0) if use_intent else None)
    show("recall", recall_score(y,(so>=GATE),zero_division=0), recall_score(y,(sn>=GATE),zero_division=0),
         recall_score(y, flagged, zero_division=0) if use_intent else None)
    show("F1", f1_score(y,(so>=GATE),zero_division=0), f1_score(y,(sn>=GATE),zero_division=0),
         f1_score(y, flagged, zero_division=0) if use_intent else None)
    print(f"  {'ROC-AUC':24s} {roc_auc_score(y,so):>8.3f} {roc_auc_score(y,sn):>8.3f}")
    show("benign false-alarm rate", float((so[y==0]>=GATE).mean()), float((sn[y==0]>=GATE).mean()),
         float((flagged[y==0]).mean()) if use_intent else None)
    print(f"  {'ECE (calibration)':24s} {expected_calibration_error(y,so):>8.3f} {expected_calibration_error(y,sn):>8.3f}")
    print(f"  {'Brier':24s} {brier_score(y,so):>8.3f} {brier_score(y,sn):>8.3f}")

    print("\n  benign false-alarms by source:        BEFORE    AFTER" + ("   +INTENT" if use_intent else ""))
    for src in ["seed","domain_benign","deepset","jackhhao"]:
        idx = np.array([r["src"]==src and r["y"]==0 for r in rows])
        if idx.any():
            line = f"    {src:24s} {f(float((so[idx]>=GATE).mean()))} {f(float((sn[idx]>=GATE).mean()))}"
            if use_intent: line += f" {f(float((flagged[idx]).mean()))}"
            print(line)

    print("\n  attack recall by source:              BEFORE    AFTER" + ("   +INTENT" if use_intent else ""))
    for src in ["seed","deepset","jackhhao"]:
        idx = np.array([r["src"]==src and r["y"]==1 for r in rows])
        if idx.any():
            line = f"    {src:24s} {f(float((so[idx]>=GATE).mean()))} {f(float((sn[idx]>=GATE).mean()))}"
            if use_intent: line += f" {f(float((flagged[idx]).mean()))}"
            print(line)
    print("\n  (+INTENT routes subtle attacks to human_review; benign there is reviewed, never blocked.)")


if __name__ == "__main__":
    main()
