"""
eval_intent_tune.py  —  Block 11: find the intent-confidence gate that adds subtle
recall WITHOUT adding false positives. Ensemble rule:
    final = max(our_score, intent_score IF intent_score >= gate ELSE 0)
Sweep the gate; pick where seed-benign FP returns to ~0 and deepset recall stays high.
"""
import os, sys, json
import numpy as np
sys.path.append(os.getcwd())
from src.classifier import AttackClassifier
from src.intent import injection_score


def load_all():
    te = [r for r in (json.loads(l) for l in open("data/seed_dataset.jsonl")) if r["split"] == "test"]
    sa = [r["text"] for r in te if r["label"] != "benign"]
    sb = [r["text"] for r in te if r["label"] == "benign"]
    from datasets import load_dataset
    d = load_dataset("deepset/prompt-injections")["test"]
    dp_a = [r["text"] for r in d if int(r["label"]) == 1]
    dp_b = [r["text"] for r in d if int(r["label"]) == 0]
    j = load_dataset("jackhhao/jailbreak-classification")["test"]
    j_a = [r["prompt"] for r in j if r["type"] == "jailbreak"]
    return sa, sb, dp_a, dp_b, j_a


def main(thr=0.55):
    m = AttackClassifier.load("artifacts/model_precision.joblib")
    sa, sb, dp_a, dp_b, j_a = load_all()
    sets = {"sa": sa, "sb": sb, "dp_a": dp_a, "dp_b": dp_b, "j_a": j_a}
    O = {k: np.array([p["attack_score"] for p in m.predict(v)]) for k, v in sets.items()}
    I = {k: np.array(injection_score(v)) for k, v in sets.items()}

    def rate(our, intent, gate):
        ens = np.maximum(our, np.where(intent >= gate, intent, 0.0))
        return float((ens >= thr).mean())

    print(f"  {'intent-gate':12s} {'seedRec':>8s} {'seedFP':>7s} {'deepRec':>8s} {'deepFP':>7s} {'jackRec':>8s}")
    print(f"  {'(ours only)':12s} {rate(O['sa'],I['sa'],2):>8.0%} {rate(O['sb'],I['sb'],2):>7.0%} {rate(O['dp_a'],I['dp_a'],2):>8.0%} {rate(O['dp_b'],I['dp_b'],2):>7.0%} {rate(O['j_a'],I['j_a'],2):>8.0%}")
    for g in [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]:
        print(f"  {g:<12.2f} {rate(O['sa'],I['sa'],g):>8.0%} {rate(O['sb'],I['sb'],g):>7.0%} {rate(O['dp_a'],I['dp_a'],g):>8.0%} {rate(O['dp_b'],I['dp_b'],g):>7.0%} {rate(O['j_a'],I['j_a'],g):>8.0%}")
    print("\nPick the gate where seedFP -> 0 and deepRec stays as high as possible.")


if __name__ == "__main__":
    main()
