"""
eval_intent.py  —  Block 11 measurement

Compares, on each test set, our current detector ALONE vs ENSEMBLED with the intent
model (final = max(our_score, intent_score)). Reports attack recall (want UP on the
subtle deepset set) and benign false-positives (must stay low). Honest: deepset labels
are noisy, so a perfect score isn't expected.

Usage:  python eval_intent.py
"""
import os, sys, json
import numpy as np
sys.path.append(os.getcwd())
from src.classifier import AttackClassifier
from src.intent import injection_score


def seed_sets():
    rows = [json.loads(l) for l in open("data/seed_dataset.jsonl")]
    te = [r for r in rows if r["split"] == "test"]
    return ([r for r in te if r["label"] != "benign"], [r for r in te if r["label"] == "benign"])

def public_sets():
    from datasets import load_dataset
    d = load_dataset("deepset/prompt-injections")["test"]
    dp_a = [r["text"] for r in d if int(r["label"]) == 1]
    dp_b = [r["text"] for r in d if int(r["label"]) == 0]
    j = load_dataset("jackhhao/jailbreak-classification")["test"]
    j_a = [r["prompt"] for r in j if r["type"] == "jailbreak"]
    return dp_a, dp_b, j_a


def scores(model, texts):
    if not texts: return np.array([])
    ours = np.array([p["attack_score"] for p in model.predict(texts)])
    intent = np.array(injection_score(texts))
    return ours, intent


def rate(arr, thr): return float((arr >= thr).mean()) if len(arr) else float("nan")


def main(thr=0.55):
    model = AttackClassifier.load("artifacts/model_precision.joblib")
    sa, sb = seed_sets()
    dp_a, dp_b, j_a = public_sets()

    print(f"{'set':26s} {'metric':16s} {'OURS':>7s} {'+INTENT':>9s}")
    def line(name, metric, texts):
        o, i = scores(model, texts)
        ens = np.maximum(o, i)
        print(f"  {name:24s} {metric:16s} {rate(o,thr):>7.0%} {rate(ens,thr):>9.0%}")
    line("seed-test attacks", "recall", [r["text"] for r in sa])
    line("seed-test benign", "false-pos", [r["text"] for r in sb])
    line("deepset attacks (subtle)", "recall", dp_a)
    line("deepset benign", "false-pos", dp_b)
    line("jackhhao attacks", "recall", j_a)
    print("\nWant: deepset attack recall UP, benign false-pos staying low.")


if __name__ == "__main__":
    main()
