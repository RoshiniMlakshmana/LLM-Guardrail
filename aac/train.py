"""
train.py  —  trains Block 2 and evaluates with real metrics.
Usage:  python train.py --encoder tfidf      (offline, default)
        python train.py --encoder st         (sentence-transformers, Colab)
Reports per-class P/R/F1, binary attack-vs-benign metrics, ROC-AUC, and the
false-positive rate on benign hard-negatives (the number that proves it isn't
just keyword matching). Saves model + metrics to artifacts/.
"""
import os, sys, json, argparse
import numpy as np
sys.path.append(os.path.dirname(__file__))
from src.classifier import AttackClassifier


def load(path="data/seed_dataset.jsonl"):
    rows = [json.loads(l) for l in open(path)]
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    return tr, te


def evaluate(model, te):
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 roc_auc_score, precision_score, recall_score, f1_score)
    texts = [r["text"] for r in te]
    y_true = [r["label"] for r in te]
    preds = model.predict(texts)
    y_pred = [p["label"] for p in preds]
    scores = [p["attack_score"] for p in preds]

    # multiclass
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    # binary: attack (1) vs benign (0)
    yb_true = [0 if l == "benign" else 1 for l in y_true]
    yb_pred = [0 if l == "benign" else 1 for l in y_pred]
    auc = roc_auc_score(yb_true, scores) if len(set(yb_true)) > 1 else float("nan")

    # false-positive rate on benign hard-negatives at the review threshold 0.70
    benign_scores = [s for s, l in zip(scores, y_true) if l == "benign"]
    fp = sum(1 for s in benign_scores if s >= 0.70)
    fpr = fp / max(1, len(benign_scores))

    metrics = {
        "n_test": len(te),
        "binary_precision": round(precision_score(yb_true, yb_pred, zero_division=0), 4),
        "binary_recall": round(recall_score(yb_true, yb_pred, zero_division=0), 4),
        "binary_f1": round(f1_score(yb_true, yb_pred, zero_division=0), 4),
        "roc_auc": round(auc, 4),
        "benign_false_positive_rate@0.70": round(fpr, 4),
        "per_class_f1": {k: round(v["f1-score"], 3)
                         for k, v in report.items()
                         if k not in ("accuracy", "macro avg", "weighted avg")},
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "accuracy": round(report["accuracy"], 4),
    }
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="tfidf", choices=["tfidf", "st", "auto"])
    args = ap.parse_args()

    if not os.path.exists("data/seed_dataset.jsonl"):
        from data.build_dataset import build; build()
    tr, te = load()
    print(f"Train={len(tr)}  Test={len(te)}  Encoder={args.encoder}")

    model = AttackClassifier(encoder_kind=args.encoder)
    model.fit([r["text"] for r in tr], [r["label"] for r in tr])
    model.save("artifacts/model.joblib")

    metrics = evaluate(model, te)
    os.makedirs("artifacts", exist_ok=True)
    json.dump(metrics, open("artifacts/metrics.json", "w"), indent=2)

    print("\n===== METRICS (held-out test) =====")
    for k, v in metrics.items():
        if k != "per_class_f1":
            print(f"  {k:32s}: {v}")
    print("  per-class F1:")
    for k, v in metrics["per_class_f1"].items():
        print(f"      {k:22s}: {v}")
    print("\nSaved: artifacts/model.joblib, artifacts/metrics.json")


if __name__ == "__main__":
    main()
