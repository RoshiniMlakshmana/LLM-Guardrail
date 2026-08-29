"""
calibrate.py  —  Block 5 (calibration) runner

Trains two models on the SAME leak-free train split:
  1. raw        : current uncalibrated logistic head (the baseline gate score).
  2. calibrated : same head wrapped in CalibratedClassifierCV (internal CV).

Then, on the UNTOUCHED test split, reports for the binary attack score
(attack_score = 1 - P(benign)):
  - ECE   (expected calibration error, equal-width bins)
  - Brier (proper scoring rule, less noisy on small test sets)
and saves a raw-vs-calibrated reliability diagram + calibration.json.

Why this matters: the gate (src/gate.py) acts at fixed thresholds (0.70 / 0.95 /
0.98). Those are only honest if a 0.70 score really means ~70% chance of attack.
Calibration is what makes the confidence-gated response defensible, not arbitrary.

Usage:
    python calibrate.py --encoder tfidf --method sigmoid
    python calibrate.py --encoder st   --method isotonic
"""
import os, sys, json, argparse
import numpy as np

sys.path.append(os.path.dirname(__file__))
from src.classifier import AttackClassifier
from src.calibration import (expected_calibration_error, brier_score,
                             reliability_curve, reliability_diagram)


def load(path="data/seed_dataset.jsonl"):
    rows = [json.loads(l) for l in open(path)]
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    return tr, te


def attack_scores(model, rows):
    """Binary y (1=attack, 0=benign) and the gate's attack_score for each row."""
    texts = [r["text"] for r in rows]
    y = np.array([0 if r["label"] == "benign" else 1 for r in rows], dtype=float)
    s = np.array([p["attack_score"] for p in model.predict(texts)], dtype=float)
    return y, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="tfidf", choices=["tfidf", "st", "auto"])
    ap.add_argument("--method", default="sigmoid", choices=["sigmoid", "isotonic"])
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--data", default="data/seed_dataset.jsonl")
    args = ap.parse_args()

    tr, te = load(args.data)
    tr_texts = [r["text"] for r in tr]
    tr_labels = [r["label"] for r in tr]
    print(f"Train={len(tr)}  Test={len(te)}  Encoder={args.encoder}  Method={args.method}")

    # 1. raw (uncalibrated) baseline
    raw = AttackClassifier(encoder_kind=args.encoder).fit(tr_texts, tr_labels)
    # 2. calibrated head, internal CV (no leakage, no test data touched)
    cal = AttackClassifier(encoder_kind=args.encoder).fit(
        tr_texts, tr_labels, calibrate=args.method)
    print(f"Calibrator: {getattr(cal, 'calibration_', {})}")

    y_raw, s_raw = attack_scores(raw, te)
    y_cal, s_cal = attack_scores(cal, te)
    assert np.array_equal(y_raw, y_cal)
    y = y_raw

    ece_raw = expected_calibration_error(y, s_raw, args.bins)
    ece_cal = expected_calibration_error(y, s_cal, args.bins)
    brier_raw = brier_score(y, s_raw)
    brier_cal = brier_score(y, s_cal)

    print("\n===== CALIBRATION (untouched test split) =====")
    print(f"  {'metric':<10}{'raw':>10}{'calibrated':>14}{'change':>12}")
    for name, a, b in [("ECE", ece_raw, ece_cal), ("Brier", brier_raw, brier_cal)]:
        delta = b - a
        arrow = "better" if delta < 0 else "worse "
        print(f"  {name:<10}{a:>10.4f}{b:>14.4f}{delta:>+11.4f} {arrow}")

    # reliability diagram
    os.makedirs("artifacts", exist_ok=True)
    diagram = reliability_diagram(
        y, {"raw (uncalibrated)": s_raw, f"calibrated ({args.method})": s_cal},
        out_path="artifacts/reliability_diagram.png", n_bins=args.bins,
        title=f"Reliability — attack score ({args.encoder}, n_test={len(te)})")
    print(f"\nReliability diagram -> {diagram}")

    out = {
        "encoder": args.encoder, "method": args.method, "n_bins": args.bins,
        "n_test": len(te), "n_train": len(tr),
        "calibrator": getattr(cal, "calibration_", {}),
        "ece_raw": round(ece_raw, 4), "ece_calibrated": round(ece_cal, 4),
        "brier_raw": round(brier_raw, 4), "brier_calibrated": round(brier_cal, 4),
        "reliability_bins_calibrated": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()}
            for b in reliability_curve(y, s_cal, args.bins)],
    }
    with open("artifacts/calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: artifacts/calibration.json")

    # persist the calibrated model so the gate can consume calibrated scores
    cal.save("artifacts/model_calibrated.joblib")
    print("Saved: artifacts/model_calibrated.joblib  (gate-facing, calibrated)")

    # honest footnote
    print("\nNOTE: test split is small (n=%d, benign~%d). ECE over %d bins is noisy;"
          % (len(te), int((y == 0).sum()), args.bins))
    print("      Brier is the more stable summary. Treat absolute values as indicative.")


if __name__ == "__main__":
    main()
