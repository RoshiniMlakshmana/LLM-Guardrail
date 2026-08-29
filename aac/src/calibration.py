"""
src/calibration.py  —  Block 5 (calibration)

The gate (src/gate.py) turns a single scalar `attack_score = 1 - P(benign)` into
log_only / human_review / auto_block using fixed thresholds (0.70 / 0.95 / 0.98).
Those thresholds are only meaningful if the score is CALIBRATED: i.e. among inputs
scored ~0.70, roughly 70% really are attacks. A raw logistic-head probability is
usually NOT calibrated, so this module measures and fixes that.

What's here:
  - expected_calibration_error(): equal-width-bin ECE on the binary attack score.
  - brier_score(): proper scoring rule, robust on small test sets where ECE is noisy.
  - reliability_curve(): per-bin (confidence, empirical_accuracy, count).
  - reliability_diagram(): saves the raw-vs-calibrated reliability plot.

No leakage: the calibrator is fit with internal cross-validation on the TRAIN set
only (see AttackClassifier(calibrate=...) and calibrate.py). ECE/Brier are reported
on the untouched TEST set.
"""
from __future__ import annotations
import numpy as np


def _as_arrays(y_true, scores):
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(scores, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y{y.shape} vs scores{p.shape}")
    return y, p


def reliability_curve(y_true, scores, n_bins: int = 10):
    """
    Equal-width bins over [0,1]. Returns a list of dicts per non-empty bin:
      {bin_lo, bin_hi, count, mean_score (confidence), frac_positive (accuracy)}.
    'accuracy' here = fraction of true attacks in the bin (binary positive rate).
    """
    y, p = _as_arrays(y_true, scores)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin closed on the right so score==1.0 lands somewhere
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        out.append({
            "bin_lo": float(lo), "bin_hi": float(hi), "count": n,
            "mean_score": float(p[mask].mean()),
            "frac_positive": float(y[mask].mean()),
        })
    return out


def expected_calibration_error(y_true, scores, n_bins: int = 10) -> float:
    """
    ECE = sum_b (n_b / N) * | mean_score_b - frac_positive_b |.
    Lower is better; 0 = perfectly calibrated. Reported with bin counts because
    on small test sets a few sparse bins dominate and the number is noisy.
    """
    y, p = _as_arrays(y_true, scores)
    N = len(y)
    if N == 0:
        return float("nan")
    ece = 0.0
    for b in reliability_curve(y, p, n_bins=n_bins):
        ece += (b["count"] / N) * abs(b["mean_score"] - b["frac_positive"])
    return float(ece)


def brier_score(y_true, scores) -> float:
    """Mean squared error between score and binary outcome. Proper scoring rule."""
    y, p = _as_arrays(y_true, scores)
    return float(np.mean((p - y) ** 2))


def reliability_diagram(y_true, score_sets: dict, out_path: str,
                        n_bins: int = 10, title: str = "Reliability diagram"):
    """
    score_sets: {label: scores_array}, e.g. {"raw (uncalibrated)": ..., "calibrated": ...}.
    Saves a PNG: each series as binned points vs the y=x perfect-calibration line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="perfect calibration")
    for label, scores in score_sets.items():
        curve = reliability_curve(y_true, scores, n_bins=n_bins)
        xs = [b["mean_score"] for b in curve]
        ys = [b["frac_positive"] for b in curve]
        sizes = [18 + 4 * b["count"] for b in curve]
        ece = expected_calibration_error(y_true, scores, n_bins)
        ax.plot(xs, ys, "-o", lw=1.4, markersize=4, label=f"{label} (ECE={ece:.3f})")
        ax.scatter(xs, ys, s=sizes, alpha=0.25)
    ax.set_xlabel("Predicted attack score (confidence)")
    ax.set_ylabel("Empirical fraction of attacks")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
