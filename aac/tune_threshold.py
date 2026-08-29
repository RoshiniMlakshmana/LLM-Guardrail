"""
tune_threshold.py  —  Step 7 (v2): goal-aligned operating points, per gate tier.

Why v2: the generic "best balance" picker ignored the actual priority (controlling
false alarms on OUR security-themed benign). v2 picks thresholds against an explicit
FALSE-ALARM BUDGET measured on the in-house benign, then takes the most attacks under
that budget — and emits TWO points to match the gate's tiers:
  - strict (critical assets): budget 0.30  -> catch more, tolerate some false alarms
  - lenient (workstation):    budget 0.10  -> quieter, fewer interruptions

Honest limits: only ~68 in-house benign exist, so the in-house false-alarm estimate
is coarse and the chosen numbers are approximate. The durable fix is MORE benign
hard-negatives. We also report on the held-out tests so nothing is judged on the
data used to pick.

Usage:  python tune_threshold.py --encoder st
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
    def conv(sp):
        return [{"text": r["text"],
                 "label": ("prompt_injection" if int(r["label"]) == 1 else "benign")}
                for r in ds[sp] if r.get("text")]
    return conv("train"), conv("test")


def predict_scores(model, rows):
    return np.array([p["attack_score"] for p in model.predict([r["text"] for r in rows])])


def recall_at(scores, y, t):
    yhat = (scores >= t).astype(int)
    from sklearn.metrics import recall_score
    return recall_score(y, yhat, zero_division=0)


def fpr_at(scores_benign, t):
    return float((scores_benign >= t).mean()) if len(scores_benign) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="st")
    args = ap.parse_args()
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    seed_tr, seed_te = load_seed()
    pub_tr, pub_te = load_deepset()
    # tag source so we can measure false alarms on IN-HOUSE benign specifically
    comb = ([{"text": r["text"], "label": r["label"], "src": "inhouse"} for r in seed_tr]
            + [{"text": r["text"], "label": r["label"], "src": "public"} for r in pub_tr])
    yb = [0 if r["label"] == "benign" else 1 for r in comb]
    fit, val = train_test_split(comb, test_size=0.25, stratify=yb, random_state=42)

    model = AttackClassifier(encoder_kind=args.encoder)
    model.fit([r["text"] for r in fit], [r["label"] for r in fit], calibrate="sigmoid")

    sv = predict_scores(model, val)
    yv = np.array([0 if r["label"] == "benign" else 1 for r in val])
    inhouse_benign = sv[np.array([r["src"] == "inhouse" and r["label"] == "benign" for r in val])]
    print(f"validation: n={len(val)}  in-house benign used for false-alarm budget = {len(inhouse_benign)}")

    print("\nsweep on VALIDATION (recall over all attacks; FPR on IN-HOUSE benign only):")
    print(f"  {'thr':>5} {'recall':>8} {'inhouse_fpr':>12}")
    grid = []
    for t in np.round(np.arange(0.40, 0.96, 0.05), 2):
        r = recall_at(sv, yv, t)
        f = fpr_at(inhouse_benign, t)
        grid.append((float(t), r, f))
        print(f"  {t:>5} {r:>8.3f} {f:>12.3f}")

    def pick(budget):
        ok = [(t, r, f) for t, r, f in grid if not np.isnan(f) and f <= budget]
        if ok:
            return max(ok, key=lambda x: x[1])[0]      # most attacks under the budget
        return min(grid, key=lambda x: x[2])[0]         # fallback: quietest

    strict_t, lenient_t = pick(0.30), pick(0.10)
    print(f"\n  strict  (critical, FPR<=0.30) -> threshold {strict_t}")
    print(f"  lenient (workstation, FPR<=0.10) -> threshold {lenient_t}")

    print("\n===== FINAL on held-out tests =====")
    se_s = predict_scores(model, seed_te)
    se_y = np.array([0 if r["label"] == "benign" else 1 for r in seed_te])
    se_benign = se_s[se_y == 0]
    pe_s = predict_scores(model, pub_te)
    pe_y = np.array([0 if r["label"] == "benign" else 1 for r in pub_te])
    print(f"  in-house seed TEST AUC={roc_auc_score(se_y, se_s):.3f} | "
          f"external public TEST AUC={roc_auc_score(pe_y, pe_s):.3f}")
    for name, t in [("strict", strict_t), ("lenient", lenient_t)]:
        print(f"\n  [{name} @ {t}]")
        print(f"    in-house false alarms : {fpr_at(se_benign, t):.3f}")
        print(f"    external attack recall: {recall_at(pe_s, pe_y, t):.3f}")

    model.save("artifacts/model_mixed_calibrated.joblib")
    json.dump({"strict": strict_t, "lenient": lenient_t},
              open("artifacts/operating_threshold.json", "w"), indent=2)
    print("\nSaved: artifacts/model_mixed_calibrated.joblib + operating_threshold.json")
    print("Caveat: ~%d in-house benign drove the budget -> numbers are approximate;"
          % len(inhouse_benign))
    print("        collecting more security-themed benign is the real lever.")


if __name__ == "__main__":
    main()
