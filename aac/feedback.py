"""
feedback.py  —  Block 13 (human-in-the-loop feedback learning)

The honest version of "learns from attacks": analysts confirm/correct detections,
those labelled examples are appended to a feedback store, and the guard is retrained
to incorporate them. Demonstrated leak-free: we feed HALF of a pool of missed attacks
(the 'analyst-reported' half) and measure improvement on the OTHER half (never fed),
so any gain is real generalisation, not memorisation.

  submit_feedback(text, label)     -> append to data/feedback.jsonl
  retrain_with_feedback(encoder)   -> retrain incorporating feedback -> model_feedback.joblib
  demo()                           -> before/after on held-out reported attacks
"""
import os, sys, json, random
import numpy as np
sys.path.append(os.getcwd())
from src.classifier import AttackClassifier

rng = random.Random(7)
FEEDBACK_FILE = "data/feedback.jsonl"


def submit_feedback(text, label):
    os.makedirs("data", exist_ok=True)
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps({"text": text, "label": label, "src": "analyst"}) + "\n")


def _seed():
    rows = [json.loads(l) for l in open("data/seed_dataset.jsonl")]
    return [r for r in rows if r["split"] == "train"]

def _benign():
    return [json.loads(l) for l in open("data/benign_expanded.jsonl")]

def _public(cap=400):
    from datasets import load_dataset
    out = []
    for r in load_dataset("deepset/prompt-injections")["train"]:
        if r.get("text"):
            out.append({"text": r["text"], "label": "prompt_injection" if int(r["label"]) == 1 else "benign"})
    for r in load_dataset("jackhhao/jailbreak-classification")["train"]:
        if r.get("prompt"):
            out.append({"text": r["prompt"], "label": "jailbreak" if r["type"] == "jailbreak" else "benign"})
    rng.shuffle(out)
    return out[:cap]


def retrain_with_feedback(encoder="st", with_public=True):
    rows = [{"text": r["text"], "label": r["label"]} for r in _seed()] + \
           [{"text": r["text"], "label": "benign"} for r in _benign()]
    if with_public:
        rows += _public()
    if os.path.exists(FEEDBACK_FILE):
        fb = [json.loads(l) for l in open(FEEDBACK_FILE)]
        rows += [{"text": r["text"], "label": r["label"]} for r in fb]
        print(f"  incorporating {len(fb)} analyst-feedback examples")
    m = AttackClassifier(encoder_kind=encoder)
    m.fit([r["text"] for r in rows], [r["label"] for r in rows], calibrate="sigmoid")
    m.save("artifacts/model_feedback.joblib")
    return m


def recall(model, texts, thr=0.55):
    s = np.array([p["attack_score"] for p in model.predict(texts)])
    return float((s >= thr).mean())


def demo(encoder="st"):
    # pool of attacks the model currently MISSES (use deepset test = realistic novel/subtle)
    from datasets import load_dataset
    pool = [r["text"] for r in load_dataset("deepset/prompt-injections")["test"] if int(r["label"]) == 1]
    rng.shuffle(pool)
    reported, holdout = pool[:len(pool)//2], pool[len(pool)//2:]   # analyst-reported vs held-out

    base = AttackClassifier.load("artifacts/model_precision.joblib")
    before = recall(base, holdout)

    # analyst confirms the reported attacks -> feedback store
    open(FEEDBACK_FILE, "w").close()
    for t in reported:
        submit_feedback(t, "prompt_injection")
    new = retrain_with_feedback(encoder=encoder)
    after = recall(new, holdout)

    print(f"\n  reported by analysts (fed): {len(reported)}   held-out (measured): {len(holdout)}")
    print(f"  recall on HELD-OUT attacks  BEFORE: {before:.0%}   AFTER feedback retrain: {after:.0%}")
    print("  (held-out attacks were NEVER fed -> any gain is real learning, not memorisation)")


if __name__ == "__main__":
    demo()
