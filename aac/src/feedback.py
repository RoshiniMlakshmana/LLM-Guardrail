"""
src/feedback.py  —  Block 13 (human-in-the-loop feedback)

When an analyst reviews a detection and confirms/corrects it, that labelled example
is appended here. The next retrain folds it in, so the model learns from the exact
mistakes it makes in the field. Honest 'learning': human-verified data -> retrain,
not magic runtime self-learning.
"""
import os, json, time

PATH = "data/feedback.jsonl"


def record_feedback(text, label, source="analyst", note=""):
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    row = {"text": text, "label": label, "source": source, "note": note,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(PATH, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def load_feedback(path=PATH):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path)]
