"""
src/intent.py  —  Block 11 (intent detector for subtle attacks)

Signatures catch obvious attacks; the ML head catches in-distribution ones. Subtle
attacks (persona/opinion manipulation, novel phrasings) need INTENT understanding.
This wraps a transformer purpose-trained on prompt injection and exposes P(attack),
to be ensembled with the existing score. Lazy-loaded so the rest of the system runs
without it.
"""
_PIPE = None
_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"


def _get_pipe():
    global _PIPE
    if _PIPE is None:
        from transformers import pipeline
        _PIPE = pipeline("text-classification", model=_MODEL,
                         truncation=True, max_length=512)
    return _PIPE


def injection_score(texts):
    """Return P(injection/attack) in [0,1] for each text."""
    pipe = _get_pipe()
    out = pipe(list(texts))
    scores = []
    for o in out:
        lab = o["label"].upper()
        p = float(o["score"])
        scores.append(p if lab.startswith("INJECT") else 1.0 - p)
    return scores
