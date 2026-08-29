"""
src/classifier.py  —  Block 2 (the ensemble)
AttackClassifier = heuristic signature features  +  semantic embedding  ->  multiclass
logistic head. Outputs:
  - label        : most likely class (incl. 'benign')
  - attack_score : 1 - P(benign)  (the number the gate uses, calibrated in Block 5)
  - probs        : full per-class probability dict
The heuristic layer is fused as extra features, so the learned model can lean on
signatures when present and on meaning when not.
"""
import os, json
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from taxonomy import ALL_CLASSES, ATTACK_CLASSES
from src.heuristics import heuristic_scan
from src.encoders import get_encoder


def _heuristic_features(texts):
    """Per-text vector: [score, n_hits/3, one-hot(hinted attack class)]."""
    rows = []
    for t in texts:
        n, cls, score = heuristic_scan(t)
        onehot = [1.0 if cls == c else 0.0 for c in ATTACK_CLASSES]
        rows.append([score, min(n, 3) / 3.0] + onehot)
    return np.asarray(rows, dtype=np.float32)


class AttackClassifier:
    def __init__(self, encoder_kind="auto", C=4.0):
        from sklearn.linear_model import LogisticRegression
        self.encoder = get_encoder(encoder_kind)
        self.clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
        self.classes_ = None
        self.calibration_ = None

    def _features(self, texts, fit=False):
        if fit:
            self.encoder.fit(texts)
        emb = self.encoder.transform(texts)
        heur = _heuristic_features(texts)
        return np.hstack([emb, heur])

    def fit(self, texts, labels, calibrate=None, cv=5):
        """
        calibrate: None | 'sigmoid' | 'isotonic'  (Block 5).
        When set, the logistic head is wrapped in CalibratedClassifierCV with
        internal `cv`-fold cross-validation: each fold trains on k-1 parts and
        calibrates on the held-out part, so NO test data and no single fixed
        holdout is needed, and there is no calibration-on-training leakage.
        The encoder (e.g. TF-IDF vocab) is still fit on all train texts, same as
        the uncalibrated baseline -- calibration only reshapes the score, so this
        does not advantage the calibrated model on the untouched test set.
        """
        X = self._features(texts, fit=True)
        if calibrate is None:
            self.clf.fit(X, labels)
            self.classes_ = list(self.clf.classes_)
        else:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression
            base = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
            order = sorted(set(labels))
            n_min = int(min(np.bincount([order.index(l) for l in labels])))
            folds = max(2, min(cv, n_min))  # never more folds than rarest class samples
            self.clf = CalibratedClassifierCV(base, method=calibrate, cv=folds)
            self.clf.fit(X, labels)
            self.classes_ = list(self.clf.classes_)
            self.calibration_ = {"method": calibrate, "cv": folds}
        return self

    def predict_proba(self, texts):
        X = self._features(texts, fit=False)
        P = self.clf.predict_proba(X)
        return [{c: float(p) for c, p in zip(self.classes_, row)} for row in P]

    def predict(self, texts):
        """Return {label, attack_score, probs}; floor score with high-precision signatures."""
        from src.heuristics import heuristic_scan
        out = []
        plist = self.predict_proba(texts)
        for text, probs in zip(texts, plist):
            label = max(probs, key=probs.get)
            attack_score = 1.0 - probs.get("benign", 0.0)
            n, hcls, hscore = heuristic_scan(text)      # signature floor (defense in depth)
            if n > 0:
                attack_score = max(attack_score, hscore)
                if label == "benign" and hcls:
                    label = hcls
            out.append({"label": label, "attack_score": round(attack_score, 4),
                        "probs": {k: round(v, 4) for k, v in probs.items()}})
        return out

    def save(self, path="artifacts/model.joblib"):
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"encoder": self.encoder, "clf": self.clf,
                     "classes_": self.classes_,
                     "calibration_": self.calibration_}, path)
        return path

    @classmethod
    def load(cls, path="artifacts/model.joblib"):
        import joblib
        obj = cls.__new__(cls)
        d = joblib.load(path)
        obj.encoder, obj.clf, obj.classes_ = d["encoder"], d["clf"], d["classes_"]
        obj.calibration_ = d.get("calibration_")
        return obj
