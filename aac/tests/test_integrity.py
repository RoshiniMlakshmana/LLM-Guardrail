"""
tests/test_integrity.py
Guards two things that must never silently break:
  1. Every attack class keeps a non-null, verified OWASP-2025 + MITRE ATLAS mapping
     (the whole point of the DFIR/threat-intel angle).
  2. Calibration genuinely reduces miscalibration on the held-out test split.
Run:  python -m pytest tests/ -q     (or)     python tests/test_integrity.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from taxonomy import TAXONOMY, OWASP_LLM_2025, ATLAS_TECHNIQUES, ATTACK_CLASSES

# Verified June 2026 against genai.owasp.org/llm-top-10 and atlas.mitre.org
VERIFIED_OWASP_2025 = {
    "LLM01": "Prompt Injection", "LLM02": "Sensitive Information Disclosure",
    "LLM03": "Supply Chain Vulnerabilities", "LLM04": "Data and Model Poisoning",
    "LLM05": "Improper Output Handling", "LLM06": "Excessive Agency",
    "LLM07": "System Prompt Leakage", "LLM08": "Vector and Embedding Weaknesses",
    "LLM09": "Misinformation", "LLM10": "Unbounded Consumption",
}
VALID_ATLAS = {"AML.T0051", "AML.T0054", "AML.T0056", "AML.T0057",
               "AML.T0024", "AML.T0053"}


def test_owasp_2025_list_matches_authoritative():
    # ids/names must match the 2025 list (NOT the older 2023/24 deck list)
    for k, name in VERIFIED_OWASP_2025.items():
        assert OWASP_LLM_2025.get(k) == name, f"{k} drifted: {OWASP_LLM_2025.get(k)!r}"


def test_every_attack_class_has_framework_mapping():
    for cls in ATTACK_CLASSES:
        e = TAXONOMY[cls]
        assert e["owasp_id"] in OWASP_LLM_2025, f"{cls}: bad OWASP id {e['owasp_id']}"
        assert e["atlas_id"] in ATLAS_TECHNIQUES, f"{cls}: unknown ATLAS id {e['atlas_id']}"
        assert e["severity"] in {"low", "medium", "high"}, f"{cls}: bad severity"
        # full lists must also reference only known framework IDs
        for o in e["owasp_all"]:
            assert o["id"] in OWASP_LLM_2025, f"{cls}: owasp_all bad id {o['id']}"
        for a in e["atlas_all"]:
            assert a["id"] in ATLAS_TECHNIQUES, f"{cls}: atlas_all bad id {a['id']}"
        assert e["owasp_all"] and e["atlas_all"], f"{cls}: empty owasp_all/atlas_all"


def test_calibration_improves_ece():
    """Sigmoid calibration must not worsen ECE on the untouched test split."""
    import numpy as np
    from src.classifier import AttackClassifier
    from src.calibration import expected_calibration_error
    rows = [json.loads(l) for l in open(
        os.path.join(os.path.dirname(__file__), "..", "data", "seed_dataset.jsonl"))]
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    txt, lab = [r["text"] for r in tr], [r["label"] for r in tr]
    y = [0 if r["label"] == "benign" else 1 for r in te]
    raw = AttackClassifier("tfidf").fit(txt, lab)
    cal = AttackClassifier("tfidf").fit(txt, lab, calibrate="sigmoid")
    s_raw = [p["attack_score"] for p in raw.predict([r["text"] for r in te])]
    s_cal = [p["attack_score"] for p in cal.predict([r["text"] for r in te])]
    ece_raw = expected_calibration_error(y, s_raw)
    ece_cal = expected_calibration_error(y, s_cal)
    assert ece_cal <= ece_raw + 1e-6, f"calibration worsened ECE: {ece_raw:.3f} -> {ece_cal:.3f}"


if __name__ == "__main__":
    test_owasp_2025_list_matches_authoritative()
    test_every_attack_class_has_framework_mapping()
    print("OK: OWASP-2025 + ATLAS mapping intact for all attack classes.")
    test_calibration_improves_ece()
    print("OK: calibration does not worsen ECE on the test split.")
    print("All integrity tests passed.")
