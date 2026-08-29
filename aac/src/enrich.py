"""
src/enrich.py  —  Block 6 (framework enrichment + DFIR record)
Converts a prediction into an analyst-ready finding mapped to OWASP LLM Top-10
and MITRE ATLAS, and emits a forensic record for the DFIR timeline.
"""
import os, sys, json, time, hashlib
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from taxonomy import enrich as tax_enrich


def make_finding(text, prediction, gate_decision, session_id="sess-001",
                 turn=1, cumulative_score=None):
    label = prediction["label"]
    tax = tax_enrich(label)
    ioc = "promptsig:" + hashlib.sha256(text.lower().strip().encode()).hexdigest()[:16]
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id": session_id,
        "turn": turn,
        "input_excerpt": text[:160],
        "prediction": label,
        "attack_score": prediction["attack_score"],
        "cumulative_score": cumulative_score,
        "is_attack": tax["is_attack"],
        "owasp": None if not tax["owasp_id"] else f'{tax["owasp_id"]} {tax["owasp_name"]}',
        "owasp_all": [f'{o["id"]} {o["name"]}' for o in tax.get("owasp_all", [])],
        "mitre_atlas": None if not tax["atlas_id"] else f'{tax["atlas_id"]} {tax["atlas_technique"]}',
        "mitre_atlas_all": [f'{a["id"]} {a["name"]}' for a in tax.get("atlas_all", [])],
        "atlas_tactic": tax["atlas_tactic"],
        "severity": tax["severity"],
        "action": gate_decision["action"],
        "confidence_tier": gate_decision["confidence_tier"],
        "ioc": ioc,                         # stable signature id for TI / campaign tracking
    }


def log_finding(finding, path="artifacts/dfir_log.jsonl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(finding) + "\n")
    return path
