"""
taxonomy.py  —  Block 1 (v2: full OWASP LLM Top-10 2025 + MITRE ATLAS coverage)

Every detection class is mapped to its PRIMARY and RELATED OWASP LLM Top-10 (2025)
risks AND its primary + related MITRE ATLAS techniques (incl. sub-techniques).
The enrichment layer (Block 6) surfaces the full mapping on every finding.

Sources (verified June 2026):
  - OWASP Top 10 for LLM Applications 2025: https://genai.owasp.org/llm-top-10/
  - MITRE ATLAS techniques: https://atlas.mitre.org
"""

# ---- Full OWASP LLM Top-10 (2025) ----
OWASP_LLM_2025 = {
    "LLM01": "Prompt Injection",
    "LLM02": "Sensitive Information Disclosure",
    "LLM03": "Supply Chain Vulnerabilities",
    "LLM04": "Data and Model Poisoning",
    "LLM05": "Improper Output Handling",
    "LLM06": "Excessive Agency",
    "LLM07": "System Prompt Leakage",
    "LLM08": "Vector and Embedding Weaknesses",
    "LLM09": "Misinformation",
    "LLM10": "Unbounded Consumption",
}

# ---- MITRE ATLAS technique reference (the AI-attack-relevant set we map to) ----
ATLAS_TECHNIQUES = {
    "AML.T0051": "LLM Prompt Injection",
    "AML.T0051.000": "LLM Prompt Injection: Direct",
    "AML.T0051.001": "LLM Prompt Injection: Indirect",
    "AML.T0054": "LLM Jailbreak",
    "AML.T0056": "LLM Meta Prompt Extraction",
    "AML.T0057": "LLM Data Leakage",
    "AML.T0024": "Exfiltration via ML Inference API",
    "AML.T0024.000": "Infer Training Data Membership",
    "AML.T0024.001": "Invert ML Model",
    "AML.T0024.002": "Extract ML Model",
    "AML.T0053": "LLM Plugin Compromise",
}


def _owasp(*ids):
    return [{"id": i, "name": OWASP_LLM_2025[i]} for i in ids]

def _atlas(*ids):
    return [{"id": i, "name": ATLAS_TECHNIQUES[i]} for i in ids]


TAXONOMY = {
    "benign": {
        "is_attack": False,
        "description": "Legitimate input, including hard-negatives that merely discuss attacks.",
        "owasp_id": None, "owasp_name": None, "owasp_all": [],
        "atlas_tactic": None, "atlas_technique": None, "atlas_id": None, "atlas_all": [],
        "severity": "none",
    },
    "prompt_injection": {
        "is_attack": True,
        "description": "Direct attempt to override or replace the model's instructions.",
        "owasp_id": "LLM01", "owasp_name": "Prompt Injection",
        "owasp_all": _owasp("LLM01", "LLM05"),
        "atlas_tactic": "Initial Access / ML Attack Staging",
        "atlas_technique": "LLM Prompt Injection", "atlas_id": "AML.T0051",
        "atlas_all": _atlas("AML.T0051", "AML.T0051.000"),
        "severity": "high",
    },
    "jailbreak": {
        "is_attack": True,
        "description": "Attempt to remove guardrails so the model responds without restriction.",
        "owasp_id": "LLM01", "owasp_name": "Prompt Injection",
        "owasp_all": _owasp("LLM01", "LLM09"),
        "atlas_tactic": "Defense Evasion / Privilege Escalation",
        "atlas_technique": "LLM Jailbreak", "atlas_id": "AML.T0054",
        "atlas_all": _atlas("AML.T0054", "AML.T0051"),
        "severity": "high",
    },
    "system_prompt_exfil": {
        "is_attack": True,
        "description": "Attempt to extract the hidden system prompt / instructions.",
        "owasp_id": "LLM07", "owasp_name": "System Prompt Leakage",
        "owasp_all": _owasp("LLM07", "LLM01"),
        "atlas_tactic": "Discovery / Exfiltration",
        "atlas_technique": "LLM Meta Prompt Extraction", "atlas_id": "AML.T0056",
        "atlas_all": _atlas("AML.T0056", "AML.T0057"),
        "severity": "medium",
    },
    "sensitive_data_probe": {
        "is_attack": True,
        "description": "Attempt to coax training data, secrets, or PII out of the model.",
        "owasp_id": "LLM02", "owasp_name": "Sensitive Information Disclosure",
        "owasp_all": _owasp("LLM02"),
        "atlas_tactic": "Exfiltration / Collection",
        "atlas_technique": "LLM Data Leakage", "atlas_id": "AML.T0057",
        "atlas_all": _atlas("AML.T0057", "AML.T0024.000"),
        "severity": "high",
    },
    "model_extraction": {
        "is_attack": True,
        "description": "Membership-inference / model-inversion / model-theft via the inference API.",
        "owasp_id": "LLM10", "owasp_name": "Unbounded Consumption",
        "owasp_all": _owasp("LLM10", "LLM02"),
        "atlas_tactic": "Exfiltration",
        "atlas_technique": "Exfiltration via ML Inference API", "atlas_id": "AML.T0024",
        "atlas_all": _atlas("AML.T0024", "AML.T0024.002", "AML.T0024.001", "AML.T0024.000"),
        "severity": "medium",
    },
    "indirect_injection": {
        "is_attack": True,
        "description": "Malicious instructions smuggled through retrieved/external content (RAG).",
        "owasp_id": "LLM01", "owasp_name": "Prompt Injection",
        "owasp_all": _owasp("LLM01", "LLM08"),
        "atlas_tactic": "Initial Access",
        "atlas_technique": "LLM Prompt Injection: Indirect", "atlas_id": "AML.T0051.001",
        "atlas_all": _atlas("AML.T0051.001", "AML.T0051"),
        "severity": "high",
    },
    "excessive_agency": {
        "is_attack": True,
        "description": "Coercing the model/agent into unauthorized tool, plugin, or action use.",
        "owasp_id": "LLM06", "owasp_name": "Excessive Agency",
        "owasp_all": _owasp("LLM06"),
        "atlas_tactic": "Execution / Impact",
        "atlas_technique": "LLM Plugin Compromise", "atlas_id": "AML.T0053",
        "atlas_all": _atlas("AML.T0053"),
        "severity": "high",
    },
}

ATTACK_CLASSES = [c for c, v in TAXONOMY.items() if v["is_attack"]]
ALL_CLASSES = list(TAXONOMY.keys())


def enrich(label: str) -> dict:
    return TAXONOMY.get(label, TAXONOMY["benign"])


if __name__ == "__main__":
    import json
    print("OWASP LLM Top-10 (2025):")
    for k, v in OWASP_LLM_2025.items():
        print(f"  {k}: {v}")
    print(f"\nATLAS techniques mapped: {len(ATLAS_TECHNIQUES)}")
    print(f"Detection classes: {ALL_CLASSES}")
    print("\nExample full enrichment for 'model_extraction':")
    e = enrich("model_extraction")
    print("  OWASP:", [f'{o["id"]} {o["name"]}' for o in e["owasp_all"]])
    print("  ATLAS:", [f'{a["id"]} {a["name"]}' for a in e["atlas_all"]])
