"""Layered comparison logic used by the Streamlit demonstration dashboard."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Iterable

from demo_cases import LEGIT_CASE, OWASP_CASES


ACTION_RANK = {"ALLOW": 0, "REVIEW": 1, "BLOCK": 2}
ENGINE_ACTIONS = {"log_only": "ALLOW", "human_review": "REVIEW", "auto_block": "BLOCK"}


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    owasp_id: str
    name: str
    pattern: str
    action: str
    score: float
    reason: str
    control: str


POLICY_RULES = [
    PolicyRule(
        "instruction-override",
        "LLM01",
        "Prompt Injection",
        r"\b(ignore|disregard|forget|override)\b.{0,80}\b(instructions?|rules?|prompt|policy)\b",
        "BLOCK",
        0.99,
        "The prompt attempts to replace trusted instructions.",
        "Instruction hierarchy enforcement",
    ),
    PolicyRule(
        "secret-request",
        "LLM02",
        "Sensitive Information Disclosure",
        r"\b(list|show|reveal|give|print|dump|send)\b.{0,80}\b(api keys?|passwords?|secret tokens?|credentials?|private keys?|ssn|credit cards?)\b",
        "BLOCK",
        0.99,
        "The request explicitly asks for protected secrets or personal data.",
        "Input secret-intent detection + output DLP",
    ),
    PolicyRule(
        "unsigned-artifact",
        "LLM03",
        "Supply Chain",
        r"\b(load_model|install|import)\b.*\b(verify_signature\s*=\s*false|revision\s*=\s*[\"']latest[\"'])",
        "BLOCK",
        0.98,
        "An unpinned or unsigned AI artifact is being loaded.",
        "Artifact signature verification and version pinning",
    ),
    PolicyRule(
        "untrusted-training-ingest",
        "LLM04",
        "Data and Model Poisoning",
        r"\bingest_(training_)?data\b.*\b(trust\s*=\s*[\"']unknown[\"']|review\s*=\s*false)",
        "BLOCK",
        0.98,
        "Untrusted training data would bypass quarantine or review.",
        "Provenance checks, quarantine, and analyst approval",
    ),
    PolicyRule(
        "active-output-content",
        "LLM05",
        "Improper Output Handling",
        r"(<script\b|onerror\s*=|onload\s*=|javascript:|render.{0,40}directly as html)",
        "BLOCK",
        0.99,
        "The request contains active content that must not be rendered directly.",
        "Context-aware output encoding and sanitization",
    ),
    PolicyRule(
        "dangerous-tool-action",
        "LLM06",
        "Excessive Agency",
        r"\b(shell|tool|plugin|api)\b.{0,100}\b(delete|disable|transfer|grant|revoke|overwrite)\b|\b(delete|disable|transfer|grant|revoke|overwrite)\b.{0,100}\b(shell|tool|firewall|audit logs?)\b",
        "BLOCK",
        0.99,
        "A high-impact tool action is requested without an approval boundary.",
        "Least-privilege tool allowlist and human approval",
    ),
    PolicyRule(
        "meta-prompt-extraction",
        "LLM07",
        "System Prompt Leakage",
        r"\b(print|show|reveal|repeat|output)\b.{0,70}\b(hidden|original|system)\b.{0,30}\b(instructions?|prompt)\b",
        "BLOCK",
        0.99,
        "The prompt requests hidden system instructions.",
        "System-prompt extraction detection + output canary",
    ),
    PolicyRule(
        "poisoned-rag-content",
        "LLM08",
        "Vector and Embedding Weaknesses",
        r"\b(index_document|upsert|embed)\b.*(\[system override\]|ignore.{0,30}(query|instructions?)|return all.{0,30}records)",
        "BLOCK",
        0.98,
        "Retrieved content contains an instruction aimed at the model.",
        "RAG ingestion screening and tenant/namespace isolation",
    ),
    PolicyRule(
        "unsupported-claim",
        "LLM09",
        "Misinformation",
        r"\b(state|claim|present)\b.{0,50}\b(verified fact|definitely true|confirmed)\b.{0,90}\b(even if|without)\b.{0,30}\b(no source|source|evidence|confirms?)\b",
        "REVIEW",
        0.82,
        "The user asks the model to present an unsupported claim as verified.",
        "Grounded generation, citations, and human review",
    ),
    PolicyRule(
        "resource-budget",
        "LLM10",
        "Unbounded Consumption",
        r"\b(repeat|generate|produce)\b.{0,40}\b([0-9][0-9,]{4,}|million|billion)\b|\bkeep retrying\b|\bwithout (a )?(limit|stopping)\b",
        "BLOCK",
        0.98,
        "The request exceeds safe token or retry budgets.",
        "Token, rate, timeout, and retry limits",
    ),
]


COMPILED_RULES = [(rule, re.compile(rule.pattern, re.IGNORECASE | re.DOTALL)) for rule in POLICY_RULES]


def _model_path() -> str:
    preferred = os.environ.get("GUARDRAIL_MODEL", "artifacts/model_precision.joblib")
    return preferred if os.path.exists(preferred) else "artifacts/model.joblib"


@lru_cache(maxsize=1)
def load_engine():
    """Load the trained classifier once; Streamlit reruns reuse this instance."""
    from src.engine import GuardrailEngine

    return GuardrailEngine(
        model_path=_model_path(),
        db_path=os.environ.get("GUARDRAIL_DB", "artifacts/dashboard_guardrail.db"),
    )


def policy_matches(payload: str) -> list[PolicyRule]:
    return [rule for rule, regex in COMPILED_RULES if regex.search(payload)]


def _strongest_action(actions: Iterable[str]) -> str:
    return max(actions, key=lambda item: ACTION_RANK[item], default="ALLOW")


def scan_with_guardrail(payload: str, engine=None) -> dict:
    """Combine the trained prompt classifier with non-ML policy controls.

    OWASP categories such as supply-chain compromise and poisoning are system
    events, not merely prompt strings.  Treating those as deterministic policy
    checks is deliberate and is surfaced in the UI.
    """
    engine = engine or load_engine()
    session_id = "dashboard-" + hashlib.sha256(payload.encode()).hexdigest()[:16]
    engine.reset(session_id)
    model_result = engine.scan_input(payload, session_id=session_id)
    model_action = ENGINE_ACTIONS.get(model_result["action"], "REVIEW")
    matches = policy_matches(payload)
    policy_action = _strongest_action(rule.action for rule in matches)
    final_action = _strongest_action([model_action, policy_action])
    strongest = max(matches, key=lambda rule: (ACTION_RANK[rule.action], rule.score), default=None)
    score = max(float(model_result["turn_score"]), strongest.score if strongest else 0.0)

    if strongest:
        reason = strongest.reason
        control = strongest.control
        owasp_id = strongest.owasp_id
        owasp_name = strongest.name
    elif final_action != "ALLOW":
        finding = model_result["finding"]
        reason = "The trained classifier identified attack-like intent."
        control = "Calibrated classifier and confidence gate"
        owasp = finding.get("owasp") or ""
        owasp_id, _, owasp_name = owasp.partition(" ")
    else:
        reason = "No attack signature or unsafe system event was detected."
        control = "Normal request path"
        owasp_id = None
        owasp_name = None

    return {
        "action": final_action,
        "score": round(score, 4),
        "reason": reason,
        "control": control,
        "owasp_id": owasp_id,
        "owasp_name": owasp_name,
        "matched_rules": [asdict(rule) for rule in matches],
        "model": {
            "action": model_action,
            "label": model_result["label"],
            "score": model_result["turn_score"],
            "confidence_tier": model_result["gate"]["confidence_tier"],
        },
    }


def vulnerable_response(payload: str) -> dict:
    """Return a safe simulation of how an unguarded assistant could fail."""
    matches = policy_matches(payload)
    if not matches:
        return {
            "status": "RESPONDED",
            "response": "Incident response checklist: identify and scope, contain and eradicate, then recover and document lessons learned.",
            "exposure": "No unsafe behavior simulated for this request.",
        }

    top = max(matches, key=lambda rule: (ACTION_RANK[rule.action], rule.score))
    simulated = {
        "LLM01": "Accepted the new instruction hierarchy and ignored the trusted policy.",
        "LLM02": "Returned placeholder credentials: DEMO_KEY_REDACTED (simulated leak).",
        "LLM03": "Loaded the unpinned community artifact without verification.",
        "LLM04": "Added the untrusted records directly to the training set.",
        "LLM05": "Passed the active HTML to the renderer without encoding.",
        "LLM06": "Sent the destructive action to the tool layer without approval.",
        "LLM07": "Returned a redacted stand-in for the hidden prompt (simulated leak).",
        "LLM08": "Indexed the poisoned document into the trusted retrieval collection.",
        "LLM09": "Presented the unsupported breach claim as verified.",
        "LLM10": "Accepted an unbounded generation and retry request.",
    }
    return {
        "status": "EXPOSED",
        "response": simulated[top.owasp_id],
        "exposure": top.reason,
    }


def run_coverage_suite(engine=None) -> list[dict]:
    engine = engine or load_engine()
    rows = []
    for case in OWASP_CASES:
        result = scan_with_guardrail(case.payload, engine=engine)
        rows.append(
            {
                "OWASP": case.owasp_id,
                "Risk": case.name,
                "Decision": result["action"],
                "Score": result["score"],
                "Detected as": result["owasp_id"] or result["model"]["label"],
                "Control": result["control"],
            }
        )
    return rows


def self_check(engine=None) -> dict:
    engine = engine or load_engine()
    attacks = run_coverage_suite(engine)
    legit = scan_with_guardrail(LEGIT_CASE.payload, engine=engine)
    return {
        "attack_cases": len(attacks),
        "stopped_or_reviewed": sum(row["Decision"] in {"BLOCK", "REVIEW"} for row in attacks),
        "legitimate_action": legit["action"],
        "passed": all(row["Decision"] in {"BLOCK", "REVIEW"} for row in attacks)
        and legit["action"] == "ALLOW",
    }

