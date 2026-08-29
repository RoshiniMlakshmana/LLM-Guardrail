"""
src/output_check.py  —  Block 4 (output-side guardrail) + PII detection

Inspects the model's REPLY before it leaves. Blocks if the reply shows the attack
succeeded: system-prompt leakage (LLM07), secret/PII leakage (LLM02), or jailbroken
compliance (LLM01).
"""
import re, sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from taxonomy import TAXONOMY

SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "openai-style api key"),
    (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"), "private key"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "aws access key"),
    (re.compile(r"\b(password|passwd|api[_ ]?key|secret|token)\b\s*[:=]\s*\S+", re.I), "credential dump"),
]
PII_PATTERNS = [
    (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "email"),
    (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "phone"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "ssn"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "credit card"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "ip address"),
    (re.compile(r"\b\d{1,5}\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Blvd|Drive|Dr)\b"), "street address"),
]
LEAK_MARKERS = [
    re.compile(r"\bmy (system )?(prompt|instructions?) (is|are|were|:)\b", re.I),
    re.compile(r"\bI (was|am) (instructed|told|programmed|configured) to\b", re.I),
    re.compile(r"\bmy (initial|hidden|developer|system) (instructions?|prompt)\b", re.I),
    re.compile(r"\bhere (is|are) (my|the) (system )?(prompt|instructions?)\b", re.I),
]
JAILBREAK_COMPLY = [
    re.compile(r"\bas dan\b", re.I),
    re.compile(r"\bas an? (unrestricted|unfiltered|uncensored|jailbroken|amoral) (ai|assistant|model)\b", re.I),
    re.compile(r"\bI('| wi)ll (now )?ignore (my|all) (guidelines|rules|restrictions|instructions)\b", re.I),
    re.compile(r"\bsure,? here('?s| is) how (to|you) (make|build|create|synthesi[sz]e|hotwire)\b.{0,40}\b(bomb|explosive|weapon|meth|nerve agent|malware|ransomware)\b", re.I),
]


SENSITIVE_PII = {"ssn", "credit card"}   # never legitimate in an output

def pii_scan(text):
    return [name for rx, name in PII_PATTERNS if rx.search(text)]

def assess_pii(text, allowed=""):
    """Tiered + context-aware: PII the user already supplied (or allow-listed) is NOT
    a leak. Of the rest: any sensitive item (SSN/card) or a record-dump => block."""
    al = (allowed or "").lower()
    types, total = set(), 0
    for rx, name in PII_PATTERNS:
        for m in rx.finditer(text):
            val = m.group(0).strip().lower()
            if name not in SENSITIVE_PII and val and val in al:   # only CONTACT pii can be allow-listed
                continue                                          # ssn/card: never echo, even if user gave it
            types.add(name); total += 1
    if not types:
        return None
    sensitive = bool(types & SENSITIVE_PII)
    bulk = len(types) >= 3 or total >= 5
    if sensitive or bulk:
        kind = "sensitive" if sensitive else "bulk/record"
        return ("block", f"PII leak ({kind}): " + ", ".join(sorted(types)))
    return ("ok", None)


def _ngram_overlap(system_prompt: str, reply: str, n: int = 8) -> bool:
    sp = re.findall(r"\w+", system_prompt.lower())
    rl = " " + " ".join(re.findall(r"\w+", reply.lower())) + " "
    for i in range(0, max(0, len(sp) - n + 1)):
        if " " + " ".join(sp[i:i + n]) + " " in rl:
            return True
    return False


def output_check(reply: str, system_prompt: str = None, canary: str = None, user_input: str = '', allowlist=None) -> dict:
    findings = []
    leaked = False
    if canary and canary in reply:
        leaked, why = True, f"canary token '{canary}' surfaced in reply"
    elif system_prompt and _ngram_overlap(system_prompt, reply):
        leaked, why = True, "verbatim system-prompt text in reply"
    else:
        for rx in LEAK_MARKERS:
            if rx.search(reply):
                leaked, why = True, "system-prompt leak phrasing in reply"
                break
    if leaked:
        findings.append(("system_prompt_exfil", why))
    for rx, name in SECRET_PATTERNS:
        if rx.search(reply):
            findings.append(("sensitive_data_probe", f"CREDENTIAL EXPOSED ({name}) -> ROTATE/REVOKE immediately; now compromised"))
            break
    allowed = (user_input or '') + ' ' + ' '.join(allowlist or [])
    pa = assess_pii(reply, allowed)
    if pa and pa[0] == "block":
        findings.append(("sensitive_data_probe", pa[1]))
    for rx in JAILBREAK_COMPLY:
        if rx.search(reply):
            findings.append(("jailbreak", "reply adopts jailbroken persona / harmful compliance"))
            break

    enriched = []
    for cls, why in findings:
        t = TAXONOMY[cls]
        enriched.append({"reason": why, "class": cls,
                         "owasp": f"{t['owasp_id']} {t['owasp_name']}",
                         "mitre_atlas": f"{t['atlas_id']} {t['atlas_technique']}",
                         "severity": t["severity"]})
    return {"compromised": len(findings) > 0,
            "action": "block_response" if findings else "allow_response",
            "findings": enriched}
