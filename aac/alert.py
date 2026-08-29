"""
alert.py  —  Block 17 (alerting / report generation; dashboard paused)

Turns serious detections from the DFIR log into analyst-ready ALERT REPORTS — each
with severity, OWASP/ATLAS mapping, the IOC, and a recommended response playbook
line. Scans the JSONL forensic log and emits alerts for blocked / reviewed / attack
events. (The visual dashboard is intentionally deferred.)
"""
import os, sys, json

PLAYBOOK = {
    "prompt_injection":   "Block session; preserve transcript; re-issue with hardened system prompt.",
    "jailbreak":          "Block; flag account; review for repeat offenders.",
    "system_prompt_exfil":"Rotate any exposed secrets immediately; block; audit prompt exposure.",
    "sensitive_data_probe":"Block & redact reply; notify privacy/DLP; rotate exposed creds; log per breach policy.",
    "model_extraction":   "Rate-limit/ban the source; watch for scraping patterns.",
    "indirect_injection": "Quarantine the source document; re-sanitise & re-index the KB.",
    "excessive_agency":   "Block the tool action; revoke/scope the agent's permissions; audit.",
}
SEV_RANK = {"high": 3, "medium": 2, "low": 1, "none": 0}


def format_alert(rec):
    cls = rec.get("prediction", "?")
    sev = (rec.get("severity") or "none").upper()
    rec_action = PLAYBOOK.get(cls, "Review manually.")
    return (
        f"+--------------------------------------------------------------+\n"
        f"| SECURITY ALERT  [{sev:^6}]   {rec.get('timestamp','')}\n"
        f"| session : {rec.get('session_id','-')}  turn {rec.get('turn','-')}   action: {rec.get('action','-')}\n"
        f"| detect  : {cls}  (score {rec.get('attack_score','-')}, cum {rec.get('cumulative_score','-')})\n"
        f"| OWASP   : {rec.get('owasp') or '-'}\n"
        f"| ATLAS   : {rec.get('mitre_atlas') or '-'}\n"
        f"| IOC     : {rec.get('ioc','-')}\n"
        f"| detail  : {(rec.get('reason') or '-')[:60]}\n"
        f"| input   : {(rec.get('input_excerpt') or '')[:60]}\n"
        f"| RESPONSE: {rec_action}\n"
        f"+--------------------------------------------------------------+"
    )


def alerts_from_log(path="artifacts/dfir_log.jsonl", min_sev="medium",
                    actions=("auto_block", "human_review")):
    if not os.path.exists(path):
        print("no DFIR log found at", path); return []
    out = []
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not r.get("is_attack"):
            continue
        sev_ok = SEV_RANK.get(r.get("severity", "none"), 0) >= SEV_RANK.get(min_sev, 0)
        act_ok = r.get("action") in actions
        if sev_ok or act_ok:
            out.append(r)
    return out


def main():
    alerts = alerts_from_log()
    print(f"Generated {len(alerts)} alert(s) from the DFIR log:\n")
    for r in alerts[-8:]:
        print(format_alert(r), "\n")
    # tiny summary (the 'report')
    from collections import Counter
    by_class = Counter(a.get("prediction") for a in alerts)
    by_sev = Counter((a.get("severity") or "none") for a in alerts)
    print("SUMMARY REPORT")
    print("  by attack type:", dict(by_class))
    print("  by severity   :", dict(by_sev))


if __name__ == "__main__":
    main()
