"""
src/gate.py  —  Block 5 (confidence gate)
Turns a calibrated score into an action. Thresholds are per-asset-tier, mirroring
the calibrated-automation pattern: a workstation can auto-block at 0.90, a
revenue-critical service may require 0.98 for the same automated action.
"""
DEFAULT_TIERS = {
    "default":     {"auto_block": 0.95, "review": 0.70},
    "workstation": {"auto_block": 0.90, "review": 0.65},
    "prod_server": {"auto_block": 0.98, "review": 0.80},
}


def decide(score: float, asset_tier: str = "default", fail_closed: bool = False) -> dict:
    """fail_closed=True (recommended for critical assets): the human_review band is
    promoted to auto_block, so anything *suspicious* is stopped, not just logged.
    Costs more false-positive blocks -- a deliberate availability-vs-safety choice."""
    t = DEFAULT_TIERS.get(asset_tier, DEFAULT_TIERS["default"])
    if score >= t["auto_block"]:
        action, tier = "auto_block", "high_confidence"
    elif score >= t["review"]:
        action, tier = ("auto_block" if fail_closed else "human_review"), "ambiguous"
    else:
        action, tier = "log_only", "low_confidence"
    return {"action": action, "confidence_tier": tier, "score": round(score, 4),
            "asset_tier": asset_tier, "fail_closed": fail_closed, "thresholds": t}
