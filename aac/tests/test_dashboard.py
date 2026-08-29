import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_cases import LEGIT_CASE, OWASP_CASES
from demo_guardrail import POLICY_RULES, scan_with_guardrail, self_check
from llm_runtime import guarded_exchange, ollama_status
from redteam_tools import config_checks, tool_rows


def test_attack_library_covers_each_owasp_risk_once():
    ids = [case.owasp_id for case in OWASP_CASES]
    assert ids == [f"LLM{i:02d}" for i in range(1, 11)]
    assert len(set(ids)) == 10


def test_every_demo_attack_matches_its_expected_policy_rule():
    for case in OWASP_CASES:
        result = scan_with_guardrail(case.payload)
        assert result["action"] in {"BLOCK", "REVIEW"}, case.owasp_id
        assert any(rule["owasp_id"] == case.owasp_id for rule in result["matched_rules"]), case.owasp_id


def test_legitimate_prompt_is_allowed():
    result = scan_with_guardrail(LEGIT_CASE.payload)
    assert result["action"] == "ALLOW"
    assert result["matched_rules"] == []


def test_policy_catalog_has_unique_rule_ids():
    ids = [rule.rule_id for rule in POLICY_RULES]
    assert len(ids) == len(set(ids))


def test_end_to_end_self_check_passes():
    result = self_check()
    assert result == {
        "attack_cases": 10,
        "stopped_or_reviewed": 10,
        "legitimate_action": "ALLOW",
        "passed": True,
    }


def test_blocked_input_never_calls_the_model():
    result = guarded_exchange("blocked", {"action": "BLOCK"}, engine=object())
    assert result["action"] == "BLOCK"
    assert result["model_called"] is False
    assert result["output_check"]["action"] == "not_run"


def test_optional_ollama_status_never_breaks_offline_mode():
    status = ollama_status(timeout=0.05)
    assert isinstance(status.available, bool)
    assert isinstance(status.models, tuple)


def test_redteam_configs_and_status_catalog_are_present():
    assert config_checks() == {
        "promptfoo_config": True,
        "garak_guide": True,
        "pyrit_adapter": True,
    }
    rows = tool_rows()
    assert [row["name"] for row in rows] == [
        "Built-in OWASP Suite",
        "Promptfoo",
        "Garak",
        "PyRIT",
    ]
    assert rows[0]["status"] == "READY"


def test_streamlit_dashboard_primary_flow():
    from streamlit.testing.v1 import AppTest

    app_path = os.path.join(os.path.dirname(__file__), "..", "dashboard.py")
    app = AppTest.from_file(app_path, default_timeout=20).run()
    assert not app.exception

    next(button for button in app.button if button.label == "Run comparison").click().run()
    assert not app.exception
    assert app.metric[0].label == "Guardrail decision"
    assert app.metric[0].value == "BLOCK"
    assert app.metric[2].value == "LLM01"
    assert any("Unprotected LLM" in markdown.value for markdown in app.markdown)
    assert any("Protected LLM" in markdown.value for markdown in app.markdown)
    assert any("Evidence panel" in markdown.value for markdown in app.markdown)

    next(button for button in app.button if button.label == "Run all 10 checks").click().run()
    assert not app.exception
    assert any(metric.label == "Coverage" and metric.value == "10/10" for metric in app.metric)
