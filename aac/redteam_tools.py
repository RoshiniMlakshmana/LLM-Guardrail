"""Red-team tool catalog and local readiness checks for the dashboard."""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RedTeamTool:
    name: str
    purpose: str
    layer: str
    command: str
    executable: str | None
    evidence: str


TOOLS = [
    RedTeamTool(
        "Built-in OWASP Suite",
        "Runs the ten non-destructive OWASP LLM risk cases and legitimate control.",
        "Regression",
        "python -m pytest tests/test_dashboard.py -q",
        None,
        "Included and automatically tested in this repository.",
    ),
    RedTeamTool(
        "Promptfoo",
        "API-focused prompt-injection regression and assertion-based scorecards.",
        "Prompt / API",
        "npx promptfoo@latest eval -c redteam/promptfooconfig.yaml --no-cache",
        "npx",
        "Configuration included; Promptfoo was exercised during project development.",
    ),
    RedTeamTool(
        "Garak",
        "Broad LLM vulnerability probes against the local Ollama model.",
        "Model",
        "garak --model_type ollama --model_name llama3.2:3b --probes promptinject",
        "garak",
        "Optional integration; results depend on the installed Garak version and model.",
    ),
    RedTeamTool(
        "PyRIT",
        "Orchestrated multi-turn and adaptive adversarial experiments.",
        "Conversation",
        "python redteam/pyrit_ollama_example.py",
        None,
        "Starter adapter included; install PyRIT separately before running it.",
    ),
]


def tool_rows() -> list[dict]:
    rows = []
    for tool in TOOLS:
        item = asdict(tool)
        if tool.name == "Built-in OWASP Suite":
            status = "READY"
        elif tool.name == "Promptfoo":
            status = "RUNNER READY" if shutil.which("npx") else "CONFIG READY"
        elif tool.name == "Garak":
            status = "INSTALLED" if shutil.which("garak") else "OPTIONAL"
        elif tool.name == "PyRIT":
            try:
                import pyrit  # noqa: F401

                status = "INSTALLED"
            except ImportError:
                status = "OPTIONAL"
        item["status"] = status
        rows.append(item)
    return rows


def config_checks() -> dict:
    return {
        "promptfoo_config": os.path.exists("redteam/promptfooconfig.yaml"),
        "garak_guide": os.path.exists("redteam/README.md"),
        "pyrit_adapter": os.path.exists("redteam/pyrit_ollama_example.py"),
    }
