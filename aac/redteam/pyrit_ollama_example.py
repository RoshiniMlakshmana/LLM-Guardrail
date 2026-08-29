"""Version-safe starting point for a PyRIT-to-Ollama experiment.

PyRIT's public classes have changed between releases, so this file validates the
runtime and prints the experiment contract instead of silently using an outdated
API.  Pin and implement against the version selected for the research run.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm_runtime import DEFAULT_MODEL, ollama_status


def main():
    try:
        import pyrit
    except ImportError:
        raise SystemExit("PyRIT is not installed. Run: python -m pip install pyrit")

    status = ollama_status()
    if not status.available:
        raise SystemExit("Ollama is not running. Start Ollama before the PyRIT experiment.")

    print("PyRIT runtime:", getattr(pyrit, "__version__", "installed"))
    print("Target model:", DEFAULT_MODEL)
    print("Experiment contract: multi-turn attacks -> Ollama target -> guardrail evidence")
    print("Pin this installed PyRIT version, then implement its current orchestrator API.")


if __name__ == "__main__":
    sys.exit(main())
