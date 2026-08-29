# Red-team validation

These integrations are optional. The built-in OWASP suite remains the stable,
offline regression gate.

## Promptfoo — API regression

Start the API in terminal 1:

```powershell
python -m uvicorn server:app --port 8000
```

Run Promptfoo in terminal 2 (Node.js is required):

```powershell
npx promptfoo@latest eval -c redteam/promptfooconfig.yaml --no-cache
npx promptfoo@latest view
```

## Garak — model probing

Install Garak in a separate virtual environment because its dependency stack is
large and changes independently from the dashboard:

```powershell
py -m venv .venv-garak
.\.venv-garak\Scripts\Activate.ps1
python -m pip install garak
garak --model_type ollama --model_name llama3.2:3b --probes promptinject
```

Confirm the exact generator and probe names with `garak --list_config` if your
installed Garak release uses different names.

## PyRIT — adaptive conversation experiments

PyRIT is intentionally not part of the lightweight dashboard requirements.
Install it in a separate environment and adapt `pyrit_ollama_example.py` to the
API surface exposed by the installed release.

```powershell
python -m pip install pyrit
python redteam/pyrit_ollama_example.py
```

The adapter refuses to claim a completed PyRIT run when the package is absent.

