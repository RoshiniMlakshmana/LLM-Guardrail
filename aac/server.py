"""
server.py  —  Block 8 real-time guardrail service (FastAPI)

Endpoints:
  GET  /health
  POST /scan_input     {text, session_id, asset_tier, fail_closed}
  POST /check_output   {reply, session_id, system_prompt, canary}
  GET  /session/{id}
  POST /reset/{id}

Run:  uvicorn server:app --host 0.0.0.0 --port 8000
Model loaded ONCE at startup; sessions persist in artifacts/guardrail.db.
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel
from src.engine import GuardrailEngine

MODEL = os.environ.get("GUARDRAIL_MODEL", "artifacts/model_precision.joblib")
DB = os.environ.get("GUARDRAIL_DB", "artifacts/guardrail.db")
USE_INTENT = os.environ.get("USE_INTENT", "0") == "1"
INTENT_THR = float(os.environ.get("INTENT_THRESHOLD", "0.5"))
engine = GuardrailEngine(model_path=MODEL, db_path=DB, use_intent=USE_INTENT, intent_threshold=INTENT_THR)
app = FastAPI(title="AI Attack Guardrail", version="0.8")


class ScanIn(BaseModel):
    text: str
    session_id: str = "sess-001"
    asset_tier: str = "default"
    fail_closed: bool = False

class OutIn(BaseModel):
    reply: str
    session_id: str = "sess-001"
    system_prompt: str | None = None
    canary: str | None = None


@app.get("/health")
def health():
    return engine.health()

@app.post("/scan_input")
def scan_input(b: ScanIn):
    return engine.scan_input(b.text, b.session_id, b.asset_tier, b.fail_closed)

@app.post("/check_output")
def check_output(b: OutIn):
    return engine.check_output(b.reply, b.session_id, b.system_prompt, b.canary)

@app.get("/session/{session_id}")
def session(session_id: str):
    return engine.session(session_id)

@app.post("/reset/{session_id}")
def reset(session_id: str):
    return engine.reset(session_id)
