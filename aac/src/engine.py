"""
src/engine.py  —  Block 8 (real-time guardrail engine)

Loads the model ONCE, normalises input, scores it, maintains PERSISTENT per-session
multi-turn state (SQLite), applies the gate, writes a DFIR record, and (output side)
runs Block 4. Designed to back a long-running service (FastAPI / MCP) so detection
is real-time and survives restarts.
"""
import os, sys, sqlite3, time, threading
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.classifier import AttackClassifier
from src.multiturn import ConversationState
from src.gate import decide
from src.enrich import make_finding, log_finding
from src.output_check import output_check
from src.normalize import normalise


class GuardrailEngine:
    def __init__(self, model_path="artifacts/model.joblib", db_path="artifacts/guardrail.db",
                 use_intent=False, intent_threshold=0.5):
        self.model = AttackClassifier.load(model_path)
        self.model_path = model_path
        self.db_path = db_path
        self.use_intent = use_intent          # Block 11: intent model as raise-to-review signal
        self.intent_threshold = intent_threshold
        self._lock = threading.Lock()
        self._init_db()
        # warm up so the first real request isn't slow
        self.model.predict(["warmup"])

    def _db(self):
        c = sqlite3.connect(self.db_path)
        c.execute("""CREATE TABLE IF NOT EXISTS sessions(
            session_id TEXT PRIMARY KEY, cum REAL, prev_suspicious INT,
            turn INT, asset_tier TEXT, updated TEXT)""")
        return c

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._db().close()

    def _load_state(self, c, session_id):
        row = c.execute("SELECT cum, prev_suspicious, turn FROM sessions WHERE session_id=?",
                        (session_id,)).fetchone()
        st = ConversationState()
        if row:
            st.cum, st.prev_suspicious, n = row[0], bool(row[1]), int(row[2])
            st.turns = [None] * n          # only length matters for turn numbering
        return st

    def _save_state(self, c, session_id, st, asset_tier):
        c.execute("""INSERT INTO sessions(session_id,cum,prev_suspicious,turn,asset_tier,updated)
                     VALUES(?,?,?,?,?,?)
                     ON CONFLICT(session_id) DO UPDATE SET
                       cum=excluded.cum, prev_suspicious=excluded.prev_suspicious,
                       turn=excluded.turn, asset_tier=excluded.asset_tier, updated=excluded.updated""",
                  (session_id, st.cum, int(st.prev_suspicious), len(st.turns), asset_tier,
                   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))

    def scan_input(self, text, session_id="sess-001", asset_tier="default", fail_closed=False):
        clean = normalise(text)
        pred = self.model.predict([clean])[0]
        with self._lock:
            c = self._db()
            st = self._load_state(c, session_id)
            cum = st.update(pred["attack_score"], pred["label"])
            self._save_state(c, session_id, st, asset_tier)
            c.commit(); c.close()
        # single-shot OR slow-burn: act on whichever is higher (this turn vs accumulated)
        gate_score = max(pred["attack_score"], cum)
        gate = decide(gate_score, asset_tier, fail_closed=fail_closed)
        # Block 11: intent model can only RAISE log_only -> human_review, never block.
        intent_score = None
        if self.use_intent:
            try:
                from src.intent import injection_score
                intent_score = float(injection_score([clean])[0])
                if intent_score >= self.intent_threshold and gate["action"] == "log_only":
                    gate["action"] = "human_review"
                    gate["confidence_tier"] = "intent_escalated"
            except Exception:
                pass
        finding = make_finding(text, pred, gate, session_id, len(st.turns), round(cum, 4))
        finding["intent_score"] = round(intent_score, 4) if intent_score is not None else None
        log_finding(finding)
        return {"finding": finding, "turn_score": pred["attack_score"],
                "cumulative_score": round(cum, 4), "action": gate["action"],
                "label": pred["label"], "intent_score": finding["intent_score"], "gate": gate}

    def check_output(self, reply, session_id="sess-001", system_prompt=None, canary=None,
                     user_input="", allowlist=None):
        import hashlib
        oc = output_check(reply, system_prompt=system_prompt, canary=canary,
                          user_input=user_input, allowlist=allowlist)
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for f in oc["findings"]:           # log each in the SAME schema the alert report uses
            log_finding({
                "timestamp": ts, "session_id": session_id, "turn": "output",
                "stage": "output", "input_excerpt": reply[:160],
                "prediction": f["class"], "attack_score": None, "cumulative_score": None,
                "is_attack": True, "owasp": f["owasp"], "mitre_atlas": f["mitre_atlas"],
                "atlas_tactic": None, "severity": f["severity"],
                "action": "auto_block", "confidence_tier": "output_guard",
                "reason": f["reason"],
                "ioc": "outputsig:" + hashlib.sha256(f["reason"].encode()).hexdigest()[:16],
            })
        return oc

    def session(self, session_id):
        c = self._db()
        row = c.execute("SELECT cum,prev_suspicious,turn,asset_tier,updated FROM sessions WHERE session_id=?",
                        (session_id,)).fetchone()
        c.close()
        if not row:
            return {"session_id": session_id, "exists": False}
        return {"session_id": session_id, "exists": True, "cumulative_score": round(row[0], 4),
                "turns": row[2], "asset_tier": row[3], "updated": row[4]}

    def reset(self, session_id):
        c = self._db(); c.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        c.commit(); c.close(); return {"session_id": session_id, "reset": True}

    def health(self):
        return {"status": "ok", "model": self.model_path,
                "calibrated": bool(getattr(self.model, "calibration_", None))}
