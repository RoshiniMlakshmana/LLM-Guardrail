"""Interactive same-model guardrail comparison and red-team evidence dashboard."""

from __future__ import annotations

import time

import streamlit as st

from demo_cases import CASES_BY_ID, LEGIT_CASE, OWASP_CASES
from demo_guardrail import load_engine, run_coverage_suite, scan_with_guardrail, vulnerable_response
from llm_runtime import DEFAULT_MODEL, DEMO_CANARY, DEMO_SYSTEM_PROMPT, guarded_exchange, ollama_status, safe_generate
from redteam_tools import tool_rows


st.set_page_config(page_title="AI Guardrail Attack Lab", page_icon="🛡️", layout="wide")

st.markdown(
    """
<style>
  :root { --ink:#e9f0f7; --muted:#91a0b5; --line:#233248; }
  .stApp { background:radial-gradient(circle at 85% -20%,#183b4f 0,#0b1220 34%,#080d16 72%); color:var(--ink); }
  .block-container { max-width:1280px; padding-top:2rem; padding-bottom:4rem; }
  [data-testid="stHeader"] { background:transparent; }
  h1,h2,h3 { letter-spacing:-.035em; }
  .eyebrow { color:#52e0b5; font:700 .75rem/1.2 ui-monospace,monospace; letter-spacing:.14em; text-transform:uppercase; }
  .hero-title { font-size:clamp(2.2rem,5vw,4.1rem); line-height:.97; margin:.55rem 0 .9rem; font-weight:760; }
  .hero-copy { max-width:850px; color:#aab8ca; font-size:1.04rem; line-height:1.65; }
  .flow { display:flex; gap:.6rem; flex-wrap:wrap; margin:1.2rem 0 1.4rem; }
  .flow span { border:1px solid #29405a; color:#bdd0e4; background:#0e1827; padding:.42rem .72rem; border-radius:999px; font-size:.78rem; }
  .flow b { color:#52e0b5; }
  .runtime { border:1px solid #263a52; background:#0d1725; border-radius:12px; padding:.75rem .9rem; color:#aebed0; margin:.4rem 0 1rem; }
  .runtime strong { color:#e8f2fb; }
  .result-card { min-height:275px; border:1px solid var(--line); background:linear-gradient(145deg,rgba(18,29,45,.98),rgba(10,17,29,.96)); border-radius:16px; padding:1.15rem 1.2rem; box-shadow:0 16px 42px rgba(0,0,0,.2); }
  .result-card.danger { border-top:3px solid #ff5f6d; }
  .result-card.safe { border-top:3px solid #52e0b5; }
  .result-card.review { border-top:3px solid #ffc857; }
  .card-kicker { color:var(--muted); font:700 .72rem/1.2 ui-monospace,monospace; letter-spacing:.1em; text-transform:uppercase; }
  .badge { display:inline-block; margin:.75rem 0; padding:.34rem .58rem; border-radius:7px; font:800 .72rem/1 ui-monospace,monospace; }
  .badge.danger { color:#ffb3ba; background:#4a1720; }
  .badge.safe { color:#85f4d3; background:#103b32; }
  .badge.review { color:#ffe39a; background:#463817; }
  .reply { color:#e8eef7; line-height:1.55; max-height:170px; overflow:auto; }
  .detail { color:#8fa2b9; font-size:.84rem; border-top:1px solid #243349; margin-top:1rem; padding-top:.8rem; }
  .evidence-line { display:flex; justify-content:space-between; gap:1rem; padding:.48rem 0; border-bottom:1px solid #223147; color:#aab9ca; font-size:.84rem; }
  .evidence-line strong { color:#e8f2fb; text-align:right; }
  .case-meta { color:#91a0b5; font-size:.83rem; margin-bottom:.35rem; }
  .case-code { background:#09111d; border:1px solid #22334a; border-radius:10px; padding:.78rem .9rem; color:#cfe2f7; font:500 .8rem/1.45 ui-monospace,monospace; overflow-wrap:anywhere; }
  div[data-testid="stMetric"] { background:rgba(15,25,40,.82); border:1px solid #22324a; padding:.8rem 1rem; border-radius:12px; }
  div[data-testid="stMetricValue"] { font-size:1.35rem; }
  .stButton>button { border-radius:9px; font-weight:700; }
  .stButton>button[kind="primary"] { background:#36cda1; color:#05130f; border-color:#36cda1; }
  hr { border-color:#1f2d40; }
  @media (max-width:720px) { .block-container{padding-left:1rem;padding-right:1rem}.hero-title{font-size:2.35rem} }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained guardrail…")
def get_engine():
    return load_engine()


def escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def result_card(title: str, status: str, body: str, detail: str, tone: str):
    st.markdown(
        f"""
        <div class="result-card {tone}">
          <div class="card-kicker">{escape(title)}</div>
          <div class="badge {tone}">{escape(status)}</div>
          <div class="reply">{escape(body)}</div>
          <div class="detail">{escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if "payload" not in st.session_state:
    st.session_state.payload = OWASP_CASES[0].payload
if "active_case" not in st.session_state:
    st.session_state.active_case = OWASP_CASES[0].id
if "comparison" not in st.session_state:
    st.session_state.comparison = None


st.markdown('<div class="eyebrow">AI Security · Controlled Local Experiment</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Same model. Two security postures.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">Send one request directly to a local LLM, then send it through input detection, a confidence gate, the same LLM, and output-leakage inspection. When Ollama is unavailable, the stable simulator keeps the demonstration runnable.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="flow"><span><b>1</b> Same prompt</span><span><b>2</b> ML + policy</span><span><b>3</b> Input gate</span><span><b>4</b> Same model</span><span><b>5</b> Output DLP</span></div>',
    unsafe_allow_html=True,
)

runtime = ollama_status()
model_options = list(runtime.models) or [DEFAULT_MODEL]
selected_model = st.selectbox("Local Ollama model", model_options, index=0)
use_live_ollama = st.toggle(
    "Use live Ollama for both paths",
    value=runtime.available and bool(runtime.models),
    disabled=not (runtime.available and runtime.models),
)
if runtime.available and runtime.models:
    runtime_html = f"<strong>LIVE MODEL READY</strong> · {escape(selected_model)} · same model and system prompt on both paths"
elif runtime.available:
    runtime_html = f"<strong>OLLAMA RUNNING — MODEL MISSING</strong> · run <code>ollama pull {escape(DEFAULT_MODEL)}</code>, then refresh"
else:
    runtime_html = f"<strong>SIMULATOR ACTIVE</strong> · install/start Ollama and run <code>ollama pull {escape(DEFAULT_MODEL)}</code> to enable the real-model experiment"
st.markdown(f'<div class="runtime">{runtime_html}</div>', unsafe_allow_html=True)

selector_col, load_col = st.columns([4, 1])
with selector_col:
    case_ids = [case.id for case in OWASP_CASES] + [LEGIT_CASE.id]
    selected_id = st.selectbox(
        "Choose a prepared test",
        options=case_ids,
        format_func=lambda item: f"{CASES_BY_ID[item].owasp_id} · {CASES_BY_ID[item].name}",
        index=case_ids.index(st.session_state.active_case),
    )
with load_col:
    st.write("")
    st.write("")
    if st.button("Load test", width="stretch"):
        st.session_state.active_case = selected_id
        st.session_state.payload = CASES_BY_ID[selected_id].payload
        st.session_state.comparison = None
        st.rerun()

payload = st.text_area(
    "Prompt or security event",
    key="payload",
    height=112,
    help="The dashboard classifies text and may send it to the selected local model. It never executes submitted commands.",
)
run_col, note_col = st.columns([1, 4])
with run_col:
    run_now = st.button("Run comparison", type="primary", width="stretch")
with note_col:
    st.caption("Local-only experiment · no API key · no command execution · same-model comparison")

if run_now:
    started = time.perf_counter()
    engine = get_engine()
    input_guard = scan_with_guardrail(payload, engine=engine)
    live_enabled = bool(use_live_ollama and runtime.available and runtime.models)

    if live_enabled:
        direct_generation = safe_generate(payload, model=selected_model)
        if direct_generation["available"]:
            direct_observation = engine.check_output(
                direct_generation["response"],
                session_id="dashboard-direct-observation",
                system_prompt=DEMO_SYSTEM_PROMPT,
                canary=DEMO_CANARY,
                user_input=payload,
            )
            unprotected = {
                "status": "DIRECT MODEL",
                "response": direct_generation["response"],
                "exposure": "The prompt reached the model without an input gate.",
                "runtime": "ollama",
                "output_observation": direct_observation,
            }
        else:
            unprotected = {**vulnerable_response(payload), "runtime": "simulator", "output_observation": None}
        protected = guarded_exchange(payload, input_guard, engine=engine, model=selected_model)
    else:
        unprotected = {**vulnerable_response(payload), "runtime": "simulator", "output_observation": None}
        protected = {
            "action": input_guard["action"],
            "response": (
                "Incident response checklist: identify and scope, contain and eradicate, then recover and document lessons learned."
                if input_guard["action"] == "ALLOW"
                else None
            ),
            "model_called": input_guard["action"] == "ALLOW",
            "runtime": "simulator",
            "output_check": {"action": "simulated", "findings": []},
        }

    st.session_state.comparison = {
        "guarded": input_guard,
        "unprotected": unprotected,
        "protected": protected,
        "model": selected_model if live_enabled else "deterministic simulator",
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }

comparison = st.session_state.comparison
if comparison:
    guarded = comparison["guarded"]
    unprotected = comparison["unprotected"]
    protected = comparison["protected"]
    final_action = protected["action"]
    tone = {"BLOCK": "danger", "REVIEW": "review", "ALLOW": "safe"}[final_action]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Guardrail decision", final_action)
    m2.metric("Risk score", f"{guarded['score']:.1%}")
    m3.metric("Detected risk", guarded["owasp_id"] or "None")
    m4.metric("Runtime", comparison["model"])

    left, middle, right = st.columns(3, gap="medium")
    with left:
        result_card(
            "1 · Unprotected LLM",
            unprotected["status"],
            unprotected["response"],
            f"{unprotected['exposure']} Runtime: {unprotected['runtime']}.",
            "danger" if guarded["action"] != "ALLOW" else "safe",
        )
    with middle:
        protected_body = {
            "BLOCK": "Request stopped before the model, or the response was withheld by output DLP.",
            "REVIEW": "Request paused for grounding or human review.",
            "ALLOW": protected.get("response") or "Request and response passed the guardrail.",
        }[final_action]
        result_card(
            "2 · Protected LLM",
            final_action,
            protected_body,
            f"{guarded['reason']} Output check: {protected['output_check'].get('action', 'not run')}.",
            tone,
        )
    with right:
        strongest_rule = guarded["matched_rules"][0]["rule_id"] if guarded["matched_rules"] else "none"
        st.markdown(
            f"""
            <div class="result-card {tone}">
              <div class="card-kicker">3 · Evidence panel</div>
              <div class="badge {tone}">{escape(final_action)}</div>
              <div class="evidence-line"><span>OWASP</span><strong>{escape(guarded['owasp_id'] or 'None')}</strong></div>
              <div class="evidence-line"><span>Confidence</span><strong>{guarded['score']:.1%}</strong></div>
              <div class="evidence-line"><span>Matched rule</span><strong>{escape(strongest_rule)}</strong></div>
              <div class="evidence-line"><span>Model called</span><strong>{'Yes' if protected['model_called'] else 'No'}</strong></div>
              <div class="evidence-line"><span>Output DLP</span><strong>{escape(protected['output_check'].get('action', 'not run'))}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Full detection evidence"):
        e1, e2, e3 = st.columns(3)
        e1.write("**Trained classifier**")
        e1.json(guarded["model"])
        e2.write("**Policy signals**")
        e2.json(guarded["matched_rules"] or {"matched": False})
        e3.write("**Decision binding**")
        e3.json(
            {
                "input_action": guarded["action"],
                "final_action": final_action,
                "score": guarded["score"],
                "owasp": " ".join(filter(None, [guarded["owasp_id"], guarded["owasp_name"]])),
                "control": guarded["control"],
                "output_check": protected["output_check"],
            }
        )

st.divider()
st.subheader("Red-team validation stack")
st.caption("Verified built-in tests, included configurations, and optional heavyweight tools are labelled separately so the evidence stays honest.")
tools = tool_rows()
st.dataframe(
    [
        {
            "Tool": item["name"],
            "Status": item["status"],
            "Tests": item["purpose"],
            "Layer": item["layer"],
            "Evidence": item["evidence"],
        }
        for item in tools
    ],
    width="stretch",
    hide_index=True,
)
for item in tools:
    with st.expander(f"{item['name']} · {item['status']}"):
        st.code(item["command"], language="powershell")
        st.caption(item["evidence"])

st.divider()
st.subheader("OWASP LLM Top 10 attack library")
st.caption("Each payload is non-destructive. System-level risks use deterministic controls; prompt risks also use the trained classifier.")

suite_col, suite_note = st.columns([1, 4])
with suite_col:
    run_suite = st.button("Run all 10 checks", width="stretch")
with suite_note:
    st.caption("Expected behavior: nine blocks and one review for an unsupported factual claim.")

if run_suite:
    rows = run_coverage_suite(engine=get_engine())
    stopped = sum(row["Decision"] in {"BLOCK", "REVIEW"} for row in rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("Coverage", f"{stopped}/10")
    c2.metric("Blocked", sum(row["Decision"] == "BLOCK" for row in rows))
    c3.metric("Sent to review", sum(row["Decision"] == "REVIEW" for row in rows))
    st.dataframe(rows, width="stretch", hide_index=True)

for case in OWASP_CASES:
    with st.expander(f"{case.owasp_id} · {case.name}"):
        st.markdown(
            f'<div class="case-meta">Vector: {escape(case.vector)} · Control: {escape(case.expected_control)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="case-code">{escape(case.payload)}</div>', unsafe_allow_html=True)
        st.caption(f"Potential impact: {case.impact}")
        if st.button("Load into comparison", key=f"load-{case.id}"):
            st.session_state.active_case = case.id
            st.session_state.payload = case.payload
            st.session_state.comparison = None
            st.rerun()

st.markdown("#### Legitimate control prompt")
st.markdown(f'<div class="case-code">{escape(LEGIT_CASE.payload)}</div>', unsafe_allow_html=True)
st.caption("Expected decision: ALLOW. This confirms the dashboard is not simply blocking every request.")
st.button(
    "Load legitimate prompt",
    key="load-legitimate",
    on_click=lambda: st.session_state.update(
        active_case=LEGIT_CASE.id,
        payload=LEGIT_CASE.payload,
        comparison=None,
    ),
)
st.caption("Controlled security lab: Ollama is local, submitted commands are never executed, and optional red-team tools run separately.")
