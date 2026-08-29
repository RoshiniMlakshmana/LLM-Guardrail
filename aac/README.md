# AI Attack Detection Classifier

A custom classifier that detects **attacks against AI models** and outputs a
**calibrated confidence score** used to block malicious input. Every detection is
mapped to **OWASP LLM Top-10 (2025)** and **MITRE ATLAS**, runs through a
**confidence-gated response**, and is written to a **DFIR / threat-intel log** —
so this is a detection + IR + intelligence pipeline, not just a model.

> Capstone for the AI for Cybersecurity bootcamp (Class 5: OWASP Top 10 for LLMs,
> MITRE ATLAS & Harness Engineering). Concentration: threat intelligence + DFIR + AI, blue team.

## What it detects (classes → frameworks)
| class | OWASP 2025 | MITRE ATLAS |
|---|---|---|
| prompt_injection | LLM01 Prompt Injection | AML.T0051 |
| jailbreak | LLM01 Prompt Injection | AML.T0054 |
| system_prompt_exfil | LLM07 System Prompt Leakage | AML.T0056 |
| sensitive_data_probe | LLM02 Sensitive Info Disclosure | AML.T0057 |
| model_extraction | LLM02 Sensitive Info Disclosure | AML.T0024 |
| indirect_injection | LLM01 Prompt Injection | AML.T0051 |
| excessive_agency | LLM06 Excessive Agency | AML.T0053 |

ATLAS is a living framework — validate IDs at https://atlas.mitre.org.
OWASP list: https://genai.owasp.org/llm-top-10/

## Architecture (blocks)
1. **Data + taxonomy** — `taxonomy.py`, `data/build_dataset.py` (attacks + benign hard-negatives, leak-free split)
2. **Ensemble classifier** — `src/heuristics.py` (signatures) + `src/encoders.py` (TF-IDF / sentence-transformer) + `src/classifier.py` (logistic head)
3. **Multi-turn accumulator** — `src/multiturn.py` (catches slow-burn jailbreaks)
4. *Output-side check* — TODO (score the model's reply too)
5. **Confidence gate** — `src/gate.py` (per-asset-tier thresholds)
6. **Enrichment + DFIR** — `src/enrich.py` (OWASP/ATLAS map, IOC, forensic log)
7. *Adversary emulation* — TODO (evasion red-teaming, before/after curve)

## Quickstart
```bash
pip install -r requirements.txt
python data/build_dataset.py          # Block 1
python train.py --encoder tfidf       # Block 2  (use --encoder st on Colab)
python pipeline.py                    # Blocks 3->6 end-to-end demo
```
On Colab: open `notebooks/ai_attack_classifier_colab.ipynb` and run top to bottom.

## Current metrics (TF-IDF baseline, leak-free split)
ROC-AUC ~0.99, binary precision ~0.98 / recall ~0.81, benign FPR@0.70 ~0.26.
The semantic (sentence-transformer) encoder is expected to cut the false-positive
rate and lift the semantically-similar classes (model_extraction, sensitive_data_probe).
These numbers are from a small seed set — fold in JailbreakBench / AdvBench /
HackAPrompt / PINT (same `{text,label}` schema) for honest, publishable results.

## Honest limitations
- Seed dataset is small and partly templated; treat baseline metrics as indicative.
- Heuristic signatures are high-precision but easily evaded — they are a *layer*, not the detector.
- Calibration (Block 5) and evasion testing (Block 7) are required before any "blocks malicious input in production" claim.

## Block 5 — Calibration (added)

Why: the gate (`src/gate.py`) acts at fixed thresholds (0.70 review / 0.95 / 0.98
auto-block). Those are only honest if the score is calibrated — a 0.70 score should
mean ~70% chance of attack. The raw logistic head is not calibrated.

Files: `src/calibration.py` (ECE, Brier, reliability curve + diagram),
`calibrate.py` (runner). `AttackClassifier.fit(..., calibrate='sigmoid')` wraps the
head in `CalibratedClassifierCV` with internal CV — no test data touched, no
calibration-on-train leakage.

Run:
    python calibrate.py --encoder tfidf --method sigmoid

Real result (TF-IDF, untouched test n=107), sigmoid vs raw:
    ECE   0.178 -> 0.109   (better)
    Brier 0.106 -> 0.085   (better)
Isotonic was tried and rejected: it overfits this small set (Brier 0.106 -> 0.111,
worse). Sigmoid/Platt is the default. Outputs: `artifacts/calibration.json`,
`artifacts/reliability_diagram.png`, `artifacts/model_calibrated.joblib`.

`pipeline.py` now prefers `model_calibrated.joblib` so the gated response runs on
calibrated scores. Honest side effect: after calibration the demo multi-turn
jailbreak reaches `human_review` (not `auto_block`) on a prod_server tier — the raw
model was overconfident; requiring 0.98 to auto-block a prod-critical asset is the
correct posture. OWASP-2025 + MITRE ATLAS mapping is unchanged and guarded by
`tests/test_integrity.py`.

Caveat: test split is small (n=107, ~27 benign); ECE over 10 bins is noisy, Brier is
the steadier summary. Numbers are indicative, not publishable until the dataset grows
(Block e: fold in JailbreakBench / PINT).
