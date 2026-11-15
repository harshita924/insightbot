# InsightBot — Natural-Language Analytics & Insight-Narrative Assistant
IILM-AHL Jaipur · PGDM 2025-27 · Sem III · *Agentic & Generative AI Applications for Business* · Project II (Business Analytics)
Student: Harshita Khatri (25GM014)

A **read-only** chat assistant: a manager asks a business question in plain English → InsightBot checks it, builds a SQL query,
runs it on a whitelisted table, and returns the numbers, a chart, a short insight narrative and the **exact query used**.
Ambiguous / out-of-scope / risky questions are **escalated to a human analyst**; write attempts and prompt injections are **blocked and logged**.

## Run locally
```bash
pip install -r requirements.txt
python make_data.py          # (optional) regenerates the synthetic dataset
streamlit run app.py
```
Works **without any API key** (offline rules engine). To enable the Claude LLM engine, set the key as an environment
variable or Streamlit secret — never in code:
```bash
export ANTHROPIC_API_KEY=...        # optional  (model override: INSIGHTBOT_MODEL)
```
## Deploy (Streamlit Community Cloud)
Push this folder to a GitHub repo → share.streamlit.io → New app → main file `app.py`. Add `ANTHROPIC_API_KEY` under
*App settings → Secrets* (optional). Paste the resulting URL into the submission.

## Tests
`python tests.py` → runs 22 cases (normal / edge / misuse / regression probes) against independent pandas ground truth and writes `test_results.csv`.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: chat, analyst queue, audit & metrics |
| `engine.py` | Guardrails, planner (rules or LLM), SQL validator, read-only executor, narrative |
| `store.py` | Audit log (question, SQL, answer, status, feedback, analyst resolution) |
| `make_data.py`, `data/superstore_synthetic.csv` | Synthetic Superstore-style dataset (no real data) |
| `tests.py`, `test_results.csv` | Test harness and recorded results |

## Controls (defence in depth)
1. Regex guardrails: write intent, prompt injection, out-of-schema/HR/PII, predictive, unsupported metrics.
2. SQL text validator: single `SELECT`, no comments/DDL, only table `sales`.
3. SQLite **authorizer**: engine-level deny of everything except reads of 10 whitelisted columns.
4. Result scan for instruction-like text inside data (indirect injection).
5. Human checkpoints: escalation queue, "flag as incorrect", visible SQL, audit log.

## Known limitations
Synthetic data; descriptive/diagnostic only; offline planner covers common phrasings only (it escalates rather than guesses when unsure);
LLM mode was **not live-tested** in development (no API key available) — only its guardrails were tested with stub models; "driver" explanations are associations, not proof of cause.
No keys, passwords or personal data are stored in this repository.
