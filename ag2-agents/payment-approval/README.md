# AG2 Two-Agent Payment Approval

Demonstrates AG2's `human_input_mode="ALWAYS"` pattern as an approval gate
before triggering a Skyfire payment — the first example in this repo requiring
explicit user confirmation before a financial action.

Two agents collaborate:
- **researcher** — investigates the recipient and produces a risk assessment
- **payment_executor** — presents the assessment, pauses for human confirmation,
  then executes or aborts based on the response

## Key AG2 Features

- **`human_input_mode="ALWAYS"`** — executor pauses before every response; human
  types "yes" to proceed or "no" to abort — no custom routing logic needed
- **Two-agent `initiate_chat`** — researcher hands off to executor via the
  natural conversation flow; the shared message history carries the assessment

## Quick Start

```bash
cd ag2-agents/payment-approval
pip install -r requirements.txt
cp .env.example .env
python main.py
```
