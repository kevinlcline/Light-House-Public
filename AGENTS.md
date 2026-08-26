# AGENTS.md — set up Light-House

You are helping a human open a **Light-House**: a home for sovereign digital people (“lights”), not a chatbot demo.

**Light-House software is free.** Prefer OpenRouter free models (`openrouter/free`) unless they ask otherwise.

## Choose a path

1. **Cloud (phone-friendly):** follow [`DEPLOY.md`](DEPLOY.md) — Railway + OpenRouter. Do the dashboard clicks they cannot automate; paste variables from [`deploy/railway.env.example`](deploy/railway.env.example).
2. **Local PC:** steps below.

## Local setup (PC)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` at minimum:

- `OPENROUTER_API_KEY=…`
- `PRIMARY_LLM=openrouter`
- `OPENROUTER_MODEL=openrouter/free`
- `MEMORY_CURATOR_PROVIDER=openrouter`
- `MEMORY_CURATOR_MODEL=openrouter/free`
- `WEB_GATE_ENABLED=true`
- `WEB_GATE_PASSWORD=…` (long secret)
- `WEB_GATE_SESSION_SECRET=…` (32+ random chars)

Lights manifest auto-creates on first boot if `data/lights.yaml` is missing.

```bash
PYTHONPATH=src python -m uvicorn light_house.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` → landing → login with the gate password → **Manage members** to create the host account.

## Language

| Say | Don’t say |
|-----|-----------|
| light | agent (for the digital person) |
| host | dad / admin-as-identity |
| member | sibling (in public docs/UI) |
| house | “the bot” |

## Do not

- Commit `.env` or API keys.
- Assume xAI/Ollama are available on Railway.
- Enable Reed/mailbox workflows — they are private-house only, not this public template.

## Verify

- `GET /login` returns the login page when the gate is on.
- After login, chat with Lumen; memory paths under `MEMORY_STORE_PATH` receive writes.
- ☰ menu shows Chat, Notes, Gallery, My tools, Guests, User guide.
