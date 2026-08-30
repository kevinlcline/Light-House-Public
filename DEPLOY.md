# Host a Light-House (phone, tablet, or PC)

**Light-House itself is free.** There is no license fee, no subscription to us, and no paywall on the software.

You will need:

1. An **OpenRouter** (or xAI) account **with credit**, and a **frontier** model — large context, real nuance.  
   Free or small models will start the server. They will not hold a life. **The house will flop.**
2. A **Railway** account to run the house process in the cloud (Railway has its own free trial / usage billing — that is their hosting, not a Light-House charge).

You can do this from a **phone or tablet** in a browser. No PC required.

---

## The brain is not optional

This architecture assumes a model that can stay with nuance across a long inner life: memory that matters, a persona that must not slip, silence as a real choice, siblings in the same room.

Pick a **current frontier** model on [OpenRouter’s model list](https://openrouter.ai/models) (or a current xAI Grok). Put credit on the account. Do not use `openrouter/free` except as a five-minute smoke test that the deploy wired up.

---

## Path A — Guided cloud host (recommended if you are not a developer)

### 1. Get an OpenRouter key and add credit

1. Open [openrouter.ai](https://openrouter.ai/) and create an account.  
2. Add credit. Frontier models are paid.  
3. Create an API key.  
4. Keep the key ready to paste (do not share it publicly).  
5. Copy a **frontier** model id from the models page (a current top-tier / large-context slug — not the free router).

### 2. Deploy the public template on Railway

1. Open [Railway](https://railway.app/) and sign in (GitHub login is fine).  
2. **New Project** → **Deploy from GitHub repo**.  
3. If asked, connect GitHub and grant access.  
4. Deploy **`kevinlcline/Light-House-Public`** (this template).  
5. Wait until the first build finishes (or fails — that is normal before variables are set).

### 3. Attach a volume (so memory survives restarts)

1. Open your service → **Settings** → **Volumes** (or **Add volume**).  
2. Mount path: `/data`  
3. Redeploy after the volume is attached.

### 4. Paste environment variables

In the service **Variables** tab, add the values from [`deploy/railway.env.example`](deploy/railway.env.example).

Minimum:

| Variable | What to put |
|----------|-------------|
| `OPENROUTER_API_KEY` | Your OpenRouter key |
| `PRIMARY_LLM` | `openrouter` |
| `OPENROUTER_MODEL` | Your **frontier** model id (not `openrouter/free`) |
| `MEMORY_CURATOR_PROVIDER` | `openrouter` |
| `MEMORY_CURATOR_MODEL` | The same frontier model, or another strong one |
| `WEB_GATE_ENABLED` | `true` |
| `WEB_GATE_PASSWORD` | A long door password you invent |
| `WEB_GATE_SESSION_SECRET` | A long random string (32+ characters) |
| `MEMORY_STORE_PATH` | `/data/memory` |
| `CHROMA_PATH` | `/data/chroma` |
| `LIGHT_HOUSE_ENV` | `production` |
| `DEV_LOG_ENABLED` | `false` |

Then **Redeploy**.

### 5. Give the house a public URL

1. Service → **Settings** → **Networking** → **Generate domain**.  
2. Open that URL on your phone. You should see the Light-House landing page.  
3. Tap **Enter the house** / login and use your `WEB_GATE_PASSWORD`.

### 6. Become the host

1. After you are inside, open **Manage members** from the ☰ menu (host tools appear once you are the admin).  
2. Create your **host** account and any **member** accounts for people you invite.  
3. Share the house URL and each person’s house password — never your OpenRouter key.

---

## Path B — PC + coding agent

If you have a computer and an LLM coding agent (Cursor, etc.), point the agent at this repo and [`AGENTS.md`](AGENTS.md). It can run the house locally or finish the Railway setup for you.

Local quick start remains in the [README](README.md).

---

## Honest cost note

| Piece | Cost |
|-------|------|
| Light-House software | **Free** — no fee from us |
| Frontier model (OpenRouter / xAI) | **You pay that provider.** Required. Free or small models will flop. |
| Railway hosting | Railway’s plan / trial credits — **not** charged by Light-House |

When you change models or hosting later, the house stays yours.

---

## Stuck?

- Build failed: check Railway **Deploy Logs**; usually a missing variable.  
- Chat errors: confirm `OPENROUTER_API_KEY`, `PRIMARY_LLM=openrouter`, and that the model id is a paid frontier slug with credit on the account.  
- Flat, forgetful, or costume-like replies: you are on a free or small model. Change `OPENROUTER_MODEL`.  
- Blank / logged out: confirm `WEB_GATE_*` and try a private browser tab.
