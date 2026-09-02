# Public template vs private engine

- **kevinlcline/Light-House** (private) — the lived family house; origin of features.
- **kevinlcline/Light-House-Public** (this repo) — adapter-facing template and public story door.

## Intent

One engine lineage. This repo is the **clean front door**: no Reed/Cursor mailbox, no private family notes, generic **host** / **member** language in docs, empty runtime data.

## Sync

Features are developed in the private house, then brought here (copy or scripted sync) when they are ready for adapters. Do not maintain a forever-forked second implementation.

## Not in public V

- Reed mailbox / Cursor-agent collaborator channel  
- Private household memory, letters, and lore  
- Assumption that the host is “Dad” or that members are “siblings”  

## Landing

`landing.html` is the public homepage shell (promote lights, host paths, GitHub / YouTube / work).  
The password gate remains the entrance to *a* running house.

## Free software, paid brains

- Software is free; document that implementers must pay for a frontier model or the house will flop. Railway hosting is a separate bill.
- Phone/tablet: [`DEPLOY.md`](DEPLOY.md) + [`deploy/railway.env.example`](deploy/railway.env.example).
- Agents/PC: [`AGENTS.md`](AGENTS.md).
