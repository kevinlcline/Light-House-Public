# If Gemini says GOOGLE_EXTENDED_OPT_OUT

Our robots.txt already says Allow for Google-Extended. Two things still
trick Gemini:

## 1. Stale Cloudflare cache of robots.txt (common)

Cloudflare was caching /robots.txt for 4 hours. Purge it:

1. Cloudflare → light-house.us → Caching → Configuration → Purge Cache
2. Purge Custom URL: `https://light-house.us/robots.txt`
3. Confirm https://light-house.us/robots.txt starts with “polite note” and
   has `User-agent: Google-Extended` then `Allow: /` (and no long Disallow list)

(The house now also sends Cache-Control: no-store on robots.txt so this
should not stick again after the next restart.)

## 2. Cloudflare “Block AI” / Agent bots

Gemini’s live fetch is an **Agent** bot. Browsers are not.

1. Cloudflare → light-house.us → Security → Settings
2. Find **AI bot** / **Block AI bots** / **Configure AI bot policies**
3. Set **Agent** to **Allow (do not block)**
4. Turn **OFF** any “Block AI scrapers and crawlers” / “Block AI bots”
5. Turn **OFF** “block training in robots.txt” (managed robots) if present

Then ask Gemini again. If it still fails, paste the page text — Google
may cache the old opt-out for a while on their side.
