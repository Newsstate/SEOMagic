# SEO Tool (Playwright, Render-ready)

Fully JS-rendered SEO scanner using **Playwright + FastAPI**. This bundle avoids `greenlet` entirely and pins **Python 3.11.9** via `render.yaml` to bypass cp313 build issues.

## Deploy on Render
1. Push to GitHub.
2. Create a **Web Service** and select this repo.
3. Render will use `render.yaml`:
   - Build installs Chromium with Playwright.
   - Start runs Uvicorn.
4. Open the service URL — the UI lives at `/`, API at `/api/analyze` (POST).

## Endpoints
- `GET /` — simple form UI.
- `POST /api/analyze` — body form or JSON:
  ```json
  {"url":"https://example.com", "mobile":true, "max_wait_ms":35000}
  ```

## Output (sample)
```json
{
  "input_url": "...",
  "final_url": "...",
  "http_status": 200,
  "timing_ms": 3120,
  "perf": {"domInteractive": 900, "domComplete": 2100},
  "title": "…",
  "meta": {"description":"…","robots":"…","viewport":"…","canonical":"…","hreflang":[...]},
  "headings": {"h1":["…"],"h2":[…],"h3":[…]},
  "links": {"internal":123,"external":45,"nofollow":8},
  "schema_types": ["Organization","BreadcrumbList"],
  "issues": ["Missing meta description"]
}
```

## Local dev
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
uvicorn app.main:app --reload
```

## Greenlet / Python 3.13 note
This template uses **Python 3.11** and **does not depend on greenlet**. If your existing project pulled `greenlet` via another dependency, pin Python to 3.11/3.12 or remove that dependency.

## Troubleshooting
- **OOM / crashes**: set `CONCURRENCY=2`.
- **Content not loaded**: increase `PAGE_TIMEOUT_MS` or add selector waits (see `browser_fetch.py`).
- **Headless blocked**: some sites block headless browsers; add `stealth` tactics if needed.
