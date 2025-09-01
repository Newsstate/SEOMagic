# app/main.py
from __future__ import annotations

import os
import asyncio
from contextlib import suppress, asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
import httpx

from .browser_fetch import get_pool, fetch_rendered, shutdown_pool
from .seo import parse_seo
from app.extractors.links import collect_links_from_html

# ---------- Windows event-loop fix (Playwright needs Proactor on Windows) ----------
if os.name == "nt":
    with suppress(Exception):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

templates = Jinja2Templates(directory="app/templates")

# ---------- FastAPI app with lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    with suppress(Exception):
        await get_pool()            # warm-up Playwright
    yield
    with suppress(Exception):
        await shutdown_pool()

app = FastAPI(lifespan=lifespan)

# Optional: silence Chrome devtools probe logs
@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def chrome_devtools_probe():
    return Response(status_code=204)

# ---------- PageSpeed fallback (optional) ----------
async def fetch_pagespeed(url: str, strategy: str = "desktop") -> Dict[str, Any]:
    api_key = os.environ.get("PSI_API_KEY") or os.environ.get("PAGESPEED_API_KEY")
    if not api_key:
        return {}
    endpoint = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
    params = {"url": url, "strategy": strategy, "key": api_key, "category": "performance"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(endpoint, params=params)
            j = r.json()
    except Exception:
        return {}

    out: Dict[str, Any] = {}
    try:
        lh = (j.get("lighthouseResult") or {})
        audits = lh.get("audits") or {}
        out["fcp_ms"] = round(audits.get("first-contentful-paint", {}).get("numericValue") or 0)
        out["lcp_ms"] = round(audits.get("largest-contentful-paint", {}).get("numericValue") or 0)
        out["cls"] = round(float(audits.get("cumulative-layout-shift", {}).get("numericValue") or 0), 3)
        out["fid_ms"] = round(audits.get("max-potential-fid", {}).get("numericValue") or 0)
        out["long_tasks_total_ms"] = round(audits.get("total-blocking-time", {}).get("numericValue") or 0)
    except Exception:
        pass

    try:
        m = (j.get("loadingExperience") or {}).get("metrics") or {}
        if m:
            out["fcp_ms"] = m.get("FIRST_CONTENTFUL_PAINT_MS", {}).get("percentile") or out.get("fcp_ms")
            out["lcp_ms"] = m.get("LARGEST_CONTENTFUL_PAINT_MS", {}).get("percentile") or out.get("lcp_ms")
            cls_p = m.get("CUMULATIVE_LAYOUT_SHIFT_SCORE", {}).get("percentile")
            if cls_p is not None:
                out["cls"] = round(float(cls_p) / 100.0, 3)
            inp = m.get("INTERACTION_TO_NEXT_PAINT", {}).get("percentile")
            if inp:
                out["fid_ms"] = inp
    except Exception:
        pass
    return out

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze")
async def analyze(
    request: Request,
    url: str = Form(...),
    mobile: Optional[bool] = Form(False),
    max_wait_ms: Optional[int] = Form(None),
    rendered: Optional[bool] = Form(True),
):
    # Rendered fetch (Playwright)
    rendered_res = await fetch_rendered(url, mobile=bool(mobile), max_wait_ms=max_wait_ms)
    html = rendered_res.get("html", "") or ""
    final_url = rendered_res.get("final_url", url)
    status = rendered_res.get("status")
    headers = rendered_res.get("headers", {}) or {}
    extras = rendered_res.get("extras", {}) or {}

    # Parse SEO
    seo = parse_seo(html, final_url)

    # Full link lists (NEW)
    links = collect_links_from_html(html, final_url)

    # Basic transport
    page_size_bytes = len(html.encode("utf-8"))
    proto = "HTTPS" if final_url.lower().startswith("https://") else "HTTP"
    http_version = None  # not exposed

    # Speed (from observers/paints)
    paints = {p.get("name"): (p.get("startTime") or 0) for p in extras.get("paints", [])}
    metrics = extras.get("metrics", {}) or {}
    speed = {
        "fcp_ms": round(paints.get("first-contentful-paint", 0) or 0) or None,
        "lcp_ms": round(metrics.get("lcp", 0) or 0) or None,
        "cls": round(float(metrics.get("cls", 0) or 0), 3) if metrics.get("cls") is not None else None,
        "fid_ms": round(metrics.get("fid", 0) or 0) if metrics.get("fid") is not None else None,
        "long_tasks_total_ms": round(metrics.get("longTasks", 0) or 0),
    }

from fastapi import Body

@app.post("/api/check_links")
async def check_links(payload: Dict[str, Any] = Body(...)):
    urls = list(dict.fromkeys((payload.get("urls") or [])))  # de-dupe, preserve order
    method = (payload.get("method") or "HEAD").upper()
    concurrency = max(1, min(int(payload.get("concurrency") or 10), 20))
    timeout = httpx.Timeout(10.0, read=10.0, connect=10.0)
    sem = asyncio.Semaphore(concurrency)

    async def one(u: str):
        async with sem:
            try:
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    r = await client.request(method, u)
                    status = r.status_code
                    if status in (405, 501) and method == "HEAD":
                        r = await client.get(u)
                        status = r.status_code
                    return {"url": u, "status": status, "final_url": str(r.url)}
            except Exception as e:
                return {"url": u, "status": None, "error": str(e)[:150]}

    results = await asyncio.gather(*(one(u) for u in urls))
    return JSONResponse({"results": results})


    # Network (ResourceTiming)
    from urllib.parse import urlparse
    base_host = urlparse(final_url).netloc
    res = extras.get("resources", []) or []
    def _itype(x): return (x or "").lower()
    total_bytes = sum(int(r.get("transferSize") or 0) for r in res)
    by_type: Dict[str, Dict[str, int]] = {}
    for r in res:
        t = _itype(r.get("initiatorType"))
        b = int(r.get("transferSize") or 0)
        x = by_type.setdefault(t or "other", {"count": 0, "bytes": 0})
        x["count"] += 1; x["bytes"] += b
    largest = sorted(
        [{"name": r.get("name"), "type": _itype(r.get("initiatorType")), "bytes": int(r.get("transferSize") or 0)} for r in res],
        key=lambda x: x["bytes"], reverse=True
    )[:10]
    from collections import defaultdict
    tp = defaultdict(lambda: {"count": 0, "bytes": 0})
    for r in res:
        host = urlparse(r.get("name") or "").netloc
        if host and base_host and host != base_host:
            tp[host]["count"] += 1
            tp[host]["bytes"] += int(r.get("transferSize") or 0)
    third_party = sorted([{"domain": k, **v} for k, v in tp.items()], key=lambda x: x["bytes"], reverse=True)[:10]
    network = {"requests": len(res), "transfer_bytes": total_bytes, "by_type": by_type, "largest": largest, "third_party": third_party}


    # --- Map resources by URL for quick lookups (for media sizing) ---
    res_by_url = {}
    for r in res:
        name = r.get("name")
        if name:
            res_by_url[name] = r

    # --- Totals: JS / CSS bytes (rough via initiatorType or file extension) ---
    def is_css_res(x: Dict[str, Any]) -> bool:
        n = (x.get("name") or "").lower()
        t = (x.get("initiatorType") or "").lower()
        return n.endswith(".css") or t in ("css", "link")

    def is_js_res(x: Dict[str, Any]) -> bool:
        n = (x.get("name") or "").lower()
        t = (x.get("initiatorType") or "").lower()
        return n.endswith(".js") or t == "script"

    css_bytes = sum(int(x.get("transferSize") or 0) for x in res if is_css_res(x))
    js_bytes  = sum(int(x.get("transferSize") or 0) for x in res if is_js_res(x))

    # --- Render-blocking lists (from page DOM) + rough "unused css" by media mismatch ---
    rb = extras.get("render_blocking") or {"stylesheets": [], "scripts": []}
    unused_css_bytes = 0
    for st in rb.get("stylesheets", []):
        if st.get("href") and st.get("media") and not st.get("media_matches"):
            b = int((res_by_url.get(st["href"]) or {}).get("transferSize") or 0)
            unused_css_bytes += b
    unused_css_pct_rough = round((unused_css_bytes / css_bytes) * 100, 1) if css_bytes else None

    perf_extra = {
        "js_bytes": js_bytes,
        "css_bytes": css_bytes,
        "unused_css_pct_rough": unused_css_pct_rough,
        "render_blocking": rb,
    }

    # --- Media audit: annotate images with transfer size, then flag ---
    images = extras.get("images") or []
    for im in images:
        b = int((res_by_url.get(im.get("src") or "") or {}).get("transferSize") or 0)
        im["bytes"] = b

    media_audit = {
        "large": sorted([im for im in images if im.get("bytes", 0) >= 100 * 1024],
                        key=lambda x: x.get("bytes", 0), reverse=True)[:50],
        "missing_dimensions": [im for im in images if not (im.get("width_attr") and im.get("height_attr"))][:50],
        "below_fold_no_lazy": [im for im in images if im.get("belowFold") and (im.get("loading") != "lazy")][:50],
    }

    # Security/Indexing headers
    h_lower = {k.lower(): v for k, v in headers.items()}
    security = {
        "hsts": bool(h_lower.get("strict-transport-security")),
        "csp": bool(h_lower.get("content-security-policy")),
        "x_frame_options": h_lower.get("x-frame-options"),
    }
    indexing = {"x_robots_tag": h_lower.get("x-robots-tag")}

    # PWA hints
    pwa = {"has_manifest": bool(extras.get("manifest")), "service_worker": bool(extras.get("sw")), "manifest_url": extras.get("manifest")}

    # Perf summary
    nav = extras.get("nav", {}) or {}
    perf = {"domInteractive": nav.get("domInteractive"), "domComplete": nav.get("domComplete")}
    timing_ms = speed.get("lcp_ms") or perf.get("domComplete") or 0

    # PSI fallback if needed
    if not any([speed.get("fcp_ms"), speed.get("lcp_ms"), speed.get("cls"), speed.get("fid_ms")]):
        psi = await fetch_pagespeed(final_url, strategy="mobile" if mobile else "desktop")
        for k in ("fcp_ms", "lcp_ms", "cls", "fid_ms", "long_tasks_total_ms"):
            if psi.get(k):
                speed[k] = psi[k]
        timing_ms = speed.get("lcp_ms") or timing_ms

    out = {
        "input_url": url,
        "final_url": final_url,
        "http_status": status,
        "page_size_bytes": page_size_bytes,
        "protocol": proto,
        "http_version": http_version,
        "timing_ms": timing_ms,
        "perf": perf,
        "speed": speed,
        "network": network,
        "security": security,
        "indexing": indexing,
        "pwa": pwa,
        **seo,
        "links": links,  # <<— lists + counts
        "perf_extra": perf_extra,     # ✅ new
        "media_audit": media_audit,   # ✅ new
    }
    return JSONResponse(out)

# Local dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
