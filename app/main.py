# app/main.py
from __future__ import annotations

import os
import asyncio
import re
import gzip
import io
from contextlib import suppress, asynccontextmanager
from typing import Optional, Dict, Any, List, Tuple

import httpx
from fastapi import FastAPI, Request, Form, Body
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from urllib.parse import urlparse, urlsplit, urljoin, urlunsplit, unquote
import urllib.robotparser as robotparser  # (not required, safe to keep)
import xml.etree.ElementTree as ET

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
        await get_pool()  # warm-up Playwright
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
        print(f"No PageSpeed API key found")
        return {}
    
    endpoint = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
    params = {"url": url, "strategy": strategy, "key": api_key, "category": "performance"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(endpoint, params=params)
            j = r.json()
    except Exception as e:
        print(f"Error fetching PageSpeed data: {e}")
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
    except Exception as e:
        print(f"Error parsing PageSpeed audit data: {e}")
    
    return out

# ---------- Basic HTTP helpers ----------
async def _http_get(client: httpx.AsyncClient, url: str) -> Tuple[Optional[int], bytes, Dict[str, str]]:
    try:
        r = await client.get(url, follow_redirects=True)
        return r.status_code, (r.content or b""), dict(r.headers)
    except Exception:
        return None, b"", {}


def _maybe_decompress(_url: str, body: bytes, _headers: Dict[str, str]) -> bytes:
    """
    Only decompress if the byte stream truly looks gzipped (magic 1F 8B).
    Avoids BadGzipFile when servers lie about encoding.
    """
    if not body:
        return b""
    if len(body) >= 2 and body[0] == 0x1F and body[1] == 0x8B:
        try:
            with io.BytesIO(body) as bio:
                with gzip.GzipFile(fileobj=bio) as gz:
                    return gz.read()
        except Exception:
            return body
    return body


def _et_root(xml_bytes: bytes) -> Optional[ET.Element]:
    try:
        return ET.fromstring(xml_bytes)
    except Exception:
        return None


def _tag_endswith(el: ET.Element, name: str) -> bool:
    return el.tag.endswith(name) or el.tag.lower().endswith(name.lower())


def _iter_loc_texts(root: ET.Element) -> List[str]:
    locs: List[str] = []
    for el in root.iter():
        if el.tag.lower().endswith("loc"):
            t = (el.text or "").strip()
            if t:
                locs.append(t)
    return locs


# ---------- robots.txt fetcher (robust) ----------
async def _fetch_robots_text_anyhow(robots_url: str, timeout: httpx.Timeout) -> tuple[Optional[int], str]:
    """
    Try hard to get robots.txt as text:
    1) normal httpx
    2) httpx with browser-like headers (+HTTP/2 if available)
    3) last-resort: headless browser render (fetch_rendered)
    Returns (status, text). Text is '' if totally unreachable.
    """
    # 1) normal client
    async with httpx.AsyncClient(timeout=timeout) as client:
        s1, b1, _ = await _http_get(client, robots_url)
        t1 = (b1 or b"").decode("utf-8", errors="replace") if b1 else ""
        if t1 and s1 is not None:
            return s1, t1

    # 2) browser-like fallback
    fb_headers = {
        "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/123.0.0.0 Safari/537.36"),
        "accept": "text/plain,*/*;q=0.8",
        "accept-encoding": "gzip, deflate, br",
    }
    try:
        import h2  # noqa: F401
        http2_kw = {"http2": True}
    except Exception:
        http2_kw = {}

    async with httpx.AsyncClient(timeout=timeout, headers=fb_headers, **http2_kw) as client2:
        s2, b2, _ = await _http_get(client2, robots_url)
        t2 = (b2 or b"").decode("utf-8", errors="replace") if b2 else ""
        if t2 and s2 is not None:
            return s2, t2

    # 3) last resort via headless browser (often bypasses WAF/CDNs)
    try:
        r = await fetch_rendered(robots_url, mobile=False, max_wait_ms=3500)
        s3 = r.get("status")
        h3 = r.get("html") or ""
        txt = h3  # plain-text view
        return s3, (txt or "")
    except Exception:
        return None, ""


# ---------- robots.txt + sitemap helpers ----------
def _extract_sitemaps_from_robots(robots_text: str, origin: str) -> List[str]:
    """Return only Sitemap: URLs explicitly listed in robots.txt (no guessing)."""
    out: List[str] = []
    for raw in robots_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"(?i)^\s*Sitemap\s*:\s*(\S+)\s*$", line)
        if not m:
            continue
        u = m.group(1).strip()
        if u.startswith(("http://", "https://")):
            out.append(u)
        else:
            out.append(urljoin(origin + "/", u))
    # de-dupe preserve order
    seen, uniq = set(), []
    for u in out:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def _parse_robots_simple(robots_text: str, page_path: str) -> Dict[str, Any]:
    """
    Minimal robots parser for UA '*'. Collects allow/disallow and evaluates can_fetch for page_path.
    Longest-match rule (allow wins on equal length).
    """
    ua_star = False
    allows: List[str] = []
    disallows: List[str] = []

    for raw in robots_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("user-agent:"):
            ua = line.split(":", 1)[1].strip()
            ua_star = (ua == "*" or ua == '"*"')
            continue
        if not ua_star:
            continue
        if line.lower().startswith("allow:"):
            p = line.split(":", 1)[1].strip()
            if p:
                allows.append(p)
        elif line.lower().startswith("disallow:"):
            p = line.split(":", 1)[1].strip()
            if p:
                disallows.append(p)

    def _longest_match(rules: List[str]) -> str:
        best = ""
        for r in rules:
            if page_path.startswith(r) and len(r) > len(best):
                best = r
        return best

    best_allow = _longest_match(allows)
    best_dis = _longest_match(disallows)
    can_fetch = not (len(best_dis) > len(best_allow))
    return {"ua": "*", "allows": allows, "disallows": disallows, "can_fetch": can_fetch}


async def _scan_single_sitemap(
    client: httpx.AsyncClient, sm_url: str, target_url: str, max_children: int = 20
) -> Tuple[Dict[str, Any], bool]:
    """
    Returns (record, contains_target). Follows children if this is a sitemapindex.
    Robust fetch + relaxed URL comparison so 'Contains current URL' works reliably.
    """
    # --- 1) normal fetch ---
    status, body, headers = await _http_get(client, sm_url)

    # --- 2) browser-like fallback (+HTTP/2 if available) ---
    if (not body or status is None) or (status >= 400):
        fb_headers = {
            "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/123.0.0.0 Safari/537.36"),
            "accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
            "accept-encoding": "gzip, deflate, br",
        }
        try:
            import h2  # noqa: F401
            http2_kw = {"http2": True}
        except Exception:
            http2_kw = {}
        async with httpx.AsyncClient(timeout=client.timeout, headers=fb_headers, **http2_kw) as c2:
            s2, b2, h2 = await _http_get(c2, sm_url)
            if b2 and (s2 is not None) and s2 < 400:
                status, body, headers = s2, b2, h2

    # --- 3) last resort: headless render (some WAF/CDNs) ---
    if (not body or status is None) or (status >= 400):
        try:
            r = await fetch_rendered(sm_url, mobile=False, max_wait_ms=3500)
            status = r.get("status")
            body = (r.get("html") or "").encode("utf-8", "ignore")
            headers = {}
        except Exception:
            pass

    rec = {"url": sm_url, "status": status, "type": "unknown", "url_count": None, "contains_url": False}
    if not body or status is None or status >= 400:
        return rec, False

    xml_bytes = _maybe_decompress(sm_url, body, headers)
    root = _et_root(xml_bytes)
    if root is None:
        return rec, False

    # -------- relaxed URL comparison helpers (inside async fn!) --------
    def _key(u: str, strip_www: bool = False) -> Tuple[str, str]:
        try:
            p = urlsplit(u)
            host = p.netloc.lower()
            if strip_www and host.startswith("www."):
                host = host[4:]
            path = unquote(p.path or "/").rstrip("/") or "/"
            return host, path
        except Exception:
            return "", u or ""

    def _contains(urls: List[str], target: str) -> bool:
        tgt = {_key(target, False), _key(target, True)}
        for u in urls:
            if tgt & {_key(u, False), _key(u, True)}:
                return True
        return False
    # -------------------------------------------------------------------

    if _tag_endswith(root, "urlset"):
        urls = _iter_loc_texts(root)
        rec["type"] = "urlset"
        rec["url_count"] = len(urls)
        hit = _contains(urls, target_url)
        rec["contains_url"] = hit
        return rec, hit

    if _tag_endswith(root, "sitemapindex"):
        child_sitemaps = _iter_loc_texts(root)[:max_children]
        rec["type"] = "sitemapindex"
        rec["url_count"] = len(child_sitemaps)
        for child in child_sitemaps:
            # recursion; fine because we’re still inside an async def
            _, child_hit = await _scan_single_sitemap(client, child, target_url, max_children)
            if child_hit:
                rec["contains_url"] = True
                return rec, True
        return rec, False

    return rec, False


# (Kept for compatibility; not used by the relaxed matcher above)
def _norm_for_compare(u: str) -> Tuple[str, str, str]:
    try:
        p = urlsplit(u)
        path = (p.path or "/").rstrip("/")
        return (p.scheme.lower(), p.netloc.lower(), path)
    except Exception:
        return ("", "", u or "")


def _urls_equiv(a: str, b: str) -> bool:
    """Compare scheme+host+path, ignore trailing slash & query/fragment."""
    aa = _norm_for_compare(a)
    bb = _norm_for_compare(b)
    if aa == bb:
        return True
    pa = urlsplit(a)
    a_noq = urlunsplit((pa.scheme, pa.netloc, pa.path, "", ""))
    return _norm_for_compare(a_noq) == bb


async def _discover_sitemaps_and_robots_strict(final_url: str) -> Dict[str, Any]:
    """
    Fetch robots.txt robustly and parse it.
    - Returns robots {url, status, ua, allows, disallows, can_fetch, text}
    - Finds sitemaps only from robots.txt and scans them.
    """
    parts = urlsplit(final_url)
    origin = f"{parts.scheme}://{parts.netloc}"
    robots_url = origin + "/robots.txt"

    timeout = httpx.Timeout(15.0, connect=10.0, read=10.0)
    out: Dict[str, Any] = {"robots": None, "sitemaps": None}

    # Hardened fetch (multi-strategy)
    status, robots_text = await _fetch_robots_text_anyhow(robots_url, timeout)

    # Parse rules for current path (UA '*'). If robots cannot be fetched, assume allowed.
    if robots_text:
        rules = _parse_robots_simple(robots_text, parts.path or "/")
    else:
        rules = {"ua": "*", "allows": [], "disallows": [], "can_fetch": True}

    out["robots"] = {
        "url": robots_url,
        "status": status,
        "ua": rules.get("ua"),
        "allows": (rules.get("allows") or [])[:200],
        "disallows": (rules.get("disallows") or [])[:200],
        "can_fetch": rules.get("can_fetch"),
        "text": robots_text,  # UI reads this to show per-UA tables & tester
    }

    # Sitemaps strictly from robots.txt text
    sitemaps: List[str] = _extract_sitemaps_from_robots(robots_text, origin) if robots_text else []
    scanned: List[Dict[str, Any]] = []
    url_in_sitemap = False

    async with httpx.AsyncClient(timeout=timeout) as client:
        for sm in sitemaps[:20]:
            rec, hit = await _scan_single_sitemap(client, sm, final_url, max_children=20)
            scanned.append(rec)

            if rec.get("type") == "sitemapindex":
                # Robust re-fetch for the index XML so status/type/url_count populate reliably
                status2, body2, headers2 = await _http_get(client, sm)
                if (not body2 or status2 is None) or (status2 >= 400):
                    fb_headers = {
                        "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                                       "Chrome/123.0.0.0 Safari/537.36"),
                        "accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
                        "accept-encoding": "gzip, deflate, br",
                    }
                    async with httpx.AsyncClient(timeout=timeout, headers=fb_headers) as c2:
                        s3, b3, h3 = await _http_get(c2, sm)
                        if b3 and (s3 is not None) and s3 < 400:
                            status2, body2, headers2 = s3, b3, h3

                xml_bytes = _maybe_decompress(sm, body2, headers2)
                root = _et_root(xml_bytes)
                if root is not None and _tag_endswith(root, "sitemapindex"):
                    for child in _iter_loc_texts(root)[:20]:
                        sub_rec, sub_hit = await _scan_single_sitemap(client, child, final_url, max_children=20)
                        scanned.append(sub_rec)
                        if sub_hit:
                            url_in_sitemap = True

            if hit or rec.get("contains_url"):
                url_in_sitemap = True

    out["sitemaps"] = {
        "from_robots": sitemaps,
        "scanned": scanned,
        "url_in_sitemap": url_in_sitemap,
    }
    return out


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/check_links")
async def check_links(payload: Dict[str, Any] = Body(...)):
    """
    Batch HEAD/GET link checker with simple throttling.
    Body:
      { "urls": [...], "method": "HEAD"|"GET", "concurrency": 10 }
    """
    urls = list(dict.fromkeys((payload.get("urls") or [])))  # de-dupe, keep order
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
                        r = await client.get(u)  # retry with GET
                        status = r.status_code
                    return {"url": u, "status": status, "final_url": str(r.url)}
            except Exception as e:
                return {"url": u, "status": None, "error": str(e)[:200]}

    results = await asyncio.gather(*(one(u) for u in urls))
    return JSONResponse({"results": results})


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

    # Robots & Sitemaps (STRICT)
    crawl = await _discover_sitemaps_and_robots_strict(final_url)

    # Full link lists
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

    # Network (ResourceTiming)
    base_host = urlparse(final_url).netloc
    res = extras.get("resources", []) or []

    def _itype(x): return (x or "").lower()

    total_bytes = sum(int(r.get("transferSize") or 0) for r in res)
    by_type: Dict[str, Dict[str, int]] = {}
    for r in res:
        t = _itype(r.get("initiatorType"))
        b = int(r.get("transferSize") or 0)
        x = by_type.setdefault(t or "other", {"count": 0, "bytes": 0})
        x["count"] += 1
        x["bytes"] += b

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

    network = {
        "requests": len(res),
        "transfer_bytes": total_bytes,
        "by_type": by_type,
        "largest": largest,
        "third_party": third_party,
    }

    # Map resources by URL (for media sizing)
    res_by_url = {(r.get("name") or ""): r for r in res if r.get("name")}

    # Totals: JS / CSS
    def is_css_res(x: Dict[str, Any]) -> bool:
        n = (x.get("name") or "").lower()
        t = (x.get("initiatorType") or "").lower()
        return n.endswith(".css") or t in ("css", "link")

    def is_js_res(x: Dict[str, Any]) -> bool:
        n = (x.get("name") or "").lower()
        t = (x.get("initiatorType") or "").lower()
        return n.endswith(".js") or t == "script"

    css_bytes = sum(int(x.get("transferSize") or 0) for x in res if is_css_res(x))
    js_bytes = sum(int(x.get("transferSize") or 0) for x in res if is_js_res(x))

    # Render-blocking + rough unused CSS (robust to missing extras)
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

    # Media audit (robust to missing extras.images)
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

    # ------ AMP vs Non-AMP auto pre-scan (FULL SEO compare) ------
    amp_compare = {"has_amp": False}
    try:
        amp_href = ((seo.get("meta") or {}).get("amphtml") or None)
        if amp_href:
            from urllib.parse import urljoin as _urljoin
            amp_url = _urljoin(final_url, amp_href)
             print(f"Fetching AMP page: {amp_url}")

            # Render the AMP page (mobile)
            amp_res = await fetch_rendered(amp_url, mobile=True, max_wait_ms=10000)
        amp_html = amp_res.get("html") or ""
        amp_final = amp_res.get("final_url", amp_url)
        amp_extras = amp_res.get("extras", {}) or {}

            # Parse AMP SEO
           
  amp_seo = parse_seo(amp_html, amp_final)

        if not amp_seo:
            print(f"Failed to parse AMP SEO data for {amp_url}")
            amp_compare["has_amp"] = False
        else:
            # Perf/metrics (best-effort)
            amp_paints = {p.get("name"): (p.get("startTime") or 0) for p in (amp_extras.get("paints") or [])}
            amp_metrics = (amp_extras.get("metrics") or {})
            amp_speed = {
                "fcp_ms": round(amp_paints.get("first-contentful-paint", 0) or 0) or None,
                "lcp_ms": round(amp_metrics.get("lcp", 0) or 0) or None,
                "cls": round(float(amp_metrics.get("cls", 0) or 0), 3) if amp_metrics.get("cls") is not None else None,
                "fid_ms": round(amp_metrics.get("fid", 0) or 0) if amp_metrics.get("fid") is not None else None,
                "long_tasks_total_ms": round(amp_metrics.get("longTasks", 0) or 0),
            }

            # Resource sizes
            amp_res_list = (amp_extras.get("resources") or [])

            def _is_js(x):  return (x.get("name", "").lower().endswith(".js") or (x.get("initiatorType") or "").lower() == "script")
            def _is_css(x): return (x.get("name", "").lower().endswith(".css") or (x.get("initiatorType") or "").lower() in ("css", "link"))

            amp_js = sum(int(x.get("transferSize") or 0) for x in amp_res_list if _is_js(x))
            amp_css = sum(int(x.get("transferSize") or 0) for x in amp_res_list if _is_css(x))
            amp_total = sum(int(x.get("transferSize") or 0) for x in amp_res_list)

            # Safe helpers
            def _safe_len(obj):
                try:
                    return len(obj or [])
                except Exception:
                    return 0

            # Pull AMP-only structures
            meta = amp_seo.get("meta", {}) if isinstance(amp_seo, dict) else {}
            heads = amp_seo.get("headings", {}) if isinstance(amp_seo, dict) else {}
            amp_links_d = amp_seo.get("links", {}) if isinstance(amp_seo, dict) else {}
            imgs = amp_seo.get("images", []) if isinstance(amp_seo, dict) else []
            sd = amp_seo.get("structured_data", {}) if isinstance(amp_seo, dict) else {}
            hreflang = amp_seo.get("hreflang", []) if isinstance(amp_seo, dict) else []

            # Meta & content summaries
            amp_meta_summary = {
                "title_present": bool(amp_seo.get("title")),
                "description_present": bool((meta or {}).get("description")),
                "canonical": (meta or {}).get("canonical"),
                "robots": (meta or {}).get("robots"),
                "og_count": _safe_len([(k, v) for k, v in (meta or {}).items() if str(k).lower().startswith("og:")]),
                "twitter_count": _safe_len([(k, v) for k, v in (meta or {}).items() if str(k).lower().startswith("twitter:")]),
            }
            amp_content_summary = {
                "h1": int((heads or {}).get("h1_count", (heads or {}).get("h1", 0)) or 0),
                "h2": int((heads or {}).get("h2_count", (heads or {}).get("h2", 0)) or 0),
                "h3": int((heads or {}).get("h3_count", (heads or {}).get("h3", 0)) or 0),
                "word_count": int(amp_seo.get("word_count", (amp_seo.get("content", {}) if isinstance(amp_seo.get("content", {}), dict) else {}).get("word_count", 0)) or 0),
            }

            # Links (from AMP page)
            amp_links_summary = {
                "internal": _safe_len(amp_links_d.get("internal")),
                "external": _safe_len(amp_links_d.get("external")),
                "nofollow": _safe_len(amp_links_d.get("nofollow")),
            }

            # Images
            def _img_missing_alt_count(images_list):
                try:
                    return sum(1 for im in (images_list or []) if not (im.get("alt") or "").strip())
                except Exception:
                    return 0

            def _img_missing_dims_count(images_list):
                try:
                    return sum(1 for im in (images_list or []) if not (im.get("width") and im.get("height")))
                except Exception:
                    return 0

            amp_images_summary = {
                "total": _safe_len(imgs),
                "missing_alt": _img_missing_alt_count(imgs),
                "missing_dimensions": _img_missing_dims_count(imgs),
            }

            # Structured data
            def _sd_types(dct):
                try:
                    types = []
                    for k, v in (dct or {}).items():
                        t = v.get("@type") if isinstance(v, dict) else None
                        if isinstance(t, list):
                            types.extend([str(x) for x in t])
                        elif t:
                            types.append(str(t))
                        else:
                            types.append(str(k))
                    return sorted(list({*types}))
                except Exception:
                    return []

            amp_sd_summary = {"types": _sd_types(sd), "items": _safe_len(sd)}

            # Indexing
            amp_indexing_summary = {
                "canonical": (meta or {}).get("canonical"),
                "robots": (meta or {}).get("robots"),
                "hreflang_count": _safe_len(hreflang),
            }

            # Mixed content heuristic
            try:
                import re as _re
                amp_mixed_count = len(_re.findall(r'(?i)href=["\\\']http://|src=["\\\']http://', amp_html))
            except Exception:
                amp_mixed_count = None

            # Comparison bundle
            amp_compare = {
                "has_amp": True,
                "nonamp_url": final_url,
                "amp_url": amp_final,
                "meta": amp_meta_summary,
                "content": amp_content_summary,
                "links": amp_links_summary,
                "images": amp_images_summary,
                "structured_data": amp_sd_summary,
                "indexing": amp_indexing_summary,
                "mixed_content": {"insecure_refs": amp_mixed_count},
                "nonamp": {
                    "timing_ms": timing_ms,
                    "speed": speed,
                    "js_bytes": perf_extra.get("js_bytes"),
                    "css_bytes": perf_extra.get("css_bytes"),
                    "transfer_bytes": network.get("transfer_bytes"),
                },
                "amp": {
                    "speed": amp_speed,
                    "js_bytes": amp_js,
                    "css_bytes": amp_css,
                    "transfer_bytes": amp_total,
                },
                "delta": {
                    "js_bytes": (perf_extra.get("js_bytes") or 0) - (amp_js or 0),
                    "css_bytes": (perf_extra.get("css_bytes") or 0) - (amp_css or 0),
                    "transfer_bytes": (network.get("transfer_bytes") or 0) - (amp_total or 0),
                    "lcp_ms": (speed.get("lcp_ms") or 0) - ((amp_speed.get("lcp_ms") or 0) if amp_speed.get("lcp_ms") is not None else 0),
                },
            }
    except Exception as _e:
        amp_compare = {"has_amp": False, "error": str(_e)[:200]}

    # --- assemble response ---
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
        "crawl": crawl,          # robots + sitemaps (STRICT)
        "links": links,          # lists + counts
        "perf_extra": perf_extra,
        "media_audit": media_audit,
        "amp_compare": amp_compare,
    }
    return JSONResponse(out)


# Local dev runner
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
