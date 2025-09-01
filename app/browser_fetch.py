# app/browser_fetch.py
from __future__ import annotations
import asyncio, os, contextlib
from typing import Optional, Dict, Any, Tuple
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

CONCURRENCY = int(os.environ.get("CONCURRENCY", "4"))
PAGE_TIMEOUT_MS = int(os.environ.get("PAGE_TIMEOUT_MS", "35000"))
ENABLE_SCREENSHOTS = os.environ.get("ENABLE_SCREENSHOTS", "false").lower() == "true"

class PlaywrightPool:
    def __init__(self) -> None:
        self._pw = None
        self._browser: Optional[Browser] = None
        self._ready = asyncio.Event()
        self._lock = asyncio.Lock()
        self._sema = asyncio.Semaphore(CONCURRENCY)

    async def start(self) -> None:
        async with self._lock:
            if self._browser:
                self._ready.set()
                return
            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-accelerated-2d-canvas",
                    "--disable-setuid-sandbox",
                ],
            )
            self._ready.set()

    async def close(self) -> None:
        async with self._lock:
            with contextlib.suppress(Exception):
                if self._browser:
                    await self._browser.close()
            if self._pw:
                with contextlib.suppress(Exception):
                    await self._pw.stop()
            self._browser=None; self._pw=None; self._ready.clear()

    async def _new_page(self, mobile: bool=False) -> Tuple[BrowserContext, Page]:
        await self._ready.wait()
        assert self._browser is not None
        context = await self._browser.new_context(
            user_agent=(
              "Mozilla/5.0 (Linux; Android 12; Pixel 6) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/128.0.0.0 Mobile Safari/537.36"
            ) if mobile else (
              "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
            ),
            viewport={"width":390,"height":844} if mobile else {"width":1366,"height":800},
            device_scale_factor=3 if mobile else 1,
            is_mobile=mobile,
            has_touch=mobile,
            java_script_enabled=True,
            ignore_https_errors=True,
        )
        # Perf observers before navigation
        await context.add_init_script("""
        (() => {
          window.__metrics = { lcp: 0, cls: 0, fid: null, longTasks: 0 };
          try {
            new PerformanceObserver(list => {
              for (const e of list.getEntries()) {
                const t = e.renderTime || e.loadTime || 0;
                if (t > window.__metrics.lcp) window.__metrics.lcp = t;
              }
            }).observe({ type: 'largest-contentful-paint', buffered: true });
          } catch (e) {}

          try {
            new PerformanceObserver(list => {
              for (const e of list.getEntries()) {
                if (!e.hadRecentInput && e.value) window.__metrics.cls += e.value;
              }
            }).observe({ type: 'layout-shift', buffered: true });
          } catch (e) {}

          try {
            new PerformanceObserver(list => {
              const e = list.getEntries()[0];
              if (e) window.__metrics.fid = e.processingStart - e.startTime;
            }).observe({ type: 'first-input', buffered: true });
          } catch (e) {}

          try {
            new PerformanceObserver(list => {
              for (const e of list.getEntries()) window.__metrics.longTasks += e.duration;
            }).observe({ type: 'longtask', buffered: true });
          } catch (e) {}
        })();
        """)
        page = await context.new_page()
        page.set_default_timeout(PAGE_TIMEOUT_MS)
        return context, page

    async def fetch(self, url: str, *, mobile: bool=False, max_wait_ms: Optional[int]=None) -> Dict[str, Any]:
        timeout = max_wait_ms or PAGE_TIMEOUT_MS
        await self._sema.acquire()
        try:
            context, page = await self._new_page(mobile=mobile)
            try:
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                # Extra settle time for SPA/lazy content
                with contextlib.suppress(Exception):
                    await page.wait_for_load_state("networkidle", timeout=int(timeout*0.6))

                if ENABLE_SCREENSHOTS:
                    with contextlib.suppress(Exception):
                        await page.screenshot(path="last_screenshot.png", full_page=True)

                # Collect perf, network, PWA, images, and render-blocking info
                extras = await page.evaluate("""
                () => {
                  const paints = performance.getEntriesByType('paint').map(e => ({name: e.name, startTime: e.startTime||0}));
                  const resources = performance.getEntriesByType('resource').map(e => ({
                    name: e.name, initiatorType: e.initiatorType||'',
                    transferSize: e.transferSize||0, encodedBodySize: e.encodedBodySize||0,
                    decodedBodySize: e.decodedBodySize||0, startTime: e.startTime||0, duration: e.duration||0,
                  }));
                  const nav = (performance.getEntriesByType('navigation')[0]) || {};
                  const metrics = (window.__metrics || {});

                  // Manifest / SW hints
                  const manifestEl = document.querySelector('link[rel="manifest"]');
                  const manifest = manifestEl ? manifestEl.href : null;
                  let sw = false;
                  if ('serviceWorker' in navigator) {
                    try { sw = !!navigator.serviceWorker.controller || false; } catch(e) {}
                  }

                  // Media audit source: images present in DOM
                  const viewH = window.innerHeight || document.documentElement.clientHeight || 0;
                  const images = [...document.querySelectorAll('img')].map(img => {
                    const rect = img.getBoundingClientRect();
                    const srcAbs = img.currentSrc || img.src || null;
                    return {
                      src: srcAbs,
                      alt: img.getAttribute('alt') || null,
                      width_attr: img.getAttribute('width') || null,
                      height_attr: img.getAttribute('height') || null,
                      naturalWidth: img.naturalWidth || null,
                      naturalHeight: img.naturalHeight || null,
                      loading: img.loading || null,
                      belowFold: rect.top > viewH
                    };
                  });

                  // Render-blocking heuristics
                  const render_blocking = {
                    stylesheets: [...document.querySelectorAll('link[rel="stylesheet"]')].map(l => ({
                      href: l.href || null,
                      media: l.media || null,
                      media_matches: l.media ? window.matchMedia(l.media).matches : true,
                      disabled: !!l.disabled,
                      in_head: !!l.closest('head'),
                      is_blocking: (!l.media || window.matchMedia(l.media).matches) && !l.disabled
                    })),
                    scripts: [...document.querySelectorAll('script')].map(s => ({
                      src: s.src || null,
                      async: !!s.async,
                      defer: !!s.defer,
                      type: s.type || null,
                      in_head: !!s.closest('head'),
                      is_blocking: (!!s.closest('head') && !s.async && !s.defer && !(s.type||'').includes('module'))
                    }))
                  };

                  return { paints, resources, nav, metrics, manifest, sw, images, render_blocking };
                }
                """)

                final_url = page.url
                status = resp.status if resp else None
                html = await page.content()
                headers = {}
                with contextlib.suppress(Exception):
                    headers = resp.headers()

                return {
                    "final_url": final_url,
                    "status": status,
                    "html": html,
                    "headers": headers,
                    "extras": extras,
                }
            finally:
                with contextlib.suppress(Exception): await page.close()
                with contextlib.suppress(Exception): await context.close()
        finally:
            self._sema.release()

_pool: Optional[PlaywrightPool] = None

async def get_pool() -> PlaywrightPool:
    global _pool
    if _pool is None:
        _pool = PlaywrightPool()
        await _pool.start()
    return _pool

async def fetch_rendered(url: str, *, mobile: bool=False, max_wait_ms: Optional[int]=None) -> Dict[str, Any]:
    pool = await get_pool()
    return await pool.fetch(url, mobile=mobile, max_wait_ms=max_wait_ms)

async def shutdown_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
    _pool = None
