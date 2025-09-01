from __future__ import annotations
from typing import Dict, Any, List
from urllib.parse import urljoin, urlparse
from selectolax.parser import HTMLParser
from bs4 import BeautifulSoup
import re
from collections import Counter

STOPWORDS = set("""
the a an and or but if then else for to of in on at by with from this that those these your our is are was were be been being as not no yes it its you we i
he she they them his her their ours mine yours our it's
""".split())

def _pairs(prefix: str, soup: BeautifulSoup) -> List[str]:
    out = []
    for m in soup.find_all("meta"):
        name = (m.get("property") or m.get("name") or "").lower()
        if name.startswith(prefix):
            val = (m.get("content") or "").strip()
            if val:
                out.append(f"{name.split(':',1)[-1]}: {val}")
    return out

def _keyword_density(text: str, top_n: int = 10) -> List[Dict[str, Any]]:
    words = re.findall(r"[A-Za-z\u0900-\u097F']{2,}", (text or "").lower())
    words = [w for w in words if w not in STOPWORDS]
    cnt = Counter(words)
    total = sum(cnt.values()) or 1
    return [{"word": w, "count": c, "percent": c/total} for w,c in cnt.most_common(top_n)]

def parse_seo(html: str, base_url: str) -> Dict[str, Any]:
    tree = HTMLParser(html)
    soup = BeautifulSoup(html, "lxml")

    # Title / meta basics
    title = tree.css_first("title").text().strip() if tree.css_first("title") else None
    meta_desc = meta_robots = meta_viewport = ""
    for m in soup.find_all("meta"):
        name = (m.get("name") or m.get("property") or "").lower()
        if name == "description":
            meta_desc = (m.get("content") or "").strip()
        elif name == "robots":
            meta_robots = (m.get("content") or "").strip()
        elif name == "viewport":
            meta_viewport = (m.get("content") or "").strip()

    # Links in <head>
    favicon = amphtml = canonical = None
    hreflangs = []
    for link in soup.find_all("link"):
        rels = [r.lower() for r in (link.get("rel") or [])]
        href = (link.get("href") or "").strip()
        if "icon" in rels: favicon = href
        if "canonical" in rels and href: canonical = href
        if "alternate" in rels and link.get("hreflang"):
            hreflangs.append({"lang": link.get("hreflang"), "href": href})
        if "amphtml" in rels: amphtml = href

    # Headings
    headings = {h: [el.get_text(strip=True) for el in soup.find_all(h)] for h in ["h1","h2","h3","h4","h5","h6"]}

    # Links counts
    a_tags = soup.find_all("a")
    internal = external = nofollow = 0
    base = urlparse(base_url).netloc
    for a in a_tags:
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#"): continue
        rels = [r.lower() for r in (a.get("rel") or [])]
        if "nofollow" in rels: nofollow += 1
        full = urljoin(base_url, href)
        if urlparse(full).netloc == base: internal += 1
        else: external += 1

    # JSON-LD types
    schema_types: List[str] = []
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            import json
            data = json.loads(tag.text)
            def collect(obj):
                if isinstance(obj, dict):
                    t = obj.get("@type")
                    if t:
                        if isinstance(t, list): schema_types.extend([str(x) for x in t])
                        else: schema_types.append(str(t))
                    for v in obj.values(): collect(v)
                elif isinstance(obj, list):
                    for v in obj: collect(v)
            collect(data)
        except Exception:
            pass

    # Content stats
    text_content = soup.get_text(" ", strip=True)
    words = len(re.findall(r"[A-Za-z\u0900-\u097F']{2,}", text_content))
    reading_time_min = round(max(1, words) / 200, 1)  # ~200 wpm

    # Images & UX
    imgs = soup.find_all("img")
    images_sample, missing_alt, missing_dims, lazy_count = [], 0, 0, 0
    for im in imgs:
        src = (im.get("src") or "").strip()
        alt = (im.get("alt") or "").strip()
        w = im.get("width"); h = im.get("height")
        loading = (im.get("loading") or "").lower()
        if not alt: missing_alt += 1
        if not (w and h): missing_dims += 1
        if loading == "lazy": lazy_count += 1
        if src:
            images_sample.append({"src": urljoin(base_url, src), "alt": alt})
    lazy_ratio = (lazy_count / max(1, len(imgs)))

    # Mixed content (DOM scan)
    insecure = []
    if base_url.lower().startswith("https://"):
        selectors = [("img","src"), ("script","src"), ("iframe","src"),
                     ("link","href"), ("audio","src"), ("video","src"), ("source","src")]
        for tag, attr in selectors:
            for el in soup.find_all(tag):
                u = (el.get(attr) or "").strip()
                if u.startswith("http://"):
                    insecure.append(u)
    insecure = list(dict.fromkeys(insecure))[:15]

    # Issues
    issues = []
    if not title: issues.append("Missing <title>")
    if not meta_desc: issues.append("Missing meta description")
    h1c = len(headings.get("h1", []))
    if h1c == 0: issues.append("Missing <h1>")
    if h1c > 1: issues.append("Multiple <h1> tags")

    # Canonical sanity (absolute & same host)
    canonical_valid = None
    if canonical:
        try:
            can = urlparse(urljoin(base_url, canonical))
            canonical_valid = (can.scheme in ("http","https")) and (can.netloc != "")
        except Exception:
            canonical_valid = False

    return {
        "title": title,
        "meta": {
            "description": meta_desc, "robots": meta_robots, "viewport": meta_viewport,
            "canonical": canonical, "favicon": favicon, "amphtml": amphtml,
            "hreflang": hreflangs, "og_pairs": _pairs("og:", soup), "twitter_pairs": _pairs("twitter:", soup),
        },
        "headings": headings,
        "links": {"internal": internal, "external": external, "nofollow": nofollow},
        "schema_types": sorted(set(schema_types)),
        "keyword_density": _keyword_density(text_content, top_n=10),
        "content": {"word_count": words, "reading_time_min": reading_time_min},
        "media": {
            "images_count": len(imgs),
            "images_missing_alt": missing_alt,
            "images_missing_dimensions": missing_dims,
            "images_lazy_ratio": round(lazy_ratio, 2),
            "images_sample": images_sample[:8],
        },
        "mixed_content": {"insecure_count": len(insecure), "sample": insecure},
        "advanced": {"canonical_valid": canonical_valid, "title_length": len(title or ""), "meta_description_length": len(meta_desc or "")},
        "issues": issues,
    }