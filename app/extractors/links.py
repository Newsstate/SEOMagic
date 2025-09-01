# app/extractors/links.py
from __future__ import annotations
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from typing import Dict, Any, List

class _AParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base = base_url
        self._text_stack: List[str] = []
        self.rows: List[Dict[str, Any]] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        attrs_dict = dict(attrs)
        raw = (attrs_dict.get("href") or "").strip()
        if not raw or raw.startswith(("javascript:", "mailto:", "tel:")):
            return
        href = urljoin(self.base, raw)
        rel = (attrs_dict.get("rel") or "").lower()
        nf = ("nofollow" in rel)
        self.rows.append({"href": href, "text": "", "rel": rel, "nofollow": nf})

    def handle_data(self, data):
        if self.rows:
            # attach the latest chunk of text to the most recent <a>
            self.rows[-1]["text"] += data

def collect_links_from_html(html: str, base_url: str) -> Dict[str, Any]:
    p = urlparse(base_url)
    host = p.netloc.lower()

    parser = _AParser(base_url)
    try:
        parser.feed(html or "")
    except Exception:
        pass

    internal_set, external_set, nofollow_set = set(), set(), set()
    internal_list, external_list, nofollow_list = [], [], []

    def _norm(item):
        return {
            "href": item["href"],
            "text": (item.get("text") or "").strip()[:300],
            "rel": item.get("rel") or "",
        }

    for a in parser.rows:
        h = urlparse(a["href"]).netloc.lower()
        is_internal = (h == host) if h else True
        entry = _norm(a)

        if is_internal:
            if entry["href"] not in internal_set:
                internal_set.add(entry["href"])
                internal_list.append(entry)
        else:
            if entry["href"] not in external_set:
                external_set.add(entry["href"])
                external_list.append(entry)

        if a.get("nofollow"):
            if entry["href"] not in nofollow_set:
                nofollow_set.add(entry["href"])
                nofollow_list.append(entry)

    return {
        "internal": len(internal_list),
        "external": len(external_list),
        "nofollow": len(nofollow_list),
        "internal_list": internal_list[:1000],
        "external_list": external_list[:1000],
        "nofollow_list": nofollow_list[:1000],
    }
