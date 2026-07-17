"""Polite, scope-limited BFS crawler for essex.ac.uk policy and
rules-of-assessment content.

Only follows links found inside the page's `richtext` content div (the CMS's
actual body-content container) so it never wanders into global nav, footer,
or unrelated cross-links/promo blocks — confirmed by inspecting the seed
pages' HTML directly."""

import hashlib
import io
import time
import urllib.robotparser
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

USER_AGENT = "RAGPoliciesBot/1.0 (personal research assistant; contact: kampouridis.michael@gmail.com)"
ALLOWED_DOMAIN = "www.essex.ac.uk"
ALLOWED_PAGE_PREFIXES = ("/governance-and-strategy/", "/student/rules-of-assessment/")
MEDIA_PREFIX = "/-/media/documents/"
REQUEST_DELAY_SECONDS = 0.7
REQUEST_TIMEOUT = 20


@dataclass
class CrawledItem:
    url: str
    content_type: str  # "pdf" or "html"
    title: str
    text: str
    content_hash: str
    links: list = field(default_factory=list)  # further page/doc links found (html only)


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl()


def _fix_relative(href: str) -> str:
    # essex.ac.uk sometimes emits document hrefs missing the leading slash,
    # e.g. "-/media/documents/..." instead of "/-/media/documents/...".
    if href.startswith("-/media/"):
        return "/" + href
    return href


class RobotsChecker:
    def __init__(self, base_url: str):
        self.rp = urllib.robotparser.RobotFileParser()
        self.rp.set_url(urljoin(base_url, "/robots.txt"))
        try:
            self.rp.read()
        except Exception:
            pass

    def allowed(self, url: str) -> bool:
        try:
            return self.rp.can_fetch(USER_AGENT, url)
        except Exception:
            return True


def _is_allowed_page(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc != ALLOWED_DOMAIN:
        return False
    return any(parsed.path.startswith(p) for p in ALLOWED_PAGE_PREFIXES)


def _is_media_doc(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc != ALLOWED_DOMAIN:
        return False
    return parsed.path.startswith(MEDIA_PREFIX)


def _extract_main_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    richtext_divs = soup.find_all("div", class_=lambda c: c and "richtext" in c.split())
    urls = []
    for div in richtext_divs:
        for a in div.find_all("a", href=True):
            href = _fix_relative(a["href"].strip())
            if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            urls.append(_normalize_url(urljoin(base_url, href)))
    return urls


def _extract_main_text(soup: BeautifulSoup) -> str:
    richtext_divs = soup.find_all("div", class_=lambda c: c and "richtext" in c.split())
    parts = [div.get_text(separator="\n", strip=True) for div in richtext_divs]
    return "\n\n".join(p for p in parts if p)


def _page_title(soup: BeautifulSoup, fallback: str) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return fallback


def _extract_pdf_text(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception:
        return ""
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(pages)


def fetch(url: str, session: requests.Session) -> CrawledItem | None:
    try:
        resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None

    content_type = resp.headers.get("Content-Type", "")
    content_hash = hashlib.sha256(resp.content).hexdigest()

    if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
        text = _extract_pdf_text(resp.content)
        title = url.rsplit("/", 1)[-1]
        return CrawledItem(url=url, content_type="pdf", title=title, text=text, content_hash=content_hash)

    if "html" not in content_type.lower():
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    title = _page_title(soup, url)
    text = _extract_main_text(soup)
    links = _extract_main_links(soup, url)
    return CrawledItem(url=url, content_type="html", title=title, text=text, content_hash=content_hash, links=links)


def crawl(seed_urls: list[str], on_item=None) -> None:
    """BFS crawl from seed_urls, restricted to ALLOWED_DOMAIN and
    ALLOWED_PAGE_PREFIXES/MEDIA_PREFIX. Calls on_item(CrawledItem) for every
    successfully fetched page or document."""
    session = requests.Session()
    robots = RobotsChecker(f"https://{ALLOWED_DOMAIN}/")

    visited: set[str] = set()
    queue: deque[str] = deque(_normalize_url(u) for u in seed_urls)

    while queue:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        if not robots.allowed(url):
            continue

        time.sleep(REQUEST_DELAY_SECONDS)
        item = fetch(url, session)
        if item is None:
            continue

        if on_item:
            on_item(item)

        if item.content_type == "html":
            for link in item.links:
                if link in visited:
                    continue
                if _is_media_doc(link) or _is_allowed_page(link):
                    queue.append(link)
