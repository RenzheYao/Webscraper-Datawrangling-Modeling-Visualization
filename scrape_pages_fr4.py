"""
Mariage Freres tea scraper (phase 1) - scrape_pages_fr4.py

Important: the site's robots.txt disallows every URL containing
"conditionnement=" (the filter in https://www.mariagefreres.com/en/tea.html?conditionnement=48),
so this script does NOT request that URL. Instead it:

  1. reads the sitemap the site publishes for crawlers (sitemap_en.xml),
  2. opens the product pages listed there (product pages are allowed),
  3. keeps only real product pages, and
  4. extracts name, tea type, origin country, weight, price (+ packaging, SKU).

The packaging filter is applied locally: set PACKAGING_REGEX below
(e.g. r"canister|tin|jar|box") to keep only certain packagings.

Every request is checked against robots.txt before it is sent
(set RESPECT_ROBOTS = False only if you have permission to ignore it).

Downloads use curl_cffi (Chrome TLS fingerprint), as in the Kusmi version.

Install:
    pip install curl_cffi beautifulsoup4
"""
import os
import re
import gzip
import json
import time
import random
import html as htmllib
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
BASE = "https://www.mariagefreres.com"
TARGET_DIR = r"D:\Business\Freelancer\Webscraper+Datawrangling"
MARKET = "EN"

SITEMAP_URLS = [f"{BASE}/media/sitemap/sitemap_en.xml"]
CATEGORY_URL = f"{BASE}/en/tea.html"     # plain category page (no filter), used only as fallback
MAX_CATALOG_PAGES = 30

MAX_PRODUCTS = None           # valid products to collect; None = all (a full run takes hours)
MAX_PAGES_FETCHED = 200    # safety cap on how many URLs get opened (non-product pages count)
URL_REGEX = None           # optional: only open sitemap URLs matching this, e.g. r"tea|the"
PACKAGING_REGEX = None     # optional: keep only products whose packaging matches, e.g. r"canister|tin|jar"

RESPECT_ROBOTS = True
DELAY_MIN, DELAY_MAX = 3.0, 6.0
MAX_RETRIES = 5
DEBUG_MAX = 2              # how many raw product pages to save for inspection
IMPERSONATE = "chrome"

# product pages look like https://www.mariagefreres.com/en/<slug>.html (one path segment)
PRODUCT_URL_RE = re.compile(r"^https://www\.mariagefreres\.com/en/[^/?#]+\.html$")

COUNTRIES = {
    "China": "China", "Chine": "China", "Japan": "Japan", "Japon": "Japan",
    "India": "India", "Inde": "India", "Sri Lanka": "Sri Lanka", "Ceylon": "Sri Lanka",
    "Taiwan": "Taiwan", "Taïwan": "Taiwan", "Nepal": "Nepal", "Népal": "Nepal",
    "Kenya": "Kenya", "South Korea": "South Korea", "Corée du Sud": "South Korea",
    "Vietnam": "Vietnam", "Thailand": "Thailand", "Thaïlande": "Thailand",
    "South Africa": "South Africa", "Afrique du Sud": "South Africa",
    "Italy": "Italy", "Italie": "Italy", "France": "France",
    "Colombia": "Colombia", "Colombie": "Colombia", "Peru": "Peru", "Pérou": "Peru",
    "Brazil": "Brazil", "Brésil": "Brazil", "Laos": "Laos",
    "Indonesia": "Indonesia", "Indonésie": "Indonesia", "Turkey": "Turkey", "Turquie": "Turkey",
    "Malawi": "Malawi", "Rwanda": "Rwanda", "Tanzania": "Tanzania", "Tanzanie": "Tanzania",
    "Burundi": "Burundi", "Madagascar": "Madagascar", "Iran": "Iran",
    "Morocco": "Morocco", "Maroc": "Morocco", "Argentina": "Argentina", "Argentine": "Argentina",
    "Uganda": "Uganda", "Ouganda": "Uganda", "Cameroon": "Cameroon", "Cameroun": "Cameroon",
    "Ethiopia": "Ethiopia", "Éthiopie": "Ethiopia", "Bangladesh": "Bangladesh",
    "Myanmar": "Myanmar", "Birmanie": "Myanmar", "Cambodia": "Cambodia", "Cambodge": "Cambodia",
    "Mexico": "Mexico", "Mexique": "Mexico", "Georgia": "Georgia", "Géorgie": "Georgia",
}

PACK_WORDS = r"(?:canisters?|tins?|jars?|boxe?s?|bags?|pouch(?:es)?|sachets?|refills?|tubs?|caddy|caddies)"
TYPE_RE = re.compile(
    r"\b((?:green|black|white|blue|yellow|oolong|pu[- ]?erh|flavou?red|scented|smoked|herbal|fruit)\s+teas?"
    r"|rooibos|infusions?|matcha|oolong)\b", re.I)

HEADERS = {"Accept-Language": "en-GB,en;q=0.9,fr;q=0.8"}
# We do NOT set User-Agent / Sec-Ch-Ua: curl_cffi sends matching Chrome headers itself.

SESSION = None
ROBOTS = None


# ----------------------------------------------------------------------
# robots.txt (with wildcard support, which urllib.robotparser lacks)
# ----------------------------------------------------------------------
class Robots:
    """Minimal robots.txt matcher for the 'User-agent: *' group.
    Supports * and $ wildcards; the longest matching rule wins, Allow wins ties."""

    def __init__(self, text):
        self.rules = []  # (is_allow, compiled_regex, length)
        agents, in_rules = [], False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            field, value = (x.strip() for x in line.split(":", 1))
            field = field.lower()
            if field == "user-agent":
                if in_rules:
                    agents, in_rules = [], False
                agents.append(value.lower())
            elif field in ("allow", "disallow"):
                in_rules = True
                if "*" in agents and value:
                    self.rules.append((field == "allow", self._regex(value), len(value)))

    @staticmethod
    def _regex(pattern):
        p = re.escape(pattern).replace(r"\*", ".*")
        if p.endswith(r"\$"):
            p = p[:-2] + "$"
        return re.compile("^" + p)

    def allowed(self, url):
        parts = urlparse(url)
        target = parts.path + (("?" + parts.query) if parts.query else "")
        hits = [(length, is_allow) for is_allow, rx, length in self.rules if rx.match(target)]
        if not hits:
            return True
        return max(hits)[1]  # longest rule wins; True (allow) beats False on equal length


# ----------------------------------------------------------------------
# Extraction helpers
# ----------------------------------------------------------------------
def countries_in(text):
    low = (text or "").lower()
    hits = []
    for name in sorted(COUNTRIES, key=len, reverse=True):
        if re.search(r"(?<!\w)" + re.escape(name.lower()) + r"(?!\w)", low):
            canon = COUNTRIES[name]
            if canon not in hits:
                hits.append(canon)
    return hits


def find_name(soup):
    h1 = soup.select_one("h1.page-title, h1.product-name, h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(" ", strip=True)
    og = soup.select_one('meta[property="og:title"]')
    return og["content"].strip() if og and og.get("content") else None


def is_product_page(soup):
    if soup.select_one('meta[property="product:price:amount"]'):
        return True
    og = soup.select_one('meta[property="og:type"]')
    return bool(og and (og.get("content") or "").lower() == "product")


def clean_main(soup):
    """Copy of the page without menus, footer, scripts and 'related products' blocks."""
    s = BeautifulSoup(str(soup), "html.parser")
    for t in s(["script", "style", "noscript", "header", "footer", "nav"]):
        t.decompose()
    for t in s.select('[class*="related"], [class*="upsell"], [class*="crosssell"], [class*="recently"]'):
        t.decompose()
    main = s.select_one("#maincontent, main, .column.main") or s
    return main


def find_weight_and_packaging(text):
    """Weight of the product (e.g. '100g'), skipping brewing doses like '2.5 g per 20 cl'."""
    pat = re.compile(r"(\d+(?:[.,]\d+)?)\s*(kg|g)\b(?!\s*(?:per|par|pour|/))", re.I)
    cands = []
    for m in pat.finditer(text or ""):
        # look for a packaging word right after the weight, in the same sentence and before the next number
        tail = re.split(r"[.\d]", text[m.end(): m.end() + 40])[0]
        pk = re.search(r"\b" + PACK_WORDS + r"\b", tail, re.I)
        cands.append((m, pk))

    chosen = next(((m, pk) for m, pk in cands if pk), None)
    if chosen is None:
        for m, pk in cands:
            grams = float(m.group(1).replace(",", ".")) * (1000 if m.group(2).lower() == "kg" else 1)
            if grams >= 10:   # ignore tiny numbers, these are usually brewing doses
                chosen = (m, pk)
                break
    if chosen is None:
        return None, None
    m, pk = chosen
    return f"{m.group(1)}{m.group(2).lower()}", (pk.group(0).lower() if pk else None)


def find_tea_type(soup, name, main_text):
    crumbs = soup.select('[class*="breadcrumb"] li, [class*="breadcrumb"] a')
    for c in crumbs:
        m = TYPE_RE.search(c.get_text(" ", strip=True))
        if m:
            return m.group(1)
    for source in (name or "", main_text or ""):
        m = TYPE_RE.search(source)
        if m:
            return m.group(1)
    return None


def extract_product(soup, url, market):
    name = find_name(soup)
    main = clean_main(soup)
    main_text = " ".join(main.stripped_strings)

    # price: meta tag first, then Magento's data-price-amount, then visible text
    price = None
    meta = soup.select_one('meta[property="product:price:amount"]')
    if meta and meta.get("content"):
        try:
            price = float(meta["content"].replace(",", "."))
        except ValueError:
            pass
    if price is None:
        el = soup.select_one("[data-price-amount]")
        if el:
            try:
                price = float(el["data-price-amount"])
            except (ValueError, KeyError):
                pass
    if price is None:
        m = re.search(r"(?:€\s*(\d+[.,]\d{2}))|(?:(\d+[.,]\d{2})\s*€)", main_text)
        if m:
            price = float((m.group(1) or m.group(2)).replace(",", "."))

    weight, packaging = find_weight_and_packaging(main_text)
    tea_type = find_tea_type(soup, name, main_text)

    # origin country: description blocks first, then the product title, then the main area
    origin = None
    for el in main.select(".product.attribute.description, .product.attribute.overview, "
                          ".product-info-main, #description, .tab-content"):
        found = countries_in(el.get_text(" ", strip=True))
        if found:
            origin = ", ".join(found)
            break
    if not origin:
        found = countries_in(name) or countries_in(main_text)
        if found:
            origin = ", ".join(found)

    sku = None
    m = re.search(r"\bREF\.?\s*:?\s*([A-Z0-9]{3,})\b", main_text)
    if m:
        sku = m.group(1)

    return {
        "market": market,
        "url": url,
        "sku": sku,
        "name": name,
        "tea_type": tea_type,
        "origin_country": origin,
        "packaging": packaging,
        "weight": weight,
        "price_eur": price,
    }


# ----------------------------------------------------------------------
# Downloading
# ----------------------------------------------------------------------
def make_session():
    from curl_cffi import requests as cr
    return cr.Session(impersonate=IMPERSONATE)


def polite_sleep():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


def looks_like_block_page(resp):
    body = resp.text[:3000].lower()
    return (len(resp.text) < 5000 or "just a moment" in body
            or "captcha" in body or "access denied" in body)


def fetch(url, check_block=True):
    """GET with robots.txt check and retries. Backs off on 429/503."""
    if ROBOTS is not None and not ROBOTS.allowed(url):
        print(f"  - skipped (disallowed by robots.txt): {url}")
        return None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.get(url, headers=HEADERS, timeout=30)
        except Exception as e:
            wait = 5 * attempt
            print(f"  ! network error on {url}: {e!r} - retry {attempt}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            if check_block and looks_like_block_page(resp):
                print(f"  ! 200 but looks like a block/challenge page ({len(resp.text)} bytes): {url}")
                print(f"    first 300 chars: {resp.text[:300]!r}")
            return resp

        if resp.status_code in (429, 503):
            ra = resp.headers.get("Retry-After")
            wait = int(ra) if ra and ra.isdigit() else 15 * attempt
            print(f"  ! HTTP {resp.status_code} on {url} - retry {attempt}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)
            continue

        print(f"  ! HTTP {resp.status_code} on {url} - giving up on this URL")
        print(f"    server: {resp.headers.get('server')} | body: {resp.text[:300]!r}")
        return None

    print(f"  ! gave up on {url} after {MAX_RETRIES} attempts")
    return None


# ----------------------------------------------------------------------
# Discovering product URLs
# ----------------------------------------------------------------------
def read_sitemap(url, depth=0):
    resp = fetch(url, check_block=False)
    if resp is None or resp.status_code != 200:
        return []
    content = resp.content
    if content[:2] == b"\x1f\x8b":
        content = gzip.decompress(content)
    xml = content.decode("utf-8", "replace")
    locs = [htmllib.unescape(x) for x in re.findall(r"<loc>\s*(.*?)\s*</loc>", xml, re.S)]
    if "<sitemapindex" in xml and depth < 2:
        out = []
        for child in locs:
            polite_sleep()
            out += read_sitemap(child, depth + 1)
        return out
    return locs


def crawl_category():
    """Fallback if the sitemap gives nothing: the plain (unfiltered) category pages."""
    links, seen = [], set()
    for page in range(1, MAX_CATALOG_PAGES + 1):
        url = CATEGORY_URL if page == 1 else f"{CATEGORY_URL}?p={page}"
        print(f"[category] page {page}: {url}")
        resp = fetch(url)
        if resp is None or resp.status_code != 200:
            break
        soup = BeautifulSoup(resp.text, "html.parser")
        new = 0
        for a in soup.select('a[href*=".html"]'):
            u = urljoin(url, a.get("href", "")).split("#")[0].split("?")[0]
            if PRODUCT_URL_RE.match(u) and u != CATEGORY_URL and u not in seen:
                seen.add(u)
                links.append(u)
                new += 1
        print(f"[category] page {page}: {new} new links (total {len(links)})")
        if new == 0:
            break
        polite_sleep()
    return links


def discover_product_urls():
    urls = []
    for sm in SITEMAP_URLS:
        print(f"[sitemap] reading {sm}")
        urls += read_sitemap(sm)
    candidates = [u for u in dict.fromkeys(urls) if PRODUCT_URL_RE.match(u)]
    if URL_REGEX:
        candidates = [u for u in candidates if re.search(URL_REGEX, u, re.I)]
    print(f"[sitemap] {len(urls)} URLs in sitemap, {len(candidates)} look like product pages")
    if candidates:
        return candidates
    print("[sitemap] nothing usable - falling back to the plain category pages")
    return crawl_category()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run_phase_one():
    global SESSION, ROBOTS
    os.makedirs(TARGET_DIR, exist_ok=True)
    debug_dir = os.path.join(TARGET_DIR, "debug_mariage_html")
    SESSION = make_session()

    if RESPECT_ROBOTS:
        r = SESSION.get(f"{BASE}/robots.txt", headers=HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"Could not load robots.txt (HTTP {r.status_code}); stopping. "
                  f"Set RESPECT_ROBOTS = False only if you have permission.")
            return
        ROBOTS = Robots(r.text)
        print(f"[robots] loaded {len(ROBOTS.rules)} rules for 'User-agent: *'")

    candidates = discover_product_urls()
    results, fetched = [], 0
    output_file = os.path.join(TARGET_DIR, "mariage_raw_results.json")

    try:
        for url in candidates:
            if MAX_PRODUCTS is not None and len(results) >= MAX_PRODUCTS:
                break
            if fetched >= MAX_PAGES_FETCHED:
                print(f"[stop] reached MAX_PAGES_FETCHED = {MAX_PAGES_FETCHED}")
                break

            polite_sleep()
            resp = fetch(url)
            fetched += 1
            if resp is None or resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            if not is_product_page(soup):
                print(f"  - not a product page, skipped: {url}")
                continue

            data = extract_product(soup, url, MARKET)
            if PACKAGING_REGEX and not re.search(PACKAGING_REGEX, data["packaging"] or "", re.I):
                print(f"  - packaging {data['packaging']!r} does not match filter, skipped: {url}")
                continue

            if len(results) < DEBUG_MAX:
                os.makedirs(debug_dir, exist_ok=True)
                with open(os.path.join(debug_dir, f"product_{len(results) + 1}.html"), "w", encoding="utf-8") as f:
                    f.write(resp.text)

            results.append(data)
            print(f"#{len(results)} {data['name']} | Type: {data['tea_type']} | "
                  f"Origin: {data['origin_country']} | Weight: {data['weight']} | "
                  f"Packaging: {data['packaging']} | Price: {data['price_eur']} EUR")
    finally:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[Phase 1 Complete] {len(results)} products scraped "
              f"({fetched} pages opened) -> {output_file}")


if __name__ == "__main__":
    run_phase_one()