"""
Kusmi Tea scraper (phase 1).

Why this version does not use Scrapy for downloading:
Kusmi answers Scrapy with HTTP 429 even on the first request, while a normal
browser works. The usual reason is the TLS fingerprint: Scrapy/Twisted does not
"look like" Chrome at the connection level, whatever headers you send.
curl_cffi can impersonate Chrome's TLS fingerprint, so we use it to download
the pages and keep BeautifulSoup for extraction.

Install:
    pip install curl_cffi beautifulsoup4
"""
import os
import json
import re
import time
import random
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests as cr

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
TARGET_DIR = r"D:\Business\Freelancer\Webscraper+Datawrangling"
CATALOG_URL = "https://www.kusmitea.com/collections/tous-les-thes"
MARKET = "FR"

MAX_PRODUCTS = None        # set to None to scrape the whole catalog
MAX_CATALOG_PAGES = 30  # safety limit for pagination
DELAY_MIN, DELAY_MAX = 3.0, 6.0   # random pause between requests (seconds)
MAX_RETRIES = 5
DEBUG_MAX = 2           # how many raw product pages to save for inspection
IMPERSONATE = "chrome"  # curl_cffi picks the latest Chrome it supports

COUNTRIES = {
    "Chine": "Chine", "Japon": "Japon", "Inde": "Inde", "Sri Lanka": "Sri Lanka",
    "Taïwan": "Taïwan", "Taiwan": "Taïwan", "Népal": "Népal", "Nepal": "Népal",
    "Kenya": "Kenya", "Corée du Sud": "Corée du Sud", "Vietnam": "Vietnam",
    "Thaïlande": "Thaïlande", "Afrique du Sud": "Afrique du Sud", "Italie": "Italie",
    "France": "France", "Colombie": "Colombie", "Pérou": "Pérou", "Brésil": "Brésil"
}

# ----------------------------------------------------------------------
# Extraction helpers (unchanged logic)
# ----------------------------------------------------------------------
def parse_weight(text):
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*(kg|g)\b', text or '', re.I)
    return f"{m.group(1)}{m.group(2).lower()}" if m else None


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
    h1 = soup.select_one("h1.product-title, h1.product__title, h1.h2, h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(" ", strip=True)
    og = soup.select_one('meta[property="og:title"]')
    return og["content"].strip() if og and og.get("content") else None


def extract_product(soup, url, market):
    name = find_name(soup)
    page_text = " ".join(soup.stripped_strings)

    price = None
    weight = None

    # 1. Shopify product JSON embedded in script tags, "Boîte" variant
    for script in soup.find_all("script", type="application/json"):
        txt = script.string or script.get_text() or ""
        if "variants" in txt and "price" in txt:
            try:
                data = json.loads(txt)
                variants = []
                if isinstance(data, dict):
                    if "variants" in data:
                        variants = data["variants"]
                    elif "product" in data and isinstance(data["product"], dict):
                        variants = data["product"].get("variants", [])
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "variants" in item:
                            variants = item["variants"]

                for v in variants:
                    title = str(v.get("title", "") or v.get("name", "")).lower()
                    if "boîte" in title or "boite" in title or "tin" in title:
                        raw_price = v.get("price")
                        if raw_price:
                            price = float(raw_price) / 100.0 if float(raw_price) > 500 else float(raw_price)
                        weight = parse_weight(title)
                        break
            except Exception:
                pass

    # 2. DOM fallback for price
    if price is None:
        price_el = soup.select_one('.price-item--regular, .product__price span, .current-price, span.price, meta[property="product:price:amount"]')
        if price_el:
            p_text = price_el.get("content") or price_el.get_text(strip=True)
            m = re.search(r'(\d+[.,]\d{2})', p_text.replace('\xa0', '').replace(',', '.'))
            if m:
                try:
                    price = float(m.group(1))
                except ValueError:
                    pass

    # 3. DOM fallback for weight
    if weight is None:
        w_match = re.search(r'Bo[îi]te\s*(\d+\s*g)\b', page_text, re.I)
        if w_match:
            weight = w_match.group(1).replace(" ", "")
        else:
            w_fallback = re.search(r'(\d+)\s*g\b', page_text, re.I)
            weight = f"{w_fallback.group(1)}g" if w_fallback else None

    # 4. Tea type / category
    tea_type = None
    category_match = re.search(r'\b(th[ée]\s+(?:noir|vert|blanc|oolong)[^|\n,.]*|rooibos|infusion|tisane|matcha)\b', page_text, re.I)
    if category_match:
        tea_type = category_match.group(1).strip()

    # 5. Country of origin
    origin_country = None
    for el in soup.select('.product-description, .product__description, .tab-content, #description'):
        txt = el.get_text(" ", strip=True)
        found = countries_in(txt)
        if found:
            origin_country = ", ".join(found)
            break

    if not origin_country:
        found_countries = countries_in(page_text)
        if found_countries:
            origin_country = ", ".join(found_countries)

    return {
        "market": market,
        "url": url,
        "name": name,
        "tea_type": tea_type,
        "origin_country": origin_country,
        "weight": weight,
        "price_eur": price,
    }


# ----------------------------------------------------------------------
# Downloading
# ----------------------------------------------------------------------
session = cr.Session(impersonate=IMPERSONATE)
HEADERS = {"Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7"}
# NOTE: we do NOT set User-Agent / Sec-Ch-Ua ourselves. curl_cffi sends the
# matching Chrome headers automatically; overriding them can break the match.


def polite_sleep():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


def looks_like_block_page(resp):
    body = resp.text[:3000].lower()
    return (
        len(resp.text) < 5000
        or "just a moment" in body
        or "captcha" in body
        or "access denied" in body
    )


def fetch(url):
    """GET with retries. Backs off on 429/503 and respects Retry-After."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
        except Exception as e:
            wait = 5 * attempt
            print(f"  ! network error on {url}: {e!r} - retry {attempt}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            if looks_like_block_page(resp):
                print(f"  ! 200 but the page looks like a block/challenge page ({len(resp.text)} bytes): {url}")
                print(f"    first 300 chars: {resp.text[:300]!r}")
            return resp

        if resp.status_code in (429, 503):
            retry_after = resp.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else 15 * attempt
            print(f"  ! HTTP {resp.status_code} on {url} - retry {attempt}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)
            continue

        print(f"  ! HTTP {resp.status_code} on {url} - giving up on this URL")
        print(f"    server: {resp.headers.get('server')} | body: {resp.text[:300]!r}")
        return None

    print(f"  ! gave up on {url} after {MAX_RETRIES} attempts")
    return None


def clean_product_url(base, href):
    abs_url = urljoin(base, href).split('#')[0].split('?')[0]
    path = urlparse(abs_url).path
    return abs_url if '/products/' in path else None


def collect_product_links():
    """Walk the paginated collection (?page=N) and return unique product URLs."""
    links = []
    seen = set()

    for page in range(1, MAX_CATALOG_PAGES + 1):
        page_url = CATALOG_URL if page == 1 else f"{CATALOG_URL}?page={page}"
        print(f"[catalog] page {page}: {page_url}")
        resp = fetch(page_url)
        if resp is None or resp.status_code != 200:
            print("[catalog] could not load this page - stopping pagination")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        new_on_page = 0
        for a in soup.select('a[href*="/products/"]'):
            url = clean_product_url(page_url, a.get("href", ""))
            if url and url not in seen:
                seen.add(url)
                links.append(url)
                new_on_page += 1

        print(f"[catalog] page {page}: {new_on_page} new product links (total {len(links)})")

        if new_on_page == 0:
            break
        if MAX_PRODUCTS is not None and len(links) >= MAX_PRODUCTS:
            break
        polite_sleep()

    return links[:MAX_PRODUCTS] if MAX_PRODUCTS is not None else links


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run_phase_one():
    os.makedirs(TARGET_DIR, exist_ok=True)
    debug_dir = os.path.join(TARGET_DIR, "debug_kusmi_html")

    product_links = collect_product_links()
    print(f"\n[phase 1] {len(product_links)} product URLs to scrape\n")

    results = []
    for i, url in enumerate(product_links, start=1):
        polite_sleep()
        resp = fetch(url)
        if resp is None or resp.status_code != 200:
            continue

        if i <= DEBUG_MAX:
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, f"product_{i}.html"), "w", encoding="utf-8") as f:
                f.write(resp.text)

        soup = BeautifulSoup(resp.text, "html.parser")
        data = extract_product(soup, url, MARKET)
        results.append(data)
        print(
            f"#{len(results)} {data['name']} | Type: {data['tea_type']} | "
            f"Origin: {data['origin_country']} | Weight: {data['weight']} | "
            f"Price: {data['price_eur']} EUR"
        )

    output_file = os.path.join(TARGET_DIR, "kusmi_raw_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Phase 1 Complete] {len(results)} pages scraped -> {output_file}")


if __name__ == "__main__":
    run_phase_one()