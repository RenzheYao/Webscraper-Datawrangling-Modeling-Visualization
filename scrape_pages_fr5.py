"""
Betjeman & Barton tea scraper (phase 1) - scrape_pages_fr5.py   (boites only, no sachets)

Important: the site's robots.txt disallows the URL you gave (collection URLs with
"filter." parameters and with %2B / '+'), so this script does NOT request it.
It uses the Shopify JSON feed of the plain collection instead, which robots.txt allows:

    https://betjemanandbarton.com/collections/les-boites-et-sachets-de-the/products.json

That feed already contains title, tags, price, variants and the product description,
so no HTML parsing of product pages is needed. Then, locally:

  * keeps only products tagged b2c or b2c+b2b   (this mirrors your b2c_b2b_filter URL parameters)
  * keeps only BOITES: title/handle/variant contains "boite" and NOT "sachet"
    ("Paquet 250g" and all sachet products are dropped)
  * extracts name, tea type, origin country, net weight, price (+ sku, packaging weight)

Net weight comes from the description ("Poids net : 125g"); if that is missing it falls back
to the image file name (e.g. "SD-boite_100g_...jpg"). The Shopify "grams" field is the SHIPPING
weight of the parcel (about 228 g for a boite), so it is stored separately as shipping_weight_g.

Every request is checked against robots.txt first (RESPECT_ROBOTS = True).
Downloads use curl_cffi (Chrome TLS fingerprint).

Install:
    pip install curl_cffi beautifulsoup4
"""
import os
import re
import json
import time
import random
import unicodedata
from urllib.parse import urlparse

from bs4 import BeautifulSoup

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
BASE = "https://betjemanandbarton.com"
TARGET_DIR = r"D:\Business\Freelancer\Webscraper+Datawrangling"
MARKET = "FR"

COLLECTION_HANDLES = ["les-boites-et-sachets-de-the"]   # add more collection handles if needed
PAGE_SIZE = 250            # Shopify maximum
MAX_PAGES = 40             # safety limit for pagination
MAX_PRODUCTS = None        # stop after N boite rows; None = all (only a few requests are needed)

REQUIRE_TAGS_ANY = {"b2c", "b2c+b2b"}   # mirrors your filter; set to None to disable
BOITE_RE = re.compile(r"\bboite\b")      # matched on accent-free lowercase text
EXCLUDE_RE = re.compile(r"sachet")

RESPECT_ROBOTS = True
DELAY_MIN, DELAY_MAX = 2.0, 4.0
MAX_RETRIES = 5
IMPERSONATE = "chrome"

# Country names are matched on accent-free lowercase text; values are the output names.
COUNTRIES = {
    "chine": "Chine", "japon": "Japon", "inde": "Inde", "sri lanka": "Sri Lanka",
    "ceylan": "Sri Lanka", "ceylon": "Sri Lanka", "taiwan": "Taïwan", "nepal": "Népal",
    "kenya": "Kenya", "coree du sud": "Corée du Sud", "vietnam": "Vietnam",
    "thailande": "Thaïlande", "afrique du sud": "Afrique du Sud", "italie": "Italie",
    "france": "France", "colombie": "Colombie", "perou": "Pérou", "bresil": "Brésil",
    "laos": "Laos", "indonesie": "Indonésie", "turquie": "Turquie", "malawi": "Malawi",
    "rwanda": "Rwanda", "tanzanie": "Tanzanie", "burundi": "Burundi",
    "madagascar": "Madagascar", "iran": "Iran", "maroc": "Maroc", "argentine": "Argentine",
    "ouganda": "Ouganda", "cameroun": "Cameroun", "ethiopie": "Éthiopie",
    "bangladesh": "Bangladesh", "birmanie": "Birmanie", "cambodge": "Cambodge",
    "mexique": "Mexique", "georgie": "Géorgie", "egypte": "Égypte",
    # well-known tea regions -> country
    "darjeeling": "Inde", "assam": "Inde", "nilgiri": "Inde",
    "yunnan": "Chine", "keemun": "Chine", "fujian": "Chine",
    "uva": "Sri Lanka", "dimbula": "Sri Lanka", "nuwara eliya": "Sri Lanka",
}
# "fabriqué en France", "conditionné en France" ... is not an origin of the tea
NOT_ORIGIN_CONTEXT = re.compile(r"(fabriqu|conditionn|emball|assembl|produit en|prepare en|made in|packag)")
COUNTRY_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(n) for n in sorted(COUNTRIES, key=len, reverse=True)) + r")(?!\w)")

TYPE_RE = re.compile(r"\b(the (?:noir|vert|blanc|bleu|jaune|oolong|fume)|tisane|rooibos|infusion|matcha)\b")
WEIGHT_UNIT = r"(\d+(?:[.,]\d+)?)\s*(kg|g)(?![a-z])"

HEADERS = {"Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8", "Accept": "application/json"}
# No User-Agent / Sec-Ch-Ua override: curl_cffi sends matching Chrome headers itself.

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
def norm(s):
    """lowercase and strip accents"""
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def countries_in(text):
    low = norm(text)
    hits = []
    for m in COUNTRY_RE.finditer(low):
        if NOT_ORIGIN_CONTEXT.search(low[max(0, m.start() - 30): m.start()]):
            continue
        canon = COUNTRIES[m.group(1)]
        if canon not in hits:
            hits.append(canon)
    return hits


def find_tea_type(title, handle, tags):
    for source in (title, (handle or "").replace("-", " "), " ".join(tags)):
        m = TYPE_RE.search(norm(source))
        if m:
            t = m.group(1)
            return {"the fume": "thé fumé"}.get(t, re.sub(r"^the\b", "thé", t))
    return None


def find_net_weight(body_text, image_srcs, title):
    """Net weight of the tea. Returns (weight, source)."""
    m = re.search(r"poids\s+net\s*:?\s*" + WEIGHT_UNIT, norm(body_text))
    if m:
        return f"{m.group(1)}{m.group(2)}", "description"
    for src in image_srcs:
        m = re.search(r"boite[_\- ]?" + WEIGHT_UNIT, norm(src))
        if m:
            return f"{m.group(1)}{m.group(2)}", "image_filename"
    m = re.search(WEIGHT_UNIT, norm(title))
    if m:
        return f"{m.group(1)}{m.group(2)}", "title"
    return None, None


def product_rows(p):
    """Turn one Shopify product object into 0..n output rows (boite variants only)."""
    tags = p.get("tags") or []
    if isinstance(tags, str):
        tags = [t for t in tags.split(",")]
    tags = [t.strip() for t in tags if t.strip()]
    if REQUIRE_TAGS_ANY and not ({t.lower() for t in tags} & REQUIRE_TAGS_ANY):
        return []

    title = p.get("title") or ""
    handle = p.get("handle") or ""
    body_text = BeautifulSoup(p.get("body_html") or "", "html.parser").get_text(" ", strip=True)
    image_srcs = [i.get("src", "") for i in (p.get("images") or [])]

    weight, weight_source = find_net_weight(body_text, image_srcs, title)
    origin = countries_in(body_text)
    tea_type = find_tea_type(title, handle, tags)

    rows = []
    for v in p.get("variants") or []:
        vtitle = v.get("title") or ""
        blob = f"{norm(title)} {norm(handle)} {norm(vtitle)}"
        if EXCLUDE_RE.search(blob) or not BOITE_RE.search(blob):
            continue
        try:
            price = float(v["price"]) if v.get("price") not in (None, "") else None
        except ValueError:
            price = None
        rows.append({
            "market": MARKET,
            "url": f"{BASE}/products/{handle}",
            "handle": handle,
            "sku": v.get("sku"),
            "name": title,
            "variant": None if vtitle == "Default Title" else vtitle,
            "tea_type": tea_type,
            "origin_country": ", ".join(origin) if origin else None,
            "weight": weight,
            "weight_source": weight_source,
            "shipping_weight_g": v.get("grams"),
            "price_eur": price,
            "available": v.get("available"),
            "tags": tags,
        })
    return rows


# ----------------------------------------------------------------------
# Downloading
# ----------------------------------------------------------------------
def make_session():
    from curl_cffi import requests as cr
    return cr.Session(impersonate=IMPERSONATE)


def polite_sleep():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


def fetch(url):
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


def collect_products(debug_dir):
    """Walk /collections/<handle>/products.json?limit=250&page=N and return unique products."""
    seen = {}
    for handle in COLLECTION_HANDLES:
        for page in range(1, MAX_PAGES + 1):
            url = f"{BASE}/collections/{handle}/products.json?limit={PAGE_SIZE}&page={page}"
            print(f"[collection] {handle} page {page}")
            resp = fetch(url)
            if resp is None:
                break
            try:
                products = json.loads(resp.text).get("products", [])
            except ValueError:
                print(f"  ! response is not JSON. First 300 chars: {resp.text[:300]!r}")
                break
            if not products:
                break

            if page == 1:
                os.makedirs(debug_dir, exist_ok=True)
                with open(os.path.join(debug_dir, f"{handle}_page1.json"), "w", encoding="utf-8") as f:
                    f.write(resp.text)

            new = 0
            for p in products:
                if p.get("id") not in seen:
                    seen[p.get("id")] = p
                    new += 1
            print(f"[collection] {handle} page {page}: {len(products)} products ({new} new, total {len(seen)})")
            if len(products) < PAGE_SIZE or new == 0:
                break
            polite_sleep()
    return list(seen.values())


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run_phase_one():
    global SESSION, ROBOTS
    os.makedirs(TARGET_DIR, exist_ok=True)
    debug_dir = os.path.join(TARGET_DIR, "debug_betjeman_json")
    SESSION = make_session()

    if RESPECT_ROBOTS:
        r = SESSION.get(f"{BASE}/robots.txt", headers=HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"Could not load robots.txt (HTTP {r.status_code}); stopping. "
                  f"Set RESPECT_ROBOTS = False only if you have permission.")
            return
        ROBOTS = Robots(r.text)
        print(f"[robots] loaded {len(ROBOTS.rules)} rules for 'User-agent: *'")

    products = collect_products(debug_dir)
    print(f"\n[phase 1] {len(products)} products in the collection, filtering for boites...\n")

    results = []
    for p in products:
        for row in product_rows(p):
            results.append(row)
            print(f"#{len(results)} {row['name']} | Type: {row['tea_type']} | Origin: {row['origin_country']} | "
                  f"Weight: {row['weight']} ({row['weight_source']}) | Price: {row['price_eur']} EUR")
        if MAX_PRODUCTS is not None and len(results) >= MAX_PRODUCTS:
            results = results[:MAX_PRODUCTS]
            break

    output_file = os.path.join(TARGET_DIR, "betjeman_raw_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Phase 1 Complete] {len(results)} boite rows from {len(products)} products -> {output_file}")


if __name__ == "__main__":
    run_phase_one()