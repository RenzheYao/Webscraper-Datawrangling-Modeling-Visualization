import os
import json
import re
import scrapy
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup

TARGET_DIR = r"D:\Business\Freelancer\Webscraper+Datawrangling"
RAW_PAGES = []
DEBUG_SAVED = 0          # how many raw product pages were saved for inspection
DEBUG_MAX = 2


# ----------------------------------------------------------------------
# Extraction helpers (no network, so they can be tested on a saved page)
# ----------------------------------------------------------------------
def parse_weight(text):
    """'Boîte de 30g' -> '30g'. Returns None if no weight is found."""
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*(kg|g)\b', text or '', re.I)
    return f"{m.group(1)}{m.group(2).lower()}" if m else None


def is_tin(label, image=""):
    """True for the 'Boîte (en métal)' option, False for sachet / vrac / pochette."""
    l = (label or "").lower()
    img = (image or "").lower()
    if any(w in l for w in ("sachet", "vrac", "pochette")):
        return False
    return any(w in l for w in ("boîte", "boite", "métal", "metal")) or "metal" in img


def _walk(obj):
    """Find the Magento config dict that has both 'attributes' and 'optionPrices'."""
    if isinstance(obj, dict):
        if "optionPrices" in obj and "attributes" in obj:
            return obj
        for v in obj.values():
            r = _walk(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _walk(v)
            if r:
                return r
    return None


def find_magento_config(soup):
    decoder = json.JSONDecoder()
    for script in soup.find_all("script"):
        txt = script.string or script.get_text() or ""
        if "optionPrices" not in txt:
            continue
        try:
            cfg = _walk(json.loads(txt))
            if cfg:
                return cfg
        except Exception:
            pass
        # Inline JS: look backwards for the '{' that starts the config object
        idx = txt.find('"optionPrices"')
        starts = [m.start() for m in re.finditer(r"\{", txt[:idx])][-300:]
        for pos in reversed(starts):
            try:
                obj, _ = decoder.raw_decode(txt, pos)
            except Exception:
                continue
            cfg = _walk(obj)
            if cfg:
                return cfg
    return None


def variants_from_config(cfg):
    out = []
    prices = cfg.get("optionPrices", {}) or {}
    for attr in (cfg.get("attributes", {}) or {}).values():
        for opt in attr.get("options", []):
            price = None
            for pid in opt.get("products", []):
                p = prices.get(str(pid))
                if p:
                    price = (p.get("finalPrice") or {}).get("amount")
                    break
            out.append({"label": opt.get("label", ""), "price": price, "image": ""})
    return out


def variants_from_dom(soup):
    out = []
    for el in soup.select(".swatch-option, [data-option-label], [option-label]"):
        img = el.find("img")
        image = (img.get("src") or "") if img else ""
        label = (el.get("data-option-label") or el.get("option-label")
                 or el.get("aria-label") or el.get("title")
                 or el.get_text(" ", strip=True) or (img.get("alt") if img else "") or "")
        if label or image:
            out.append({"label": label.strip(), "price": None, "image": image})
    # Plain-text fallback: "Boîte de 30g"
    if not out:
        text = soup.get_text(" ", strip=True)
        for m in re.finditer(r"(Bo[iî]te[^|]{0,40}?\d+\s*g)\b", text, re.I):
            out.append({"label": m.group(1).strip(), "price": None, "image": ""})
    return out


def meta_price(soup):
    el = soup.select_one('meta[property="product:price:amount"], meta[name="product:price:amount"]')
    try:
        return float(el["content"].replace(",", ".")) if el else None
    except Exception:
        return None


def find_name(soup):
    h1 = soup.select_one("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(" ", strip=True)
    og = soup.select_one('meta[property="og:title"]')
    return og["content"].strip() if og and og.get("content") else None


def find_origin(soup):
    """Value next to the 'Origine :' label in the 'Spécificités' block."""
    texts = list(soup.stripped_strings)
    for i, t in enumerate(texts):
        m = re.match(r"^Origine\s*:?\s*(.*)$", t, re.I)
        if m:
            if m.group(1).strip():
                return m.group(1).strip()
            if i + 1 < len(texts):
                return texts[i + 1].strip()
    return None


COUNTRIES = {
    "Chine": "Chine", "Japon": "Japon", "Inde": "Inde", "Sri Lanka": "Sri Lanka",
    "Taïwan": "Taïwan", "Taiwan": "Taïwan", "Népal": "Népal", "Nepal": "Népal",
    "Kenya": "Kenya", "Corée du Sud": "Corée du Sud", "Corée": "Corée du Sud",
    "Vietnam": "Vietnam", "Viêt Nam": "Vietnam", "Thaïlande": "Thaïlande",
    "Afrique du Sud": "Afrique du Sud", "Géorgie": "Géorgie", "Turquie": "Turquie",
    "Indonésie": "Indonésie", "Birmanie": "Birmanie", "Myanmar": "Birmanie",
    "Madagascar": "Madagascar", "Brésil": "Brésil", "Argentine": "Argentine",
    "Maroc": "Maroc", "Égypte": "Égypte", "Egypte": "Égypte", "Iran": "Iran",
    "Malawi": "Malawi", "Rwanda": "Rwanda", "Tanzanie": "Tanzanie", "Ouganda": "Ouganda",
    "Laos": "Laos", "Cambodge": "Cambodge", "Italie": "Italie", "France": "France",
    "Colombie": "Colombie", "Pérou": "Pérou", "Sénégal": "Sénégal", "Cameroun": "Cameroun",
    "Burundi": "Burundi", "Mozambique": "Mozambique", "Zimbabwe": "Zimbabwe",
    "Paraguay": "Paraguay", "Malaisie": "Malaisie", "Bhoutan": "Bhoutan",
    "Bangladesh": "Bangladesh", "Russie": "Russie", "Azerbaïdjan": "Azerbaïdjan",
    "Afrique": "Afrique",   # the site writes e.g. "Thé noir (Inde, Afrique)"
}
# Region -> country, used only when no country is written on the page
REGION_TO_COUNTRY = {
    "yunnan": "Chine", "fujian": "Chine", "anhui": "Chine", "zhejiang": "Chine",
    "hunan": "Chine", "sichuan": "Chine", "jiangxi": "Chine", "guangdong": "Chine",
    "darjeeling": "Inde", "assam": "Inde", "nilgiri": "Inde", "kerala": "Inde",
    "ceylan": "Sri Lanka", "uva": "Sri Lanka", "dimbula": "Sri Lanka", "nuwara eliya": "Sri Lanka",
    "uji": "Japon", "shizuoka": "Japon", "kagoshima": "Japon", "kyoto": "Japon",
    "alishan": "Taïwan",
}


def countries_in(text):
    """All known countries named in a text, canonical French spelling, in order."""
    low = (text or "").lower()
    hits = []
    for name in sorted(COUNTRIES, key=len, reverse=True):   # longest first: 'Corée du Sud' before 'Corée'
        m = re.search(r"(?<!\w)" + re.escape(name.lower()) + r"(?!\w)", low)
        if m:
            canon = COUNTRIES[name]
            if canon not in [h[1] for h in hits]:
                hits.append((m.start(), canon))
    return [c for _, c in sorted(hits)]


def region_country(text):
    low = (text or "").lower()
    for region, country in REGION_TO_COUNTRY.items():
        if re.search(r"(?<!\w)" + re.escape(region) + r"(?!\w)", low):
            return country
    return None


def find_subtitle(soup):
    """Line right under the h1, e.g. 'Earl Grey - Thé noir du Yunnan et bergamote'."""
    h1 = soup.select_one("h1")
    if not h1:
        return None
    name = h1.get_text(" ", strip=True)
    cands = []
    for s in h1.find_all_next(string=True, limit=60):
        if h1 in s.parents or s.parent.name in ("script", "style", "noscript"):
            continue
        t = " ".join(str(s).split())
        if not t or t == name:
            continue
        cands.append(t)
        if len(cands) == 8:
            break
    for t in cands:
        if re.search(r"\bth[ée]s?\b|infusion|rooibos|mat[ée]\b|tisane|matcha|oolong|pu-?erh", t, re.I):
            return t
    return cands[0] if cands else None


def find_origin_country(soup, subtitle):
    """Country of origin: 'Origine :' label, else tea ingredient '(Chine)', else region in the subtitle."""
    detail = find_origin(soup)
    if detail:
        found = countries_in(detail)
        if found:
            return ", ".join(found), detail, "origine"
        rc = region_country(detail)
        if rc:
            return rc, detail, "origine (region)"

    for ing in ingredient_texts(soup):
        found = []
        for seg in split_top_level(ing):
            if TEA_BASE.match(seg):
                # every bracket of the tea ingredient: "Thé vert * (95%) (Japon)", "Thé noir* (Inde, Afrique) (97,3%)"
                for group in re.findall(r"\(([^)]*)\)", seg):
                    for c in countries_in(group):
                        if c not in found:
                            found.append(c)
        if found:
            return ", ".join(found), detail, "ingredients"

    rc = region_country(subtitle)
    return (rc, detail, "subtitle (region)") if rc else (None, detail, None)


TEA_BASE = re.compile(r"\s*(th[ée]|rooibos|mat[ée]|infusion)", re.I)

CATEGORY_PATTERNS = [
    ("noir", r"th[ée]s?\s+noirs?"),
    ("vert", r"th[ée]s?\s+verts?|matcha"),
    ("blanc", r"th[ée]s?\s+blancs?"),
    ("oolong", r"oolong|th[ée]s?\s+bleus?"),
    ("pu-erh", r"pu-?erh|th[ée]s?\s+sombres?"),
    ("jaune", r"th[ée]s?\s+jaunes?"),
    ("rooibos", r"rooibos"),
    ("maté", r"\bmat[ée]\b"),
    ("infusion", r"infusion|tisane"),
]


def split_top_level(s):
    """Split an ingredient list on commas that are not inside brackets."""
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch in ",;" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def ingredient_texts(soup):
    texts = list(soup.stripped_strings)
    for i, t in enumerate(texts):
        m = re.match(r"^Ingr[ée]dients?\s*:?\s*(.*)$", t, re.I)
        if m:
            yield m.group(1) or " ".join(texts[i + 1:i + 3])


def find_label_value(soup, label_regex):
    """Value next to a 'Label :' line in the Spécificités block."""
    texts = list(soup.stripped_strings)
    for i, t in enumerate(texts):
        m = re.match(r"^(?:%s)\s*:?\s*(.*)$" % label_regex, t, re.I)
        if m:
            if m.group(1).strip():
                return m.group(1).strip()
            if i + 1 < len(texts):
                return texts[i + 1].strip()
    return None


def category_in(text):
    """Earliest tea category named in a text: noir / vert / blanc / oolong / ..."""
    best = None
    for cat, pattern in CATEGORY_PATTERNS:
        m = re.search(pattern, text or "", re.I)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), cat)
    return best[1] if best else None


def find_tea_category(soup, subtitle, name):
    """noir / vert / ... from 'Couleur :', else subtitle, else the tea ingredient, else the name."""
    couleur = find_label_value(soup, r"Couleur")
    cat = category_in(couleur)
    if cat:
        return cat, "couleur"
    cat = category_in(subtitle)
    if cat:
        return cat, "subtitle"
    for ing in ingredient_texts(soup):
        for seg in split_top_level(ing):
            if TEA_BASE.match(seg):
                cat = category_in(seg)
                if cat:
                    return cat, "ingredients"
    cat = category_in(name)
    return (cat, "name") if cat else (None, None)


def extract_product(soup, url, market):
    cfg = find_magento_config(soup)
    variants = variants_from_config(cfg) if cfg else variants_from_dom(soup)
    source = "magento_config" if cfg else "dom"

    tin = next((v for v in variants if is_tin(v["label"], v["image"]) and v["price"] is not None), None) \
        or next((v for v in variants if is_tin(v["label"], v["image"])), None)

    price = tin["price"] if tin else None
    price_note = source
    # Single-option page: the displayed price IS the tin price
    if tin and price is None and len([v for v in variants if v["label"] or v["image"]]) == 1:
        price = meta_price(soup)
        price_note = "meta_price (single option)"

    subtitle = find_subtitle(soup)
    name = find_name(soup)
    country, origin_detail, origin_source = find_origin_country(soup, subtitle)
    category, category_source = find_tea_category(soup, subtitle, name)

    # Fallbacks when the option label has no weight / the config has no price
    page_text = " ".join(soup.stripped_strings)
    weight = parse_weight(tin["label"]) if tin else None
    if tin and not weight:
        m = re.search(r"Bo[iî]te[^\d€]{0,30}(\d+(?:[.,]\d+)?\s*(?:kg|g)\b)", page_text, re.I)
        weight = parse_weight(m.group(1)) if m else None
    if tin and price is None:
        m = re.search(r"Bo[iî]te[^€]{0,80}?(\d+[.,]\d{2})\s*€", page_text, re.I)
        if m:
            price = float(m.group(1).replace(",", "."))
            price_note = "text next to option label"

    return {
        "market": market,
        "url": url,
        "name": name,
        "tea_type": subtitle,
        "tea_category": category,
        "tea_category_source": category_source,
        "origin_country": country,
        "origin_detail": origin_detail,
        "origin_source": origin_source,
        "boite_metal_found": tin is not None,
        "boite_metal_label": tin["label"] if tin else None,
        "boite_metal_weight": weight,
        "boite_metal_price_eur": price,
        "price_source": price_note if price is not None else None,
        "options_seen": [v["label"] or v["image"] for v in variants],
    }


# ----------------------------------------------------------------------
# Spider
# ----------------------------------------------------------------------
class PureScraperSpider(scrapy.Spider):
    name = "pure_scraper"

    def __init__(self, shops_config=None, max_products=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shops_config = shops_config or []
        self.max_products = max_products
        self.seen_products = set()
        self.seen_pages = set()
        self.target_tea_keywords = ['china', 'matcha', 'jasmin', 'oolong', 'noir', 'schwarztee', 'gruentee']
        self.forbidden_accessories = ['mug', 'tasse', 'accessoire', 'theiere', 'boite', 'coffret',
                                      'livres-et-guides', 'dose', 'zubehoer']

    def start_requests(self):
        for shop in self.shops_config:
            self.logger.info(f"[{shop['market']}] Starting request: {shop['url']}")
            yield scrapy.Request(
                url=shop["url"],
                callback=self.parse,
                meta={"market": shop["market"], "lang": shop["lang"]},
                dont_filter=True,
            )

    async def start(self):
        """Scrapy 2.13+ entry point. Without it, no request is sent and you get []."""
        for request in self.start_requests():
            yield request

    def parse(self, response):
        market = response.meta.get("market", "UNKNOWN")
        self.logger.info(f"[{market}] Catalog page {response.url} -> status {response.status}")

        tile_links = response.css(
            'a.product-item-link::attr(href), '
            '.product-item a.product-item-photo::attr(href), '
            '.product-item-info a::attr(href), '
            'li.product a::attr(href), '
            '.product-tile a::attr(href)'
        ).getall()
        links = response.css('a::attr(href)').getall()
        candidates = tile_links if tile_links else links
        self.logger.info(f"[{market}] links={len(links)} tile_links={len(tile_links)}")

        
        product_links = []
        for link in candidates:
            absolute_url = response.urljoin(link).split('#')[0].split('?')[0]
            link_lower = absolute_url.lower()
            if not link_lower.endswith('.html') or any(bad in link_lower for bad in self.forbidden_accessories):
                continue
            if not tile_links and not any(k in link_lower for k in self.target_tea_keywords):
                continue
            if absolute_url not in product_links:
                product_links.append(absolute_url)

        for link in product_links:
            if link in self.seen_products:
                continue
            if self.max_products is not None and len(self.seen_products) >= self.max_products:
                break
            self.seen_products.add(link)
            yield response.follow(link, callback=self.parse_product, meta={"market": market})

        if self.max_products is None or len(self.seen_products) < self.max_products:
            next_pages = response.css(
                'link[rel="next"]::attr(href), a[rel="next"]::attr(href), '
                'a.next::attr(href), li.pages-item-next a::attr(href)'
            ).getall()
            next_pages += [l for l in links if re.search(r'[?&](p|page)=\d+', l)]
            for page in next_pages:
                page_url = response.urljoin(page)
                if page_url not in self.seen_pages:
                    self.seen_pages.add(page_url)
                    yield response.follow(page_url, callback=self.parse, meta=response.meta)
        
    def parse_product(self, response):
        global DEBUG_SAVED
        # Keep the raw HTML of the first pages so extraction problems can be inspected
        if DEBUG_SAVED < DEBUG_MAX:
            debug_dir = os.path.join(TARGET_DIR, "debug_html")
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, f"product_{DEBUG_SAVED + 1}.html"), "w", encoding="utf-8") as f:
                f.write(response.text)
            DEBUG_SAVED += 1

        soup = BeautifulSoup(response.text, "html.parser")
        data = extract_product(soup, response.url, response.meta.get("market"))
        RAW_PAGES.append(data)

        if data["boite_metal_found"]:
            self.logger.info(f"#{len(RAW_PAGES)} {data['name']} | {data['tea_category']} | {data['tea_type']} | {data['origin_country']} | "
                             f"{data['boite_metal_weight']} | {data['boite_metal_price_eur']} EUR")
        else:
            self.logger.warning(f"#{len(RAW_PAGES)} {data['name']}: NO metal tin option. "
                                f"Options seen: {data['options_seen']}")


def run_phase_one():
    os.makedirs(TARGET_DIR, exist_ok=True)

    shops = [{"url": "https://www.palaisdesthes.com/fr/thes/", "market": "FR", "lang": "fr"}]

    process = CrawlerProcess(settings={
        'USER_AGENT': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
        'DOWNLOAD_DELAY': 2,
        'ROBOTSTXT_OBEY': False,
        'AUTOTHROTTLE_ENABLED': True,
        'LOG_LEVEL': 'INFO',
    })

    # max_products=5 for a quick test; set to None for the full crawl
    process.crawl(PureScraperSpider, shops_config=shops, max_products=None)
    process.start()

    output_file = os.path.join(TARGET_DIR, "raw_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(RAW_PAGES, f, indent=2, ensure_ascii=False)
    found = sum(1 for p in RAW_PAGES if p["boite_metal_found"])
    print(f"\n[Phase 1 Complete] {len(RAW_PAGES)} pages, {found} with a metal tin -> {output_file}")


if __name__ == "__main__":
    run_phase_one()