import os
import json
import re
import scrapy
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup

TARGET_DIR = r"D:\Business\Freelancer\Webscraper+Datawrangling"
RAW_PAGES = []
DEBUG_SAVED = 0
DEBUG_MAX = 2

COUNTRIES = {
    "Chine": "Chine", "Japon": "Japon", "Inde": "Inde", "Sri Lanka": "Sri Lanka",
    "Taïwan": "Taïwan", "Taiwan": "Taïwan", "Népal": "Népal", "Nepal": "Népal",
    "Kenya": "Kenya", "Corée du Sud": "Corée du Sud", "Vietnam": "Vietnam", 
    "Thaïlande": "Thaïlande", "Afrique du Sud": "Afrique du Sud", "Italie": "Italie", 
    "France": "France", "Colombie": "Colombie", "Pérou": "Pérou", "Brésil": "Brésil"
}

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
    h1 = soup.select_one("h1.h3, h1.page-title, h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(" ", strip=True)
    og = soup.select_one('meta[property="og:title"]')
    return og["content"].strip() if og and og.get("content") else None

def extract_product(soup, url, market):
    name = find_name(soup)

    # Price extraction
    price = None
    price_el = soup.select_one('span.current-price-value, [itemprop="price"], .current-price span, meta[property="product:price:amount"]')
    if price_el:
        p_text = price_el.get("content") or price_el.get("aria-label") or price_el.get_text(strip=True)
        m = re.search(r'(\d+[.,]\d{2})', p_text.replace('\xa0', '').replace(',', '.'))
        if m:
            try:
                price = float(m.group(1))
            except ValueError:
                pass
                
    if price is None:
        page_text = " ".join(soup.stripped_strings)
        m = re.search(r'(\d+[.,]\d{2})\s*€', page_text)
        if m:
            try:
                price = float(m.group(1).replace(',', '.'))  # Fixed typo here
            except ValueError:
                pass

    # Weight extraction
    page_text = " ".join(soup.stripped_strings)
    weight_match = re.search(r'(\d+)\s*g\b', page_text, re.I)
    weight = f"{weight_match.group(1)}g" if weight_match else None

    # Tea type/category
    tea_type = None
    category_match = re.search(r'\b(th[ée]s?\s+(?:noir|vert|blanc|oolong)|rooibos|infusion|tisane|carcadet)\b', page_text, re.I)
    if category_match:
        tea_type = category_match.group(1)

    # Country of origin restricted to data sheets
    origin_country = None
    for el in soup.select('.data-sheet, .product-features, .product-description'):
        txt = el.get_text(" ", strip=True)
        found = countries_in(txt)
        if found:
            origin_country = ", ".join(found)
            break

    return {
        "market": market,
        "url": url,
        "name": name,
        "tea_type": tea_type,
        "origin_country": origin_country,
        "weight": weight,
        "price_eur": price,
    }


class PureScraperSpider(scrapy.Spider):
    name = "pure_scraper_fr2"

    def __init__(self, shops_config=None, max_products=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shops_config = shops_config or []
        self.max_products = max_products
        self.seen_products = set()
        self.seen_pages = set()

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
        for request in self.start_requests():
            yield request

    def parse(self, response):
        market = response.meta.get("market", "UNKNOWN")
        self.logger.info(f"[{market}] Catalog page {response.url} -> status {response.status}")

        all_links = response.css('a::attr(href)').getall()
        product_links = []
        for link in all_links:
            absolute_url = response.urljoin(link).split('#')[0].split('?')[0]
            # STRICT FILTER: Only grab product pages that are specifically tin/boîte versions and end in .html
            if absolute_url.startswith("https://www.dammann.fr/fr/") and absolute_url.endswith(".html"):
                if 'boite' in absolute_url:  
                    if absolute_url not in product_links:
                        product_links.append(absolute_url)

        self.logger.info(f"[{market}] Found {len(product_links)} boîte product links on this page.")

        for link in product_links:
            if link in self.seen_products:
                continue
            if self.max_products is not None and len(self.seen_products) >= self.max_products:
                break
            self.seen_products.add(link)
            yield response.follow(link, callback=self.parse_product, meta={"market": market})

        # Pagination handling
        next_page = response.css(
            'a.next::attr(href), '
            '.page-list .next a::attr(href), '
            '.pagination a.js-search-link::attr(href), '
            'link[rel="next"]::attr(href)'
        ).get()

        if next_page:
            next_url = response.urljoin(next_page)
            if next_url not in self.seen_pages:
                self.seen_pages.add(next_url)
                self.logger.info(f"[{market}] Following next page: {next_url}")
                yield response.follow(next_url, callback=self.parse, meta=response.meta)
        
    def parse_product(self, response):
        global DEBUG_SAVED
        if DEBUG_SAVED < DEBUG_MAX:
            debug_dir = os.path.join(TARGET_DIR, "debug_dammann_html")
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, f"product_{DEBUG_SAVED + 1}.html"), "w", encoding="utf-8") as f:
                f.write(response.text)
            DEBUG_SAVED += 1

        soup = BeautifulSoup(response.text, "html.parser")
        data = extract_product(soup, response.url, response.meta.get("market"))
        RAW_PAGES.append(data)

        self.logger.info(f"#{len(RAW_PAGES)} {data['name']} | Type: {data['tea_type']} | Origin: {data['origin_country']} | Weight: {data['weight']} | Price: {data['price_eur']} EUR")


def run_phase_one():
    os.makedirs(TARGET_DIR, exist_ok=True)

    shops = [
        {"url": "https://www.dammann.fr/fr/boites-et-sachets/boites/", "market": "FR", "lang": "fr"}
    ]

    process = CrawlerProcess(settings={
        'USER_AGENT': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
        'DOWNLOAD_DELAY': 2,
        'ROBOTSTXT_OBEY': False,
        'AUTOTHROTTLE_ENABLED': True,
        'LOG_LEVEL': 'INFO',
    })

    # Set to None to scrape all pages and all tin products
    process.crawl(PureScraperSpider, shops_config=shops, max_products=None)
    process.start()

    output_file = os.path.join(TARGET_DIR, "dammann_raw_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(RAW_PAGES, f, indent=2, ensure_ascii=False)
    print(f"\n[Phase 1 Complete] {len(RAW_PAGES)} pages scraped -> {output_file}")


if __name__ == "__main__":
    run_phase_one()