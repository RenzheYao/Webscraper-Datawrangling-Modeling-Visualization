"""
process_tea_data.py - merge the scraper JSON files, normalise them to English with
rules + a local Qwen3-4B model (via Ollama), then export JSON and CSV.

Output columns:
    market, company, name, tea_type, weight, price, country_of_origin

How the work is split (this keeps the small model's job small and the results stable):
  * market            copied from the JSON
  * company           derived from the url in code (never by the LLM)
  * name              kept as scraped (product names are proper names, not translated)
  * tea_type          rules (EN/FR/DE keywords, instant). The LLM is only asked when the record HAS a
                      type value that the rules cannot map (e.g. an unfamiliar spelling) - if the record
                      says nothing about the type there is nothing to extract, so no LLM call.
                      Blends of several colours ("thé noir et vert") get NO type (null).
  * country_of_origin only what the scraped data states, translated to English (rules, then LLM
                      for unfamiliar spellings). No origin in the data -> null.
  * weight, price     parsed in code and written as "100 g" and "€12,50" (missing -> null)

The LLM is called through Ollama with a JSON schema, temperature 0 and thinking off, and every
answer is validated (tea_type must be in the allowed list, country must look like a country).
Records where the LLM supplied a value are written to needs_review.json so you can spot-check.
If Ollama is not running, the script still works with rules only.

Setup:
    1. install Ollama, then:  ollama pull qwen3-4b:latest
    2. python process_tea_data.py
"""
import csv
import html
import json
import os
import re
import sys
import unicodedata
import urllib.request
from collections import Counter
from urllib.parse import urlparse

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
DATA_DIR = r"D:\Business\Freelancer\Webscraper+Datawrangling"   # where the scraper JSON files are
JSON_FILES = [
    "raw_results.json",
    "dammann_raw_results.json",
    "kusmi_raw_results.json",
    "mariage_raw_results.json",
    "betjeman_raw_results.json",
]
OUTPUT_JSON = "combined_processed.json"
OUTPUT_CSV = "combined_output.csv"
REVIEW_JSON = "needs_review.json"

COLUMNS = ["market", "company", "name", "tea_type", "weight", "price", "country_of_origin"]

# False: country_of_origin is filled only when the scraped record has an origin value.
# True : also guess it from the product name (e.g. "Darjeeling ..." -> India).
INFER_COUNTRY_FROM_NAME = True

# Countries that show up in scraped origins only because the page mentions where the tea is packed
# or the shop is based ("Paris, France"); they are never the origin of the leaves. Ignored.
NOT_TEA_ORIGINS = {"France", "Italy"}

# Skip records that have neither a weight nor a price (e.g. Palais des Thés products sold without a box).
DROP_EMPTY_ROWS = True

USE_LLM = True
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3-4b:latest"
LLM_TIMEOUT = 180            # seconds per call (first call loads the model)
LLM_MAX_CONSECUTIVE_FAILURES = 5

PRETTY_BRAND_NAMES = True    # False -> company is just the domain label, e.g. "kusmitea"

# Key names the scrapers may use; the first non-empty one wins.
KEYS = {
    "market": ["market"],
    "url": ["url", "link", "product_url"],
    "name": ["name", "title", "product_name"],
    "tea_type": ["tea_type", "type", "category"],
    "weight": ["weight", "net_weight", "boite_metal_weight", "size"],      # boite_metal_*: Palais des Thés file
    "price": ["price_eur", "price", "boite_metal_price_eur", "price_value"],
    "country": ["origin_country", "country_of_origin", "origin", "country"],
}

BRANDS = {
    "kusmitea": "Kusmi Tea",
    "mariagefreres": "Mariage Frères",
    "betjemanandbarton": "Betjeman & Barton",
    "dammann": "Dammann Frères",
    "dammann-freres": "Dammann Frères",
    "dammannfreres": "Dammann Frères",
    "palaisdesthes":"palais des thes"
}

# ----------------------------------------------------------------------
# Tea types.  Order matters: SPECIFIC types win over colour types, because you listed
# "jasmine tea, matcha, earl grey" as types of their own.  Reorder if you prefer otherwise.
# A text that matches two different colour types (e.g. "thé noir et vert") is ambiguous and
# goes to the LLM.  Keywords are accent-free lowercase.
# ----------------------------------------------------------------------
SPECIFIC_TYPES = [
    ("Matcha", ["matcha"]),
    ("Jasmine tea", ["jasmine", "jasmin", "jasmintee"]),
    ("Earl Grey", ["earl grey"]),
    ("Chai", ["chai", "masala"]),
    ("Pu-erh tea", ["pu-erh", "puerh", "pu erh"]),
    ("Dark tea", ["the sombre", "dark tea", "hei cha"]),                       # fermented teas ("thé sombre")
    ("Oolong tea", ["oolong", "wulong", "blue tea", "the bleu", "blauer tee"]),  # "thé bleu" = oolong
    ("Rooibos", ["rooibos", "redbush", "red bush"]),
]
BASIC_TYPES = [
    ("Green tea", ["green tea", "the vert", "gruner tee", "gruntee", "grun tee"]),
    ("Black tea", ["black tea", "the noir", "schwarzer tee", "schwarztee", "schwarz tee", "tee schwarz"]),
    ("White tea", ["white tea", "the blanc", "weisser tee", "weisstee", "weiss tee"]),
    ("Yellow tea", ["yellow tea", "the jaune", "gelber tee"]),
    ("Herbal tea", ["herbal tea", "herbal infusion", "tisane", "infusion", "krautertee", "krauter"]),
    ("Fruit tea", ["fruit tea", "the de fruits", "fruchtetee", "fruchte tee"]),
]
TYPE_CHOICES = [t for t, _ in SPECIFIC_TYPES + BASIC_TYPES] + ["Flavoured tea", "Other"]
_COLOUR = r"(?:noir|vert|blanc|black|green|white|schwarz\w*|grun\w*|weiss\w*)"
TWO_COLOURS_RE = re.compile(
    r"(?<!\w)" + _COLOUR + r"\s+(?:et|and|und|&)\s+" + _COLOUR + r"(?!\w)"
    r"|(?<!\w)(?:thes?|teas?|tees?)\s+(?:et|and|und|&)\s+(?:infusions?|tisanes?|herbal)")   # "thés et infusions"

OOLONG_BLEND_RE = re.compile(                                  # "thé vert et oolong"
    r"(?<!\w)" + _COLOUR + r"\s+(?:et|and|und|&)\s+oolongs?(?!\w)"
    r"|(?<!\w)oolongs?\s+(?:et|and|und|&)\s+" + _COLOUR + r"(?!\w)")

# Scraped type fields sometimes contain the site's navigation menu or page boilerplate.
BOILERPLATE_RE = re.compile(r"passer au contenu|pr[eé]c[eé]dent", re.I)   # keep only the text before this
MENU_RE = re.compile(r"voir tout", re.I)                                  # the field is the menu itself -> ignore it

# ----------------------------------------------------------------------
# Countries (English name -> aliases in EN / FR / DE, plus famous tea regions).
# ----------------------------------------------------------------------
COUNTRY_ALIASES = {
    "China": ["china", "chine", "yunnan", "keemun", "fujian"],
    "Japan": ["japan", "japon"],
    "India": ["india", "inde", "indien", "darjeeling", "assam", "nilgiri"],
    "Sri Lanka": ["sri lanka", "ceylon", "ceylan", "dimbula", "nuwara eliya"],
    "Taiwan": ["taiwan"],
    "Nepal": ["nepal"],
    "Kenya": ["kenya", "kenia"],
    "South Korea": ["south korea", "coree du sud", "sudkorea", "korea"],
    "Vietnam": ["vietnam", "viet nam"],
    "Thailand": ["thailand", "thailande"],
    "South Africa": ["south africa", "afrique du sud", "sudafrika"],
    "Italy": ["italy", "italie", "italien"],
    "France": ["france", "frankreich"],
    "Germany": ["germany", "allemagne", "deutschland"],
    "Colombia": ["colombia", "colombie", "kolumbien"],
    "Peru": ["peru", "perou"],
    "Brazil": ["brazil", "bresil", "brasilien"],
    "Laos": ["laos"],
    "Indonesia": ["indonesia", "indonesie", "indonesien"],
    "Turkey": ["turkey", "turquie", "turkei"],
    "Malawi": ["malawi"],
    "Rwanda": ["rwanda"],
    "Tanzania": ["tanzania", "tanzanie", "tansania"],
    "Burundi": ["burundi"],
    "Madagascar": ["madagascar", "madagaskar"],
    "Iran": ["iran"],
    "Morocco": ["morocco", "maroc", "marokko"],
    "Argentina": ["argentina", "argentine", "argentinien"],
    "Uganda": ["uganda", "ouganda"],
    "Cameroon": ["cameroon", "cameroun", "kamerun"],
    "Ethiopia": ["ethiopia", "ethiopie", "athiopien"],
    "Bangladesh": ["bangladesh"],
    "Myanmar": ["myanmar", "burma", "birmanie", "birma"],
    "Mozambique": ["mozambique", "mosambik"],
    "Cambodia": ["cambodia", "cambodge", "kambodscha"],
    "Mexico": ["mexico", "mexique", "mexiko"],
    "Georgia": ["georgia", "georgie", "georgien"],
    "Egypt": ["egypt", "egypte", "agypten"],
}
COUNTRY_LOOKUP = {alias: country for country, aliases in COUNTRY_ALIASES.items() for alias in aliases}
COUNTRY_TEXT_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(a) for a in sorted(COUNTRY_LOOKUP, key=len, reverse=True)) + r")(?!\w)")

UNIT_TO_G = {"g": 1.0, "gr": 1.0, "kg": 1000.0, "oz": 28.3495, "lb": 453.592, "lbs": 453.592}


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def norm(s):
    """lowercase, accent-free, single-spaced"""
    s = html.unescape(str(s or "")).replace("ß", "ss")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", s).strip()


def has_kw(text, keywords):
    """Whole-word match; every word may carry a plural: 'the noir' also finds 'thés noirs'."""
    for k in keywords:
        pattern = r"[\s-]+".join(re.escape(w) + r"(?:s|es)?" for w in re.split(r"[\s-]+", k))
        if re.search(r"(?<!\w)" + pattern + r"(?!\w)", text):
            return True
    return False


def pick(rec, field):
    for key in KEYS[field]:
        v = rec.get(key)
        if v is not None and str(v).strip() != "":
            return v
    return None


# ----------------------------------------------------------------------
# Deterministic extractors
# ----------------------------------------------------------------------
def company_from_url(url):
    if not url:
        return None
    host = (urlparse(url if "//" in str(url) else "//" + str(url)).hostname or "").lower()
    parts = [p for p in host.split(".") if p and p != "www"]
    if len(parts) < 2:
        return None
    label = parts[-3] if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net"} else parts[-2]
    if not PRETTY_BRAND_NAMES:
        return label
    return BRANDS.get(label, label.replace("-", " ").title())


def clean_type_text(v):
    """Remove navigation-menu text that the scraper picked up together with the tea type."""
    if v is None:
        return None
    text = str(v)
    if MENU_RE.search(text):
        return None
    return BOILERPLATE_RE.split(text, maxsplit=1)[0].strip() or None


def parse_price(v):
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return round(float(v), 2)
    s = re.sub(r"[^\d.,]", "", str(v))
    if not s:
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def parse_weight_g(v):
    """-> grams as a number, or None (e.g. '25 sachets' is not a weight)."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        grams = float(v)
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|gr|g|oz|lbs|lb)\b", norm(v).replace(",", "."))
        if not m:
            return None
        grams = float(m.group(1)) * UNIT_TO_G[m.group(2)]
    grams = round(grams, 1)
    return int(grams) if grams == int(grams) else grams


def format_price(value):
    """12.5 -> '€12,50'"""
    return None if value is None else "€" + f"{value:.2f}".replace(".", ",")


def format_weight(grams):
    """100 -> '100 g', 99.2 -> '99,2 g'"""
    return None if grams is None else f"{grams} g".replace(".", ",")


def rule_tea_type(text):
    """-> (type or None, blocked).  blocked=True means the text names several colour types
    (a blend such as 'thé noir et vert'): the type stays null and the LLM is not asked."""
    t = norm(text)
    if OOLONG_BLEND_RE.search(t):
        return None, True
    for label, kws in SPECIFIC_TYPES:
        if has_kw(t, kws):
            return label, False
    if TWO_COLOURS_RE.search(t):                      # "thé noir et vert", "black and green tea"
        return None, True
    hits = [label for label, kws in BASIC_TYPES if has_kw(t, kws)]
    if len(hits) == 1:
        return hits[0], False
    return None, len(hits) > 1                        # nothing found -> LLM may try; several -> blend


def rule_country(origin_raw, name):
    """-> (value or None, complete).  complete=False means the LLM should look at it."""
    if origin_raw:
        found, unknown = [], False
        for piece in re.split(r"[,;/&]|\band\b|\bet\b|\bund\b", str(origin_raw)):
            if not piece.strip():
                continue
            c = COUNTRY_LOOKUP.get(norm(piece))
            if c:
                if c not in found:
                    found.append(c)
            else:
                unknown = True
        if found:
            kept = [c for c in found if c not in NOT_TEA_ORIGINS]
            return (", ".join(kept) or None), not unknown
        return None, False
    if not INFER_COUNTRY_FROM_NAME:
        return None, True            # no origin in the data -> null, nothing for the LLM to do
    found = []
    for m in COUNTRY_TEXT_RE.finditer(norm(name)):
        c = COUNTRY_LOOKUP[m.group(1)]
        if c not in found and c not in NOT_TEA_ORIGINS:
            found.append(c)
    return (", ".join(found), True) if found else (None, False)


# ----------------------------------------------------------------------
# LLM (Ollama + Qwen3-4B)
# ----------------------------------------------------------------------
def build_system_prompt():
    if INFER_COUNTRY_FROM_NAME:
        country_rule = (
            "country_of_origin: the English name of the country where the tea is grown; several countries "
            "separated by \", \". Use it only if the text names a country, or a famous tea region or estate that "
            "belongs to exactly one country (for example Darjeeling -> India). Never use the country where the "
            "tea is packed or sold. Otherwise answer \"Unknown\".")
    else:
        country_rule = (
            "country_of_origin: the English name(s) of the country named in origin_text, separated by \", \". "
            "Use origin_text only - never the product name or the url. If origin_text names no country, "
            "answer \"Unknown\".")
    return (
        "You normalise tea product data. Reply with JSON only.\n"
        "Use ONLY the text you are given. The text may be English, French or German; always answer in English.\n\n"
        f"tea_type: choose exactly one of: {', '.join(TYPE_CHOICES)}.\n"
        "Pick the most specific one that applies (Matcha, Jasmine tea, Earl Grey, Chai, Pu-erh tea, Oolong tea, "
        "Rooibos) before the colour types (Green tea, Black tea, White tea). If the text names two or more "
        "different tea colours (for example black and green), or does not say, answer \"Unknown\".\n\n"
        + country_rule)


def ollama_ready():
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=5) as r:
            models = [m.get("name", "") for m in json.load(r).get("models", [])]
    except Exception as e:
        print(f"[llm] Ollama not reachable at {OLLAMA_URL} ({e!r}) -> rules only")
        return False
    if not any(m.startswith(OLLAMA_MODEL) for m in models):
        print(f"[llm] model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}   -> rules only")
        return False
    return True


def call_llm(view, need):
    """Ask Qwen3 for the fields in `need` ('tea_type', 'country'). Returns a dict."""
    props = {}
    if "tea_type" in need:
        props["tea_type"] = {"type": "string", "enum": TYPE_CHOICES + ["Unknown"]}
    if "country" in need:
        props["country_of_origin"] = {"type": "string"}
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "think": False,                                   # Qwen3: no reasoning, just the answer
        "format": {"type": "object", "properties": props, "required": list(props)},
        "options": {"temperature": 0, "num_ctx": 4096},
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": json.dumps(view, ensure_ascii=False) + "\n/no_think"},
        ],
    }
    req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
        content = json.load(r)["message"]["content"]
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    return json.loads(content)


def clean_llm_type(v):
    if isinstance(v, str) and v.strip() in TYPE_CHOICES:
        return v.strip()
    return None            # includes "Unknown" and anything outside the allowed list


def clean_llm_country(v):
    if not isinstance(v, str):
        return None
    out = []
    for piece in re.split(r"[,;/&]", v):
        piece = piece.strip()
        if not piece or norm(piece) in {"unknown", "none", "n/a", "null"}:
            continue
        c = COUNTRY_LOOKUP.get(norm(piece))              # maps FR/DE names the model may leak
        if c is None and re.fullmatch(r"[A-Z][A-Za-z .'-]{2,30}", piece):
            c = piece
        if c and c not in out:
            out.append(c)
    return ", ".join(out) or None


# ----------------------------------------------------------------------
# Record processing
# ----------------------------------------------------------------------
class Stats:
    def __init__(self):
        self.n = 0
        self.rule = {"tea_type": 0, "country": 0}
        self.llm = {"tea_type": 0, "country": 0}
        self.missing = {"tea_type": 0, "country": 0, "weight_g": 0, "price_eur": 0}
        self.llm_calls = 0
        self.llm_failures = 0
        self.consecutive_failures = 0
        self.unmatched = Counter()      # values present in the data that nothing could map


def process_record(rec, source, llm, cache, stats, review):
    """Returns the normalised row. `llm` is a callable (view, need) -> dict, or None."""
    name = re.sub(r"\s+", " ", html.unescape(str(pick(rec, "name") or ""))).strip() or None
    raw_type, raw_origin = clean_type_text(pick(rec, "tea_type")), pick(rec, "country")
    url = pick(rec, "url")

    tea_type, type_blocked = rule_tea_type(f"{raw_type or ''} {name or ''}")
    row = {
        "market": pick(rec, "market"),
        "company": company_from_url(url),
        "name": name,
        "tea_type": tea_type,
        "weight": format_weight(parse_weight_g(pick(rec, "weight"))),
        "price": format_price(parse_price(pick(rec, "price"))),
        "country_of_origin": None,
    }
    country, country_complete = rule_country(raw_origin, name)
    row["country_of_origin"] = country
    stats.n += 1
    rule_type, rule_country_value = row["tea_type"], country
    if rule_type:
        stats.rule["tea_type"] += 1
    if country and country_complete:
        stats.rule["country"] += 1

    need = []
    if row["tea_type"] is None and not type_blocked and raw_type:   # blends keep a null type
        need.append("tea_type")
    if not country_complete:
        need.append("country")

    llm_filled = {}
    if need and llm is not None and stats.consecutive_failures < LLM_MAX_CONSECUTIVE_FAILURES:
        view = {"name": name, "type_text": raw_type, "origin_text": raw_origin, "url": url}
        key = json.dumps([view, need], sort_keys=True, ensure_ascii=False)
        if key not in cache:
            answer = None
            for _attempt in range(2):
                try:
                    stats.llm_calls += 1
                    answer = llm(view, need)
                    stats.consecutive_failures = 0
                    break
                except Exception as e:
                    stats.llm_failures += 1
                    stats.consecutive_failures += 1
                    print(f"  ! LLM call failed ({e!r})")
            cache[key] = answer or {}
        answer = cache[key]

        if "tea_type" in need:
            t = clean_llm_type(answer.get("tea_type"))
            if t:
                row["tea_type"] = t
                llm_filled["tea_type"] = t
                stats.llm["tea_type"] += 1
        if "country" in need:
            c = clean_llm_country(answer.get("country_of_origin"))
            if c:
                row["country_of_origin"] = c
                llm_filled["country_of_origin"] = c
                stats.llm["country"] += 1

    if raw_type and row["tea_type"] is None and not type_blocked:
        stats.unmatched[f"tea_type: {raw_type}"] += 1
    if raw_origin and row["country_of_origin"] is None:
        stats.unmatched[f"origin: {raw_origin}"] += 1

    if llm_filled:
        review.append({"source": source, "url": url, "name": name, "llm_filled": llm_filled,
                       "rule_tea_type": rule_type, "rule_country": rule_country_value,
                       "raw_type": raw_type, "raw_origin": raw_origin})

    for col, stat_key in (("tea_type", "tea_type"), ("country_of_origin", "country"),
                          ("weight", "weight_g"), ("price", "price_eur")):
        if row[col] is None:
            stats.missing[stat_key] += 1
    return row


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def load_records():
    records = []
    for file in JSON_FILES:
        path = file if os.path.isabs(file) else os.path.join(DATA_DIR, file)
        if not os.path.exists(path):
            print(f"[warn] {path} not found - skipped")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[error] {path} is not valid JSON ({e}) - skipped")
            continue
        items = [data] if isinstance(data, dict) else data if isinstance(data, list) else []
        items = [i for i in items if isinstance(i, dict)]
        print(f"[load] {file}: {len(items)} records")
        records += [(file, i) for i in items]
    return records


def main():
    records = load_records()
    if not records:
        print("No records loaded - check DATA_DIR / JSON_FILES.")
        return

    llm = call_llm if (USE_LLM and ollama_ready()) else None
    stats, cache, review, rows = Stats(), {}, [], []
    dropped = Counter()
    for i, (source, rec) in enumerate(records, start=1):
        row = process_record(rec, source, llm, cache, stats, review)
        if DROP_EMPTY_ROWS and row["weight"] is None and row["price"] is None:
            dropped[source] += 1
            continue
        rows.append(row)
        if llm is not None and i % 25 == 0:
            print(f"  ... {i}/{len(records)} records, {stats.llm_calls} LLM calls so far")

    out_json = os.path.join(DATA_DIR, OUTPUT_JSON)
    out_csv = os.path.join(DATA_DIR, OUTPUT_CSV)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:      # utf-8-sig: opens correctly in Excel
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: ("" if r[c] is None else r[c]) for c in COLUMNS})
    if review:
        with open(os.path.join(DATA_DIR, REVIEW_JSON), "w", encoding="utf-8") as f:
            json.dump(review, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {len(rows)} rows written ({stats.n} records read) -> {out_csv}")
    for source, n in dropped.items():
        print(f"  skipped {n} records from {source}: no weight and no price (DROP_EMPTY_ROWS)")
    empty = {c: sum(1 for r in rows if r[c] is None) for c in COLUMNS}
    print(f"  empty cells: tea_type {empty['tea_type']} | weight {empty['weight']} | price {empty['price']} "
          f"| country_of_origin {empty['country_of_origin']}   (of {len(rows)} rows)")
    print(f"  filled by the LLM: tea_type {stats.llm['tea_type']}, country {stats.llm['country']}")
    print(f"  LLM calls: {stats.llm_calls} ({stats.llm_failures} failed) | spot-check list: "
          f"{REVIEW_JSON if review else 'none needed'}")
    if stats.unmatched:
        print("  values in the data that could not be mapped (add keywords for the frequent ones):")
        for value, n in stats.unmatched.most_common(15):
            print(f"    {n:>4} x {value}")


if __name__ == "__main__":
    main()