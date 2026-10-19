"""
wrangle_tea_csv.py - clean the combined tea CSV (market, company, name, tea_type, weight, price,
country_of_origin).

Steps, in this order:
  0. read every cell as text, trim spaces, collapse repeated spaces, and treat "", "null", "none",
     "n/a", "nan" as missing
  0b. set market from the company's country (French company -> FR, German company -> DE), so that
      duplicates are judged on the final values
  1. find exact duplicate rows (every column identical)
  2. remove them (the first occurrence is kept)
  3. drop rows whose weight or price is missing
  4. fill a missing country_of_origin with "Unknown"
  4b. add the column extracted_at (date, month, year and time of the extraction) just before `market`
  5. write <name>_clean.csv, and <name>_removed.csv listing every dropped row and why

Usage:
    pip install pandas
    python wrangle_tea_csv.py combined_output.csv
    python wrangle_tea_csv.py combined_output.csv -o cleaned.csv
    python wrangle_tea_csv.py combined_output.csv --extracted-at "2026-10-18 09:30"   # when the scrape really ran
"""
import argparse
import os
from datetime import datetime
import re
import sys
import unicodedata
from collections import Counter

import pandas as pd

# ----------------------------------------------------------------------
# Rules (edit here)
# ----------------------------------------------------------------------
DROP_IF_MISSING = ["weight", "price"]                  # a row with an empty value in any of these is dropped
FILL_MISSING = {"country_of_origin": "Unknown"}        # column -> value used when it is empty
MISSING_TOKENS = {"", "null", "none", "n/a", "nan"}    # cell texts that count as "missing" (case-insensitive)
NEAR_DUPLICATE_KEY = ["company", "name", "weight"]     # only used for the informational check at the end

# Market rule: market = market of the country the COMPANY belongs to (not the language of the site).
# Company names are matched ignoring case, accents, spaces and "&" vs "and". Add new companies here.
COMPANY_COUNTRY = {
    "Kusmi Tea": "France",
    "Mariage Frères": "France",
    "Dammann Frères": "France",
    "Palais des Thés": "France",
    "Betjeman & Barton": "France",
    "TeeGschwendner": "Germany",       # examples for German companies
    "Ronnefeldt": "Germany",
    "Teekanne": "Germany",
}
COUNTRY_MARKET = {"France": "FR", "Germany": "DE"}

# extracted_at column: day/month/year and time of the extraction, placed right before `market`
EXTRACTED_COLUMN = "extracted_at"
EXTRACTED_FORMAT = "%d/%m/%Y %H:%M"          # e.g. 18/10/2026 09:30


def company_key(name):
    """'Palais des Thés', 'palais des thes' and 'Palaisdesthes' all become 'palaisdesthes'."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]", "", s)


KEY_TO_MARKET = {company_key(c): COUNTRY_MARKET[country] for c, country in COMPANY_COUNTRY.items()}


def apply_market_rule(df):
    """Set `market` from the company. Companies that are not in COMPANY_COUNTRY keep their market."""
    new = df["company"].map(lambda c: KEY_TO_MARKET.get(company_key(c)) if pd.notna(c) else None)
    known = new.notna()
    changed = known & (df["market"].fillna("") != new.fillna(""))
    info = {
        "rows_changed": int(changed.sum()),
        "changes": Counter(zip(df.loc[changed, "company"], df.loc[changed, "market"].fillna(""), new[changed])),
        "unknown_companies": sorted(df.loc[~known & df["company"].notna(), "company"].unique()),
    }
    df = df.copy()
    df["market"] = new.where(known, df["market"])
    return df, info


def add_extracted_at(df, stamp):
    """Insert the extraction date/time as a column just before `market`."""
    df = df.drop(columns=EXTRACTED_COLUMN, errors="ignore")
    df.insert(df.columns.get_loc("market"), EXTRACTED_COLUMN, stamp)
    return df


def parse_stamp(text):
    for fmt in (EXTRACTED_FORMAT, "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).strftime(EXTRACTED_FORMAT)
        except ValueError:
            pass
    sys.exit(f'Cannot read --extracted-at "{text}". Use e.g. "2026-10-18 09:30" or "18/10/2026 09:30".')


def load(path):
    """Read as text only, so '100 g' and '€19,80' stay exactly as written."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for col in df.columns:
        s = df[col].str.replace(r"\s+", " ", regex=True).str.strip()
        df[col] = s.mask(s.str.lower().isin(MISSING_TOKENS), pd.NA)
    return df


def wrangle(df):
    """Returns (clean_df, removed_df, stats). `removed_df` has the reason for every dropped row."""
    df = df.copy()
    df.insert(0, "source_line", df.index + 2)            # line number in the CSV file (line 1 is the header)
    data_cols = [c for c in df.columns if c != "source_line"]
    stats = {"rows_in": len(df)}

    # 0b. market from the company's country
    df, stats["market"] = apply_market_rule(df)

    # 1-2. exact duplicates: every column identical; keep the first occurrence
    dup = df.duplicated(subset=data_cols, keep="first")
    first_line = df.groupby(data_cols, dropna=False)["source_line"].transform("min")   # line of the row that is kept
    removed_dup = df[dup].copy()
    removed_dup.insert(1, "reason", "exact duplicate of line " + first_line[dup].astype(str))
    df = df[~dup]
    stats["duplicates"] = len(removed_dup)

    # 3. drop rows with a missing weight or price
    missing = df[DROP_IF_MISSING].isna()
    drop = missing.any(axis=1)
    reasons = ["missing " + " and ".join(c for c in DROP_IF_MISSING if missing.at[i, c]) for i in df.index[drop]]
    removed_missing = df[drop].copy()
    removed_missing.insert(1, "reason", reasons)
    df = df[~drop]
    stats["dropped_missing"] = len(removed_missing)
    stats["dropped_by_column"] = {c: int(missing[c].sum()) for c in DROP_IF_MISSING}

    # 4. fill the remaining gaps
    stats["filled"] = {c: int(df[c].isna().sum()) for c in FILL_MISSING}
    for col, value in FILL_MISSING.items():
        df[col] = df[col].fillna(value)

    parts = [p for p in (removed_dup, removed_missing) if len(p)]
    removed = pd.concat(parts).sort_values("source_line") if parts else removed_dup
    return df.drop(columns="source_line").reset_index(drop=True), removed, stats


def near_duplicates(df):
    """Rows that survive but share company + name + weight. Informational only - nothing is removed."""
    key = [c for c in NEAR_DUPLICATE_KEY if c in df.columns]
    if len(key) < 2:
        return []
    return [g for _, g in df.groupby(key, dropna=False) if len(g) > 1]


def main():
    ap = argparse.ArgumentParser(description="Clean the combined tea CSV.")
    ap.add_argument("input", nargs="?", default="combined_output.csv", help="CSV to clean (default: combined_output.csv)")
    ap.add_argument("-o", "--output", help="clean CSV to write (default: <input>_clean.csv)")
    ap.add_argument("--extracted-at", help='date and time of the extraction, e.g. "2026-10-18 09:30" '
                                           "(default: last-modified time of the input file)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"File not found: {args.input}")
    stem = os.path.splitext(args.input)[0]
    out_path = args.output or stem + "_clean.csv"
    removed_path = os.path.splitext(out_path)[0].removesuffix("_clean") + "_removed.csv"

    df = load(args.input).drop(columns=EXTRACTED_COLUMN, errors="ignore")
    stamp = parse_stamp(args.extracted_at) if args.extracted_at else \
        datetime.fromtimestamp(os.path.getmtime(args.input)).strftime(EXTRACTED_FORMAT)
    needed = set(DROP_IF_MISSING) | set(FILL_MISSING) | {"market", "company"}
    if needed - set(df.columns):
        sys.exit(f"Missing column(s) {sorted(needed - set(df.columns))}; the file has {list(df.columns)}")

    clean, removed, st = wrangle(df)
    clean, removed = add_extracted_at(clean, stamp), add_extracted_at(removed, stamp)
    clean.to_csv(out_path, index=False, encoding="utf-8-sig")           # utf-8-sig: opens correctly in Excel
    removed.to_csv(removed_path, index=False, encoding="utf-8-sig")

    print(f"Read {st['rows_in']} rows from {args.input}")
    mk = st["market"]
    print(f"0b.  market set from the company's country   : {mk['rows_changed']} rows changed")
    for (company, old, new), n in sorted(mk["changes"].items()):
        print(f"       {company}: {old or '(empty)'} -> {new}  ({n} rows)")
    if mk["unknown_companies"]:
        print(f"       not in COMPANY_COUNTRY, market left as it was: {', '.join(mk['unknown_companies'])}")
    print(f"1-2. exact duplicate rows found and removed : {st['duplicates']}")
    by = ", ".join(f"{c}: {n}" for c, n in st["dropped_by_column"].items())
    print(f"3.   rows dropped for a missing value        : {st['dropped_missing']}   (empty cells - {by})")
    for c, n in st["filled"].items():
        print(f"4.   {c} filled with '{FILL_MISSING[c]}' : {n}")
    print(f"4b.  {EXTRACTED_COLUMN} added before market        : {stamp}")
    print(f"Result: {len(clean)} rows -> {out_path}")
    print(f"Dropped rows with the reason: {removed_path}")

    if "company" in clean.columns:
        before = df["company"].value_counts(dropna=False)
        after = clean["company"].value_counts(dropna=False)
        print("\nRows per company (before -> after):")
        for company, n in before.items():
            print(f"  {str(company):22} {n:4} -> {int(after.get(company, 0)):4}")

    left = [c for c in clean.columns if c not in FILL_MISSING and clean[c].isna().any()]
    if left:
        print("\nStill empty (not covered by your rules): " + ", ".join(f"{c} ({int(clean[c].isna().sum())})" for c in left))

    groups = near_duplicates(clean)
    if groups:
        print(f"\nNot removed - {len(groups)} products that share {'+'.join(NEAR_DUPLICATE_KEY)} but differ in another column:")
        for g in groups[:10]:
            print(g.drop(columns=["market"], errors="ignore").to_string(index=False, header=False))
            print()


if __name__ == "__main__":
    main()