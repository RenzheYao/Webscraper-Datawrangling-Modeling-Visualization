"""
combine_clean_csvs.py - put every combined_output_clean_DDMMYYYY.csv of the data folder into ONE file,
tea_price_list_DDMMYYYY.csv (DDMMYYYY = today).

* Finds files named combined_output_clean_DDMMYYYY.csv (and the _2, _3 ... copies the pipeline makes when it runs
  twice on one day). Anything else in the folder is ignored - including earlier tea_price_list_*.csv files.
* Rows are stacked oldest file first (by the date in the file name, not the file's modified time).
* Files made before the extracted_at column existed get it from the date in their file name (DD/MM/YYYY 00:00).
* DUPLICATES are removed: rows that are identical in every column EXCEPT extracted_at are kept once, the
  earliest one (so a product whose data did not change since last month appears only once, with the date it
  was first seen). A product whose price, weight, type, ... changed is a different row and stays.
  With --keep-history only rows identical including extracted_at are removed, so an unchanged product
  appears once per extraction.
* All files must have the same columns, otherwise nothing is written and the problem is listed.

Usage:
    pip install pandas
    python combine_clean_csvs.py                          # uses the data folder below
    python combine_clean_csvs.py "D:\\path\\to\\data"       # another folder
    python combine_clean_csvs.py --keep-history           # keep unchanged products once per extraction date
    python combine_clean_csvs.py -o "D:\\path\\to\\tea_price.csv"
"""
import argparse
import os
import re
import sys
from datetime import datetime

import pandas as pd

DATA_FOLDER = r"D:\Business\Freelancer\Webscraper+Datawrangling\database"
INPUT_PATTERN = re.compile(r"^combined_output_clean_(\d{2})(\d{2})(\d{4})(?:_\d+)?\.csv$", re.IGNORECASE)
OUTPUT_PREFIX = "tea_price_list"
DATE_COLUMN = "extracted_at"
DATE_FORMAT = "%d/%m/%Y %H:%M"


def find_inputs(folder):
    """[(date, filename)] oldest first; files whose name carries an impossible date are reported and skipped."""
    found, skipped = [], []
    for name in os.listdir(folder):
        m = INPUT_PATTERN.match(name)
        if not m:
            continue
        d, mo, y = map(int, m.groups())
        try:
            found.append((datetime(y, mo, d), name))
        except ValueError:
            skipped.append(name)
    return sorted(found), skipped


def combine(folder, keep_history=False):
    """Returns (dataframe, info). Raises ValueError with a readable message if the files do not fit together."""
    files, bad_names = find_inputs(folder)
    if not files:
        raise ValueError(f"No combined_output_clean_DDMMYYYY.csv file found in {folder}")
    frames, filled_from_name, columns = [], [], None
    for date, name in files:
        df = pd.read_csv(os.path.join(folder, name), dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if DATE_COLUMN not in df.columns:                      # older file: date from the file name
            df.insert(0, DATE_COLUMN, date.strftime(DATE_FORMAT))
            filled_from_name.append(name)
        elif (df[DATE_COLUMN].str.strip() == "").any():
            df[DATE_COLUMN] = df[DATE_COLUMN].mask(df[DATE_COLUMN].str.strip() == "", date.strftime(DATE_FORMAT))
        if columns is None:
            columns = list(df.columns)
        elif list(df.columns) != columns:
            raise ValueError(f"{name} has different columns than {files[0][1]}:\n  {list(df.columns)}\n  {columns}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    before = len(out)
    if keep_history:
        out = out.drop_duplicates()
    else:                                                      # same data in another month: keep the earliest date
        when = pd.to_datetime(out[DATE_COLUMN], format=DATE_FORMAT, errors="coerce")
        out = out.assign(_when=when).sort_values("_when", kind="stable", na_position="last")
        out = out.drop_duplicates(subset=[c for c in out.columns if c not in (DATE_COLUMN, "_when")]).sort_values(
            "_when", kind="stable", na_position="last").drop(columns="_when")
    out = out.reset_index(drop=True)
    return out, {"files": [n for _, n in files], "rows_in": before, "duplicates": before - len(out),
                 "filled_from_name": filled_from_name, "bad_names": bad_names}


def main():
    ap = argparse.ArgumentParser(description="Combine all combined_output_clean_DDMMYYYY.csv files into tea_price_list_DDMMYYYY.csv.")
    ap.add_argument("folder", nargs="?", default=DATA_FOLDER, help=f"folder with the dated csv files (default: {DATA_FOLDER})")
    ap.add_argument("--keep-history", action="store_true", help="keep an unchanged product once per extraction date")
    ap.add_argument("-o", "--output", help="output file (default: <folder>/tea_price_list_DDMMYYYY.csv, today's date)")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"Folder not found: {args.folder}")
    try:
        df, info = combine(args.folder, args.keep_history)
    except ValueError as e:
        sys.exit(str(e))

    out_path = args.output or os.path.join(args.folder, f"{OUTPUT_PREFIX}_{datetime.now():%d%m%Y}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")        # utf-8-sig: opens correctly in Excel

    print(f"Combined {len(info['files'])} files:")
    for name in info["files"]:
        print(f"  {name}")
    for name in info["bad_names"]:
        print(f"  [skipped] {name}: the date in the name is not a real date")
    if info["filled_from_name"]:
        print(f"{DATE_COLUMN} taken from the file name (these files had no such column): {', '.join(info['filled_from_name'])}")
    print(f"{info['rows_in']} rows read, {info['duplicates']} duplicates removed -> {len(df)} rows")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()