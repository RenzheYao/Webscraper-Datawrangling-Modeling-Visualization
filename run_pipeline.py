import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

# ----------------------------------------------------------------------
# Settings (edit here)
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))              # where the scripts are
DATA_DIR = r"D:\Business\Freelancer\Webscraper+Datawrangling"        # where the scrapers write their JSON files
COMBINED_CSV = "combined_output.csv"
EXTRACTED_FORMAT = "%Y-%m-%d %H:%M"                                  # format passed to wrangle_tea_csv.py

# (label, script, output file the step must produce, or None)
STEPS = [
    ("Palais des Thés",   "scrape_pages_fr1.py",    "raw_results.json"),
    ("Dammann Frères",    "scrape_pages_fr2.py",    "dammann_raw_results.json"),
    ("Kusmi Tea",         "scrape_pages_fr3.py",    "kusmi_raw_results.json"),
    ("Mariage Frères",    "scrape_pages_fr4.py",    "mariage_raw_results.json"),
    ("Betjeman & Barton", "scrape_pages_fr5.py",    "betjeman_raw_results.json"),
    ("Combine JSON -> CSV", "combine_json_to_csv.py", COMBINED_CSV),
    ("Wrangle / clean",   "wrangle_tea_csv.py",     None),
]
N_SCRAPERS = 5          # the first five steps are scrapers; the rest are not allowed to fail with --keep-going


def output_problem(path, started):
    """None if the file is fine, otherwise a sentence saying what is wrong with it."""
    if not os.path.exists(path):
        return f"expected output not found: {path}"
    if os.path.getmtime(path) < started - 1:
        return f"{os.path.basename(path)} was not updated by this run (old file from an earlier run)"
    if os.path.getsize(path) == 0:
        return f"{os.path.basename(path)} is empty"
    if path.lower().endswith(".json"):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (ValueError, OSError) as e:
            return f"{os.path.basename(path)} is not valid JSON ({e})"
        if not data:
            return f"{os.path.basename(path)} contains no products"
    return None


def run_step(number, label, script, output, run_started):
    """Returns (ok, seconds, message)."""
    script_path = os.path.join(SCRIPT_DIR, script)
    if not os.path.exists(script_path):
        return False, 0.0, f"script not found: {script_path}"

    cmd = [sys.executable, "-u", script_path]
    if script == "wrangle_tea_csv.py":
        cmd += [os.path.join(DATA_DIR, COMBINED_CSV), "--extracted-at", run_started.strftime(EXTRACTED_FORMAT)]

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    started = time.time()
    try:
        code = subprocess.call(cmd, cwd=SCRIPT_DIR, env=env)       # output streams live to this console
    except KeyboardInterrupt:
        return False, time.time() - started, "interrupted by you (Ctrl+C)"
    took = time.time() - started
    if code != 0:
        return False, took, f"{script} exited with code {code}"
    if output:
        problem = output_problem(os.path.join(DATA_DIR, output), started)
        if problem:
            return False, took, problem
    return True, took, "ok"


def parse_numbers(values, total):
    bad = [v for v in values if not 1 <= v <= total]
    if bad:
        sys.exit(f"Step number(s) {bad} do not exist: the steps are 1 to {total}.")
    return set(values)


def main():
    ap = argparse.ArgumentParser(description="Run the tea scraping + cleaning pipeline.")
    ap.add_argument("--only", type=int, nargs="+", metavar="N", help="run only these step numbers")
    ap.add_argument("--skip", type=int, nargs="+", metavar="N", help="run everything except these step numbers")
    ap.add_argument("--from-step", type=int, metavar="N", help="start at this step number")
    ap.add_argument("--keep-going", action="store_true", help="if a scraper fails, still run the other scrapers")
    ap.add_argument("--dry-run", action="store_true", help="show the plan and stop")
    args = ap.parse_args()

    total = len(STEPS)
    numbers = list(range(1, total + 1))
    if args.only:
        numbers = [n for n in numbers if n in parse_numbers(args.only, total)]
    if args.from_step:
        parse_numbers([args.from_step], total)
        numbers = [n for n in numbers if n >= args.from_step]
    if args.skip:
        numbers = [n for n in numbers if n not in parse_numbers(args.skip, total)]
    if not numbers:
        sys.exit("Nothing to run.")

    print("Pipeline plan:")
    for n, (label, script, output) in enumerate(STEPS, 1):
        mark = "run " if n in numbers else "skip"
        print(f"  {n}. [{mark}] {script:24} {label}" + (f"  ->  {output}" if output else ""))
    if args.dry_run:
        return

    run_started = datetime.now()
    results = []                                         # (number, label, ok, seconds, message)
    stop = False
    for n in numbers:
        label, script, output = STEPS[n - 1]
        print(f"\n{'=' * 78}\nSTEP {n}/{total}: {label}  ({script})\n{'=' * 78}", flush=True)
        ok, took, msg = run_step(n, label, script, output, run_started)
        results.append((n, label, ok, took, msg))
        if not ok:
            print(f"\n!! STEP {n} FAILED: {msg}", flush=True)
            if not (args.keep_going and n <= N_SCRAPERS):
                stop = True
                break
            print("   --keep-going: continuing with the next scraper", flush=True)

    failed = [r for r in results if not r[2]]
    ran = {r[0] for r in results}
    print(f"\n{'=' * 78}\nSUMMARY  (started {run_started:%d/%m/%Y %H:%M}, total {sum(r[3] for r in results) / 60:.1f} min)\n{'=' * 78}")
    for n, label, ok, took, msg in results:
        print(f"  {n}. {'OK    ' if ok else 'FAILED'}  {label:22} {took:7.1f}s  {'' if ok else msg}")
    for n in numbers:
        if n not in ran:
            print(f"  {n}. not run  {STEPS[n - 1][0]}")

    if failed:
        first = failed[0][0]
        print(f"\nPipeline did NOT finish cleanly. After fixing step {first}, resume with: python run_pipeline.py --from-step {first}")
        sys.exit(1)
    if stop:                                             # cannot happen without a failure, kept as a safeguard
        sys.exit(1)
    if total in numbers:
        print(f"\nFinal cleaned data: {os.path.join(DATA_DIR, os.path.splitext(COMBINED_CSV)[0] + '_clean.csv')}")
    print("Pipeline finished.")


if __name__ == "__main__":
    main()