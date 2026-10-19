"""
1. Automatic process
run_pipeline.py - run the whole tea pipeline, one step after the other:

    scrape_pages_fr1.py  -> raw_results.json            (Palais des Thés)
    scrape_pages_fr2.py  -> dammann_raw_results.json    (Dammann Frères)
    scrape_pages_fr3.py  -> kusmi_raw_results.json      (Kusmi Tea)
    scrape_pages_fr4.py  -> mariage_raw_results.json    (Mariage Frères)
    scrape_pages_fr5.py  -> betjeman_raw_results.json   (Betjeman & Barton)
    combine_json_to_csv.py -> combined_output.csv       (reads the five JSON files)
    wrangle_tea_csv.py   -> combined_output_clean.csv   (+ combined_output_removed.csv)
    archive_to_database.py -> arrange the json and csv files in folder database with _DDMMYYYY in the end

Every script is started with the same Python that runs this file, from the folder where the scripts
are. A scraper counts as successful only if it exits with code 0 AND its JSON file was (re)written
during this run and is not empty - so an old file from a previous run can never be mistaken for fresh data.
The pipeline stops at the first failed step (a half-finished run would merge stale data).

Usage (from the folder that contains the scripts):
    python run_pipeline.py                    # everything
    python run_pipeline.py --only 3 4         # only steps 3 and 4        (numbers as printed in the plan)
    python run_pipeline.py --from-step 6      # restart at step 6, e.g. after a scraper failed and was fixed
    python run_pipeline.py --skip 1 2         # everything except steps 1 and 2
    python run_pipeline.py --keep-going       # do not stop when a scraper fails (the others still run)
    python run_pipeline.py --dry-run          # only show the plan
"""

2. Human verification and combine data from different dates
In Folder Database
run: tea_price_list.py -> combine "combined_output_clean_DDMMYYYY.csv" to "tea_price_list_DDMMYYYY.csv"
"tea_price_list_DDMMYYYY.csv" has column of date when the infos were extracted. It allows to monitor the price change and analyse the trend.

Final output: tea_price_list_DDMMYYYY.csv in Database folder: prepared for data modeling and data visualisation
 
