Local LLM used in the project: Qwen3-4B-Instruct-2507-Q6_K.gguf from Huggingface

1. Automatic process for webscraping
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

2. Human verification and combine data from different dates (Data wrangling)
In Folder Database
run: tea_price_list.py -> combine "combined_output_clean_DDMMYYYY.csv" to "tea_price_list_DDMMYYYY.csv"
"tea_price_list_DDMMYYYY.csv" has column of date when the infos were extracted. It allows to monitor the price change and analyse the trend.

Final output: tea_price_list_DDMMYYYY.csv in Database folder: prepared for data modeling and data visualisation
 
3. Data_modeling
The original model_training.py follows a standard machine-learning evaluation process:
- Split the dataset into 80% training and 20% testing, keeping variants of the same product together.
- Use 5-fold cross-validation on the training set to compare models.
- Evaluate the selected model once on the held-out test set.
- Retrain the selected model on all data for production use.
Models are evaluated using metrics such as \(R^2\), MAE, RMSE, and percentage errors, alongside simple median-price baselines. 

The packaging bias problem
The original model included package weight as an input feature. This created a misleading shortcut: premium matcha is often sold in small tins, so the model could learn that smaller packages imply a higher price per kilogram.
The modified model addressed this by:
- Removing weight_in_g and log_weight from the predictive features.
- Restricting training data to packages weighing 80–130 g.
- Predicting the intrinsic price per kilogram rather than learning packaging-related price differences.
This makes the prediction more focused on the tea itself, although the restricted dataset also limits the model's training coverage. 

The improved version implements a two-stage model: 
Stage 1 uses an SVM classifier to categorize tea as Luxury, Premium, or Normal, while Stage 2 uses an ensemble regressor combining Random Forest and Gradient Boosting. 
The 80–130 g package-weight filter was removed. However, the two-stage model did not outperform standalone Gradient Boosting. Without package weight as a predictive feature and without filtering by package weight, Gradient Boosting achieved an \(R^2\) of 0.85 on the test dataset for price per kilogram (EUR/kg). Therefore, standalone Gradient Boosting without weight as a feature or a package-weight filter is the best-performing model.

4. Data Visualization
Overall Market Overview (Price vs. Volume / Market Share):Average Price per Kg by Tea Type: Highlights that Matcha is by far the most expensive tea category per kilogram (~€1,360/kg), followed by White tea (~€800/kg), whereas Pu-erh, Chai, and Jasmine tea are the least expensive on average.   
Distribution of Tea Types in the French Market: Shows product volume/availability, revealing that Black tea and Green tea dominate the French market by a wide margin (over 120+ products each), while specialty teas like Pu-erh, Fruit tea, and Chai have very low product counts.  
Country-of-Origin Analysis:Country of Origin Proportion for Black Tea: Breaks down source countries, with India accounting for the largest single-origin market share (25.0%), followed by Japan (14.6%), blend origins (China/Sri Lanka at 14.6%), and China alone (12.5%).   
Black Tea Price Comparison (China vs. India): Compares pricing metrics between the two major producers. China has a higher average price (€204.1/kg vs. €182.9/kg) and higher minimum price, though India reaches a higher maximum price peak (€311.0/kg vs. €253.0/kg).   
Green Tea Price Comparison (China vs. Japan): Shows that Japanese Green Tea commands significantly higher prices than Chinese Green Tea across average (€356.8 vs. €192.6/kg) and maximum (€896.7 vs. €622.0/kg) metrics.   
Detailed Price Distributions (Box Plots):Chinese Black Tea: Shows a stable, tight price range with a median of €214.74/kg (ranging from €140.00 to €253.00/kg).   Chinese Green Tea: Features a low median (€173.33/kg) with most products clustered below €215/kg, but contains extreme premium outliers reaching up to €622.00/kg. Jasmine Tea: Clustered closely around a median of €203.00/kg, with a single high outlier at €445.00/kg.   
Earl Grey: Has a low core price range (median of €183.14/kg), but extreme high-end outliers stretching up to €1,495.00/kg.   
Japanese Matcha: Demonstrates a significantly elevated price distribution with a median of €1,166.67/kg and extreme luxury outliers reaching up to €2,825.00/kg. 
A report was automatically generated by the local LLM for market entry strategy  
