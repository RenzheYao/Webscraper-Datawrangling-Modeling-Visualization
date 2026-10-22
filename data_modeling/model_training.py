"""
model_training.py - predict the price per kilo of a boxed tea.

Workflow (the standard hold-out + cross-validation scheme):
  1. Split the data ONCE: 80% "train" and 20% "test". The split keeps all variants of a product together and is
     stratified by price band, so the few extreme products are spread fairly between the two parts.
  2. On the 80% train part run 5-fold cross-validation (each fold: 80% of it to fit, 20% to check). This is where
     models are compared and chosen. The 20% test part is not used here.
  3. Fit the chosen model on the whole 80% and score it ONCE on the 20% test part. No changes after that.
  4. Refit the chosen model on ALL rows and save it - that is the model you use in production.

Metrics: R2 (log and EUR/kg), MAE, RMSE, median and mean % error, RMSLE - each next to two naive baselines
("always predict the shop's median price per kilo" and "always predict the overall median").

Model: price per kilo is learned on a log scale (the range is 36 - 8970 EUR/kg), from company, weight, tea type,
origin and name keywords; one scikit-learn pipeline, so the saved file takes the raw columns and returns EUR/kg.

Usage:
    python model_training.py                       # newest ../database/tea_price_list_*.csv
    python model_training.py path\\to\\tea_price_list_19102026.csv
    python model_training.py --seed 7              # another random 80/20 split (to see how much the test score depends on it)
Output files go next to this script: tea_price_model.pkl, model_features.json, tea_price_drivers.png
"""
import argparse
import glob
import json
import os
import re
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")                                     # draw to a file, no window needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_GLOB = os.path.join(HERE, "..", "database", "tea_price_list_*.csv")

CATEGORICAL = ["tea_type", "country", "company"]
NUMERIC = [
    "weight_in_g", "log_weight",
    "is_bio", "is_grand_cru", "is_imperial", "is_premium",
    "is_uji", "is_ceylon", "is_darjeeling", "is_sencha", "is_smoky",
    "is_gift_set", "n_origins", "origin_known",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "price_per_kilo"

TEST_FRACTION = 0.20            # share of the data kept aside as the final test
CV_FOLDS = 5                    # cross-validation folds on the train part
STRATIFY_EXTREME_ABOVE = 1000   # EUR/kg: products above this form their own band when splitting (only used to balance the split)
N_BOOTSTRAP = 1000              # resamples used for the uncertainty range of the test score


# ----------------------------------------------------------------------
# Cleaning and features
# ----------------------------------------------------------------------
def parse_weight(w_str):
    """Weight in grams from '100 g', '99,2 g', '1 kg'."""
    w = str(w_str).lower().strip().replace(",", ".")
    match = re.search(r"(\d+(?:\.\d+)?)", w)
    if not match:
        return np.nan
    val = float(match.group(1))
    return val * 1000 if "kg" in w else val


def parse_price(p_str):
    """'€19,80' -> 19.8"""
    p = re.sub(r"[^\d,\.]", "", str(p_str)).replace(",", ".")
    try:
        return float(p)
    except ValueError:
        return np.nan


def clean_data(df_raw):
    """Prices and weights as numbers, the target, and the features."""
    df = df_raw.copy()
    df["price_clean"] = df["price"].apply(parse_price)
    df["weight_in_g"] = df["weight"].apply(parse_weight)
    df = df[(df["price_clean"] > 0) & (df["weight_in_g"] > 0)].copy()
    df[TARGET] = df["price_clean"] / (df["weight_in_g"] / 1000)
    df["log_weight"] = np.log(df["weight_in_g"])

    # the CSV calls it country_of_origin (the old code looked for "country" and never found it)
    origin_col = "country_of_origin" if "country_of_origin" in df.columns else "country"
    df["country"] = df[origin_col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    df["origin_known"] = (df["country"] != "Unknown").astype(int)
    df["n_origins"] = df["country"].apply(lambda c: 0 if c == "Unknown" else len([x for x in c.split(",") if x.strip()]))

    df["company"] = df["company"].fillna("Unknown").astype(str).str.lower().str.strip()
    df["tea_type"] = df["tea_type"].fillna("Unknown").astype(str).replace("", "Unknown")

    name = df["name"].fillna("").str.lower()
    df["is_bio"] = name.str.contains("bio|organic", regex=True).astype(int)
    df["is_grand_cru"] = name.str.contains("grand cru", regex=True).astype(int)
    df["is_imperial"] = name.str.contains("impérial|imperial", regex=True).astype(int)
    df["is_premium"] = name.str.contains("premium|special|reserve", regex=True).astype(int)
    df["is_uji"] = name.str.contains("uji", regex=True).astype(int)
    df["is_ceylon"] = name.str.contains("ceylon", regex=True).astype(int)
    df["is_darjeeling"] = name.str.contains("darjeeling", regex=True).astype(int)
    df["is_sencha"] = name.str.contains("sencha", regex=True).astype(int)
    df["is_smoky"] = name.str.contains("smoky|fumé|fume", regex=True).astype(int)
    df["is_gift_set"] = name.str.contains("calendrier|coffret|avent|à tartiner|a tartiner", regex=True).astype(int)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def build_model(kind="ensemble"):
    """Raw columns in -> EUR per kilo out. Learns on log(price per kilo) and converts back.
    kind: 'ensemble' (Random Forest + Gradient Boosting averaged), 'rf' or 'gb'."""
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL), ("num", "passthrough", NUMERIC)]
    )
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)
    estimator = {"ensemble": VotingRegressor([("rf", rf), ("gb", gb)]), "rf": rf, "gb": gb}[kind]
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", prep), ("model", estimator)]), func=np.log, inverse_func=np.exp
    )


CANDIDATES = {                                    # compared by cross-validation on the train part
    "Random Forest + Gradient Boosting": "ensemble",
    "Random Forest": "rf",
    "Gradient Boosting": "gb",
}
SELECT_BY = "R2 log"                              # the candidate with the best mean of this metric is chosen


# ----------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------
def product_groups(df):
    """Same shop + same product name = one group (a product sold as several variants is never split)."""
    key = df["name"].fillna("").str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    return (df["company"] + "|" + key).values


def price_bands(y):
    """0-3 = price quartiles among regular products, 4 = extreme products. Used only to balance the splits."""
    y = pd.Series(y)
    band = pd.qcut(y.where(y <= STRATIFY_EXTREME_ABOVE), 4, labels=False)
    return band.fillna(4).astype(int).values


def stratified_group_folds(df, n_splits, seed):
    """Folds with whole products together and a similar price mix in every fold. Yields (train_idx, test_idx)."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return splitter.split(df, price_bands(df[TARGET].values), groups=product_groups(df))


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), 1e-6, None)
    pct = np.abs(p / y_true - 1)
    return {
        "R2 log": r2_score(np.log(y_true), np.log(p)),
        "R2 EUR/kg": r2_score(y_true, p),
        "MAE": mean_absolute_error(y_true, p),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, p))),
        "median % error": float(np.median(pct)),
        "mean % error": float(np.mean(pct)),
        "RMSLE": float(np.sqrt(mean_squared_error(np.log(y_true), np.log(p)))),
    }


def baseline_predictions(train, test):
    """Two naive predictors, built from the train rows only."""
    overall = float(np.median(train[TARGET]))
    by_company = train.groupby("company")[TARGET].median()
    return {
        "baseline: shop's median": test["company"].map(by_company).fillna(overall).values,
        "baseline: overall median": np.full(len(test), overall),
    }


def fmt(name, value):
    if name.endswith("% error"):
        return f"{value:6.1%}"
    if name in ("MAE", "RMSE"):
        return f"{value:6.0f}"
    return f"{value:6.2f}"


SHOW = ["R2 log", "R2 EUR/kg", "MAE", "RMSE", "median % error", "mean % error", "RMSLE"]
HEAD = ["R2 log", "R2 eur", "MAE", "RMSE", "med %err", "mean %err", "RMSLE"]


def print_table(rows, with_sd=False):
    """rows: {label: [metrics dict per fold]} or {label: metrics dict}"""
    print(f"    {'':36s}" + "".join(f"{h:>{16 if with_sd else 10}s}" for h in HEAD))
    for label, r in rows.items():
        cells = []
        for k in SHOW:
            if with_sd:
                vals = np.array([m[k] for m in r])
                cells.append(f"{fmt(k, vals.mean())} ±{fmt(k, vals.std()).strip():>5s}")
                cells[-1] = f"{cells[-1]:>16s}"
            else:
                cells.append(f"{fmt(k, r[k]):>10s}")
        print(f"    {label:36s}" + "".join(cells))


def predict_price_per_kilo(model, rows):
    """rows: DataFrame with the raw CSV columns (company, name, tea_type, weight, country_of_origin; price is not needed).
    Returns EUR per kilo."""
    rows = rows.copy()
    if "price" not in rows.columns:
        rows["price"] = "€1,00"                     # clean_data needs a price column; its value is not used for prediction
    return model.predict(clean_data(rows)[FEATURES])


def top_drivers(fitted):
    """Feature importance of the fitted pipeline, summed per original column."""
    names = fitted.named_steps["prep"].get_feature_names_out()
    est = fitted.named_steps["model"]
    if hasattr(est, "named_estimators_"):             # the ensemble: use its Random Forest half
        est = est.named_estimators_["rf"]
    imp = pd.Series(est.feature_importances_, index=[re.sub(r"^(cat|num)__", "", n) for n in names])
    return imp.groupby(lambda i: next((c for c in CATEGORICAL if i.startswith(c + "_")), i)).sum()


# ----------------------------------------------------------------------
def train_and_export(csv_path, seed=42):
    print(f"Loading {csv_path}")
    df = clean_data(pd.read_csv(csv_path))
    y_all = df[TARGET].values
    print(f"{len(df)} products | companies: {df['company'].value_counts().to_dict()}")
    sq = (y_all - y_all.mean()) ** 2
    print(f"Note: the 5 most extreme products carry {np.sort(sq)[-5:].sum() / sq.sum():.0%} of the EUR/kg variance "
          f"(max {y_all.max():.0f} EUR/kg, median {np.median(y_all):.0f}).")

    # ---- 1. one 80/20 split: the test part is not touched until step 3 ----
    n_splits_for_test = round(1 / TEST_FRACTION)
    train_idx, test_idx = next(iter(stratified_group_folds(df, n_splits_for_test, seed)))
    train, test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    print(f"\n1) Split (seed {seed}): {len(train)} train / {len(test)} test ({len(test) / len(df):.0%}); "
          f"products above {STRATIFY_EXTREME_ABOVE} EUR/kg: {int((train[TARGET] > STRATIFY_EXTREME_ABOVE).sum())} in train, "
          f"{int((test[TARGET] > STRATIFY_EXTREME_ABOVE).sum())} in test")
    print("   shops in test:", test["company"].value_counts().to_dict())

    # ---- 2. cross-validation on the train part only ----
    print(f"\n2) {CV_FOLDS}-fold cross-validation on the {len(train)} train rows (mean ± sd over the folds):")
    cv = {name: [] for name in CANDIDATES}
    cv_base = {"baseline: shop's median": [], "baseline: overall median": []}
    for fit_idx, val_idx in stratified_group_folds(train, CV_FOLDS, seed):
        fit, val = train.iloc[fit_idx], train.iloc[val_idx]
        for name, kind in CANDIDATES.items():
            m = build_model(kind).fit(fit[FEATURES], fit[TARGET])
            cv[name].append(metrics(val[TARGET], m.predict(val[FEATURES])))
        for name, pred in baseline_predictions(fit, val).items():
            cv_base[name].append(metrics(val[TARGET], pred))
    print_table({**cv, **cv_base}, with_sd=True)
    best = max(cv, key=lambda n: np.mean([m[SELECT_BY] for m in cv[n]]))
    print(f"   -> chosen by the best mean {SELECT_BY}: {best}")

    # ---- 3. the one and only look at the test part ----
    print(f"\n3) Final test: '{best}' fitted on all {len(train)} train rows, scored once on the {len(test)} test rows")
    model = build_model(CANDIDATES[best]).fit(train[FEATURES], train[TARGET])
    pred_test = model.predict(test[FEATURES])
    result = {
        "model: train rows": metrics(train[TARGET], model.predict(train[FEATURES])),
        "model: TEST rows": metrics(test[TARGET], pred_test),
    }
    for name, pred in baseline_predictions(train, test).items():
        result[name.replace("baseline", "baseline (test)")] = metrics(test[TARGET], pred)
    print_table(result)
    pct = np.abs(np.clip(pred_test, 1e-6, None) / test[TARGET].values - 1)
    print(f"   typical error on the test rows: median {np.median(pct):.0%}; 75% of predictions within {np.percentile(pct, 75):.0%}, "
          f"90% within {np.percentile(pct, 90):.0%}   (basis of the +/-20% band in evaluate_tea.py)")

    rng = np.random.RandomState(0)
    boot = {k: [] for k in ("R2 log", "MAE", "median % error")}
    for _ in range(N_BOOTSTRAP):
        i = rng.randint(0, len(test), len(test))
        m = metrics(test[TARGET].values[i], pred_test[i])
        for k in boot:
            boot[k].append(m[k])
    print("   95% range of the test score if the test products had been drawn differently (bootstrap over the test rows):")
    for k, v in boot.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        print(f"      {k:15s} {fmt(k, result['model: TEST rows'][k]).strip():>6s}   [{fmt(k, lo).strip()} - {fmt(k, hi).strip()}]")
    print("   Only 77 test products: treat the test score as one draw, not an exact number "
          f"(try --seed to see another split).")

    # ---- 4. production model: refit on ALL rows ----
    print(f"\n4) Refitting '{best}' on all {len(df)} rows for production (this model was not scored - the test rows are inside it)")
    final = build_model(CANDIDATES[best]).fit(df[FEATURES], df[TARGET])
    joblib.dump(final, os.path.join(HERE, "tea_price_model.pkl"))
    with open(os.path.join(HERE, "model_features.json"), "w", encoding="utf-8") as f:
        json.dump({
            "input_columns": FEATURES, "categorical": CATEGORICAL, "numeric": NUMERIC,
            "output": "EUR per kilo", "model": best,
            "test_score": {k: round(float(v), 4) for k, v in result["model: TEST rows"].items()},
            "how_to_use": "model = joblib.load('tea_price_model.pkl'); model.predict(DataFrame[input_columns]); "
                          "or call predict_price_per_kilo(model, rows) from model_training.py with the raw CSV columns",
        }, f, indent=4, ensure_ascii=False)
    print("Saved tea_price_model.pkl and model_features.json")

    grouped = top_drivers(final.regressor_)
    top = grouped.sort_values().tail(15)
    plt.figure(figsize=(10, 7))
    sns.barplot(x=top.values, y=top.index, color="#3b7a78")
    plt.title("Price drivers (feature importance, summed per column)")
    plt.xlabel("Relative importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "tea_price_drivers.png"), dpi=300)
    plt.close()
    print("Saved tea_price_drivers.png")
    print("Top drivers:", ", ".join(f"{k} {v:.0%}" for k, v in grouped.sort_values(ascending=False).head(5).items()))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*least populated class.*")
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="tea_price_list csv (default: newest ../database/tea_price_list_*.csv)")
    ap.add_argument("--seed", type=int, default=42, help="random seed of the 80/20 split and the CV folds (default 42)")
    args = ap.parse_args()
    path = args.csv
    if not path:
        found = sorted(glob.glob(DEFAULT_CSV_GLOB), key=os.path.getmtime)
        if not found:
            raise SystemExit(f"No file found for {DEFAULT_CSV_GLOB} - pass the csv path as an argument.")
        path = found[-1]
    train_and_export(path, args.seed)