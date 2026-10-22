"""
model_training.py - predict the price per kilo (and the box price) of a tea.
Updated with SVM Classifier (Stage 1) and Stacking Regressor (Stage 2 / Ensembles).

Workflow:
  1. Split the data ONCE: 80% "train" and 20% "test".
  2. Run 5-fold cross-validation on the 80% train part to compare models.
  3. Fit the chosen model on the whole 80% and score it ONCE on the 20% test part.
  4. Refit the chosen model on ALL rows and save it for production.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_GLOB = os.path.join(HERE, "..", "database", "tea_price_list_*.csv")

CATEGORICAL = ["tea_type", "country", "company"]
NUMERIC = [
    
    "is_bio", "is_grand_cru", "is_imperial", "is_premium",
    "is_uji", "is_ceylon", "is_darjeeling", "is_sencha", "is_smoky",
    "is_gift_set", "n_origins", "origin_known",
]
FEATURES = CATEGORICAL + NUMERIC + ["name"]
WEIGHT_COLS = ["weight_in_g", "log_weight"]
TIERS = ["Discount", "Normal", "Premium", "Luxury"]
TIER_QUANTILES = (0.25, 0.75, 0.94)
TARGET = "price_per_kilo"

TEST_FRACTION = 0.20
CV_FOLDS = 5
STRATIFY_EXTREME_ABOVE = 1000
N_BOOTSTRAP = 1000


# ----------------------------------------------------------------------
# Cleaning and features
# ----------------------------------------------------------------------
def parse_weight(w_str):
    w = str(w_str).lower().strip().replace(",", ".")
    match = re.search(r"(\d+(?:\.\d+)?)", w)
    if not match:
        return np.nan
    val = float(match.group(1))
    return val * 1000 if "kg" in w else val


def parse_price(p_str):
    p = re.sub(r"[^\d,\.]", "", str(p_str)).replace(",", ".")
    try:
        return float(p)
    except ValueError:
        return np.nan


def clean_data(df_raw):
    df = df_raw.copy()
    df["price_clean"] = df["price"].apply(parse_price)
    df["weight_in_g"] = df["weight"].apply(parse_weight)
    df = df[(df["price_clean"] > 0) & (df["weight_in_g"] > 0)].copy()
    df[TARGET] = df["price_clean"] / (df["weight_in_g"] / 1000)
    df["log_weight"] = np.log(df["weight_in_g"])

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
# Models (Updated with Stacking & SVM)
# ----------------------------------------------------------------------
def build_model(kind="stacking"):
    """
    kind: 'stacking' (Stacking Regressor combining RF & GB with a Ridge meta-model),
          'ensemble' (Voting Regressor), 'rf', 'gb',
          'two_stage' (SVM Tier Classifier -> Stacking/GB Price Regressor),
          'two_stage_strict' (two stage without weight).
    """
    if kind == "two_stage":
        return TwoStageModel()
    if kind == "two_stage_strict":
        return TwoStageModel(stage1_company_weight=False, stage2_weight=False, use_proba=False)

    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL), ("num", "passthrough", NUMERIC)]
    )
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)
    
    if kind == "stacking":
        # Stacking Regressor: base models feed into a Ridge meta-estimator
        estimator = StackingRegressor(
            estimators=[("rf", rf), ("gb", gb)],
            final_estimator=Ridge(alpha=1.0)
        )
    else:
        estimator = {"ensemble": VotingRegressor([("rf", rf), ("gb", gb)]), "rf": rf, "gb": gb}[kind]

    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", prep), ("model", estimator)]), func=np.log, inverse_func=np.exp
    )


class TwoStageModel(BaseEstimator, RegressorMixin):
    """
    Stage 1: SVM Classifier with TF-IDF and categorical features to predict price tiers.
    Stage 2: Stacking/Gradient Boosting Regressor incorporating tier predictions.
    """
    def __init__(self, stage1_company_weight=True, stage2_weight=True, use_proba=True,
                 tier_quantiles=TIER_QUANTILES, inner_folds=5, seed=0):
        self.stage1_company_weight = stage1_company_weight
        self.stage2_weight = stage2_weight
        self.use_proba = use_proba
        self.tier_quantiles = tier_quantiles
        self.inner_folds = inner_folds
        self.seed = seed

    def _new_stage1(self):
        cats = ["tea_type", "country"] + (["company"] if self.stage1_company_weight else [])
        parts = [
            ("words", TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1, 2), sublinear_tf=True), "name"),
            ("chars", TfidfVectorizer(strip_accents="unicode", lowercase=True, analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True), "name"),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ]
        # REMOVED WEIGHT_COLS SCALING HERE COMPLETELY
        return Pipeline([("features", ColumnTransformer(parts)), ("clf", SVC(probability=True, C=5.0, kernel="rbf", random_state=42))])

    def _new_stage2(self):
        cats = ["tier", "company", "tea_type", "country"]
        # Only keep tier probabilities, strip out weight entirely
        nums = ["p_" + t for t in TIERS] if self.use_proba else []
        prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats), ("num", "passthrough", nums)])
        
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)
        stacker = StackingRegressor(estimators=[("rf", rf), ("gb", gb)], final_estimator=Ridge(alpha=1.0))
        
        return Pipeline([("prep", prep), ("model", stacker)])

    @staticmethod
    def _tidy(X):
        X = X.reset_index(drop=True).copy()
        X["name"] = X["name"].fillna("").astype(str)
        return X

    @staticmethod
    def _probabilities(classifier, X):
        P = np.zeros((len(X), len(TIERS)))
        classes = classifier.named_steps["clf"].classes_
        probs = classifier.predict_proba(X)
        for idx, cls_val in enumerate(classes):
            if cls_val < len(TIERS):
                P[:, cls_val] = probs[:, idx]
        return P

    @staticmethod
    def _with_tier(X, P):
        X = X.copy()
        X["tier"] = np.array(TIERS)[P.argmax(axis=1)]
        for i, t in enumerate(TIERS):
            X["p_" + t] = P[:, i]
        return X

    def fit(self, X, y):
        X, y = self._tidy(X), np.asarray(y, dtype=float)
        self.tier_edges_ = np.quantile(y, self.tier_quantiles)
        tier = np.digitize(y, self.tier_edges_)
        
        oof = np.zeros((len(X), len(TIERS)))
        folds = StratifiedGroupKFold(n_splits=self.inner_folds, shuffle=True, random_state=self.seed).split(
            X, price_bands(y), groups=product_groups(X))
        for a, b in folds:
            oof[b] = self._probabilities(self._new_stage1().fit(X.iloc[a], tier[a]), X.iloc[b])
            
        self.stage1_ = self._new_stage1().fit(X, tier)
        self.stage2_ = self._new_stage2().fit(self._with_tier(X, oof), np.log(y))
        return self

    def predict_tier(self, X):
        X = self._tidy(X)
        P = self._probabilities(self.stage1_, X)
        out = pd.DataFrame(P, columns=["p_" + t for t in TIERS])
        out.insert(0, "tier", np.array(TIERS)[P.argmax(axis=1)])
        return out

    def predict(self, X):
        X = self._tidy(X)
        P = self._probabilities(self.stage1_, X)
        return np.exp(self.stage2_.predict(self._with_tier(X, P)))

    def known_categories(self):
        enc = self.stage2_.named_steps["prep"].named_transformers_["cat"]
        known = dict(zip(["tier", "company", "tea_type", "country"], enc.categories_))
        return {k: known[k] for k in CATEGORICAL}


CANDIDATES = {
    "Stacking Regressor (RF + GB -> Ridge)": "stacking",
    "Random Forest + Gradient Boosting": "ensemble",
    "Random Forest": "rf",
    "Gradient Boosting": "gb",
    "Two-stage (SVM tier -> Stacking price)": "two_stage",
    "Two-stage strict (no weight)": "two_stage_strict",
}
SELECT_BY = "R2 log"


# ----------------------------------------------------------------------
# Splitting & Metrics (Kept fully compatible)
# ----------------------------------------------------------------------
def product_groups(df):
    key = df["name"].fillna("").str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    return (df["company"] + "|" + key).values


def price_bands(y):
    y = pd.Series(y)
    band = pd.qcut(y.where(y <= STRATIFY_EXTREME_ABOVE), 4, labels=False)
    return band.fillna(4).astype(int).values


def stratified_group_folds(df, n_splits, seed):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return splitter.split(df, price_bands(df[TARGET].values), groups=product_groups(df))


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
    overall = float(np.median(train[TARGET]))
    by_company = train.groupby("company")[TARGET].median()
    return {
        "baseline: shop's median": test["company"].map(by_company).fillna(overall).values,
        "baseline: overall median": np.full(len(test), overall),
    }


def tier_scores(model, df):
    true = np.digitize(df[TARGET].values, model.tier_edges_)
    pred = model.predict_tier(df[FEATURES])[["p_" + t for t in TIERS]].values.argmax(axis=1)
    return float(np.mean(true == pred)), float(np.mean(np.abs(true - pred) <= 1)), true, pred


def fmt(name, value):
    if name.endswith("% error"):
        return f"{value:6.1%}"
    if name in ("MAE", "RMSE"):
        return f"{value:6.0f}"
    return f"{value:6.2f}"


SHOW = ["R2 log", "R2 EUR/kg", "MAE", "RMSE", "median % error", "mean % error", "RMSLE"]
HEAD = ["R2 log", "R2 eur", "MAE", "RMSE", "med %err", "mean %err", "RMSLE"]


def print_table(rows, with_sd=False):
    print(f"    {'':42s}" + "".join(f"{h:>{16 if with_sd else 10}s}" for h in HEAD))
    for label, r in rows.items():
        cells = []
        for k in SHOW:
            if with_sd:
                vals = np.array([m[k] for m in r])
                cells.append(f"{fmt(k, vals.mean())} ±{fmt(k, vals.std()).strip():>5s}")
                cells[-1] = f"{cells[-1]:>16s}"
            else:
                cells.append(f"{fmt(k, r[k]):>10s}")
        print(f"    {label:42s}" + "".join(cells))


def top_drivers(model):
    fitted = model.stage2_ if isinstance(model, TwoStageModel) else model.regressor_
    names = [re.sub(r"^(cat|num)__", "", n) for n in fitted.named_steps["prep"].get_feature_names_out()]
    est = fitted.named_steps["model"]
    if hasattr(est, "named_estimators_"):
        # For Stacking or Voting regressors, extract feature importances from underlying tree models if available
        if hasattr(est, "estimators_"):
            rf_est = next((e for name, e in est.named_estimators_.items() if "rf" in name.lower()), list(est.named_estimators_.values())[0])
            imp = pd.Series(rf_est.feature_importances_, index=names)
        else:
            imp = pd.Series(est.named_estimators_["rf"].feature_importances_, index=names)
    elif hasattr(est, "feature_importances_"):
        imp = pd.Series(est.feature_importances_, index=names)
    else:
        imp = pd.Series(np.ones(len(names)) / len(names), index=names)
        
    columns = CATEGORICAL + ["tier"]
    def column_of(i):
        if i.startswith("p_"):
            return "tier probabilities"
        return next((c for c in columns if i.startswith(c + "_")), i)
    return imp.groupby(column_of).sum()


def print_summary(best_name, how_chosen, cv, cv_base, test_m, base_test, train_m, pct, box_mae, boot, n_test,
                  tier_info, cv_tier):
    r2_cv = np.array([m["R2 log"] for m in cv[best_name]])
    base_mae = base_test["baseline (test): shop's median"]["MAE"]
    lines = [
        f"Model saved: {best_name} ({how_chosen}).",
        f"Accuracy on the {n_test} test products: predicted price per kilo off by median {np.median(pct):.0%}.",
        f"   In euros: average miss of about EUR {test_m['MAE']:.0f}/kg, i.e. about EUR {box_mae:.1f} on one box.",
        f"Versus shop's median baseline: error is {base_mae / test_m['MAE']:.1f}x smaller.",
        f"R2 log = {test_m['R2 log']:.2f} (Bootstrap 95% range: {boot['R2 log'][0]:.2f} - {boot['R2 log'][1]:.2f}).",
    ]
    print("\n" + "=" * 100 + "\nSUMMARY\n" + "=" * 100)
    for ln in lines:
        print(("* " if not ln.startswith("   ") else "") + ln)
    print("=" * 100)


def train_and_export(csv_path, seed=42, force_model="auto"):
    print(f"Loading {csv_path}")
    df = clean_data(pd.read_csv(csv_path))
    y_all = df[TARGET].values
    print(f"{len(df)} products | companies: {df['company'].value_counts().to_dict()}")

    n_splits_for_test = round(1 / TEST_FRACTION)
    train_idx, test_idx = next(iter(stratified_group_folds(df, n_splits_for_test, seed)))
    train, test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    
    print(f"\n1) Split (seed {seed}): {len(train)} train / {len(test)} test")
    
    print(f"\n2) {CV_FOLDS}-fold cross-validation on train part:")
    cv = {name: [] for name in CANDIDATES}
    cv_base = {"baseline: shop's median": [], "baseline: overall median": []}
    cv_tier = {name: [] for name, kind in CANDIDATES.items() if kind.startswith("two_stage")}
    
    for fit_idx, val_idx in stratified_group_folds(train, CV_FOLDS, seed):
        fit, val = train.iloc[fit_idx], train.iloc[val_idx]
        for name, kind in CANDIDATES.items():
            m = build_model(kind).fit(fit[FEATURES], fit[TARGET])
            cv[name].append(metrics(val[TARGET], m.predict(val[FEATURES])))
            if name in cv_tier:
                cv_tier[name].append(tier_scores(m, val)[:2])
        for name, pred in baseline_predictions(fit, val).items():
            cv_base[name].append(metrics(val[TARGET], pred))
            
    print_table({**cv, **cv_base}, with_sd=True)
    auto_best = max(cv, key=lambda n: np.mean([m[SELECT_BY] for m in cv[n]]))
    
    if force_model == "auto":
        best, how = auto_best, f"chosen automatically: best mean {SELECT_BY}"
    else:
        best = next(n for n, k in CANDIDATES.items() if k == force_model)
        how = f"forced with --model {force_model}"

    print(f"   -> {best} ({how})")

    print(f"\n3) Final test: '{best}' scored on {len(test)} test rows")
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
    box_pred = pred_test * test["weight_in_g"].values / 1000
    box_mae = float(np.mean(np.abs(box_pred - test["price_clean"].values)))

    tier_info = None
    if isinstance(model, TwoStageModel):
        acc, within1, true, pred = tier_scores(model, test)
        majority = float(np.mean(true == np.bincount(np.digitize(train[TARGET], model.tier_edges_)).argmax()))
        tier_info = {"acc": acc, "within1": within1, "majority": majority}

    rng = np.random.RandomState(0)
    boot = {k: [] for k in ("R2 log", "MAE", "median % error")}
    for _ in range(N_BOOTSTRAP):
        i = rng.randint(0, len(test), len(test))
        m = metrics(test[TARGET].values[i], pred_test[i])
        for k in boot:
            boot[k].append(m[k])
    boot_range = {k: tuple(np.percentile(v, [2.5, 97.5])) for k, v in boot.items()}

    print(f"\n4) Refitting '{best}' on all {len(df)} rows for production")
    final = build_model(CANDIDATES[best]).fit(df[FEATURES], df[TARGET])
    joblib.dump(final, os.path.join(HERE, "tea_price_model.pkl"))
    with open(os.path.join(HERE, "model_features.json"), "w", encoding="utf-8") as f:
        json.dump({
            "input_columns": FEATURES, "categorical": CATEGORICAL, "numeric": NUMERIC,
            "output": "EUR per kilo", "model": best,
            "test_score": {k: round(float(v), 4) for k, v in result["model: TEST rows"].items()},
        }, f, indent=4, ensure_ascii=False)
    print("Saved tea_price_model.pkl and model_features.json")

    grouped = top_drivers(final)
    top = grouped.sort_values().tail(15)
    plt.figure(figsize=(10, 7))
    sns.barplot(x=top.values, y=top.index, color="#3b7a78")
    plt.title("Price drivers (feature importance)")
    plt.xlabel("Relative importance")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "tea_price_drivers.png"), dpi=300)
    plt.close()

    two_stage_cv_tier = None
    if best in cv_tier and cv_tier[best]:
        a = np.array(cv_tier[best])
        two_stage_cv_tier = {"acc": a[:, 0].mean(), "within1": a[:, 1].mean()}
        
    print_summary(best, how, cv, cv_base, result["model: TEST rows"], result, result["model: train rows"], pct, box_mae,
                  boot_range, len(test), tier_info, two_stage_cv_tier)


def cli():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*least populated class.*")
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="tea_price_list csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="auto", choices=["auto"] + sorted(set(CANDIDATES.values())))
    args = ap.parse_args()
    path = args.csv
    if not path:
        found = sorted(glob.glob(DEFAULT_CSV_GLOB), key=os.path.getmtime)
        if not found:
            raise SystemExit(f"No file found for {DEFAULT_CSV_GLOB}")
        path = found[-1]
    train_and_export(path, args.seed, args.model)


if __name__ == "__main__":
    import importlib
    import sys
    sys.path.insert(0, HERE)
    importlib.import_module(os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]).cli()