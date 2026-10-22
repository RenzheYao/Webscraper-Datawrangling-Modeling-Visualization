"""
evaluate_tea.py - check whether a tea is priced above or below what the market model expects.

Needs, in the same folder: tea_price_model.pkl (made by model_training.py) and model_training.py itself
(the evaluator re-uses its feature code, so training and evaluation can never drift apart).

Compared with the first evaluator:
  * the model now also needs the COMPANY (the shop is a major price driver) - one new argument;
  * the model is a complete pipeline: it takes the raw values, so there is no get_dummies / reindex step and
    no model_features.json to align with;
  * text values are matched to what the model saw in training ignoring case and accents ("black tea" ->
    "Black tea", "Palais des Thés" -> "palais des thes"). A value the model never saw (new shop, new tea
    type, new country) is reported in a warning instead of being silently ignored;
  * the verdict has a tolerance band. On held-out products the model's typical error was about 8%
    (3 out of 4 within 18%, 9 out of 10 within 32%), so a gap of a few percent means nothing.
    +/-20% = "in line with the market" (change TOLERANCE if you want it stricter or looser).
"""
import os
import re
import sys
import unicodedata
import tkinter as tk
from tkinter import messagebox, ttk

import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_training as mt             # FEATURES, CATEGORICAL, clean_data

MODEL_FILE = "tea_price_model.pkl"
TOLERANCE = 0.20                        # +/-20% around expected price = "in line"


def load_model(path=None):
    """Loads the trained pipeline (looked up next to this script, then in current folder)."""
    candidates = [path] if path else [os.path.join(HERE, MODEL_FILE), MODEL_FILE]
    for p in candidates:
        if p and os.path.exists(p):
            return joblib.load(p)
    raise FileNotFoundError(f"{MODEL_FILE} not found. Run model_training.py first.")


def _key(text):
    """'Palais des Thés' / 'palais  des thes' -> 'palais des thes'; '&' = 'and'."""
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower().replace("&", " and ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9, ]", " ", s)).strip()


def _known_values(model):
    """{'tea_type': [...], 'country': [...], 'company': [...]} as seen in training."""
    enc = model.regressor_.named_steps["prep"].named_transformers_["cat"]
    return dict(zip(mt.CATEGORICAL, enc.categories_))


def _match(value, known, label, warnings):
    """Return spelling model knows; warn if model never saw this value."""
    by_key = {_key(k): k for k in known}
    hit = by_key.get(_key(value))
    if hit is not None:
        return hit
    warnings.append(f"{label} '{value}' was not in training data - model ignores it "
                    f"(known: {', '.join(sorted(map(str, known))[:12])}{' ...' if len(known) > 12 else ''})")
    return str(value)


def evaluate_new_tea(name, tea_type, country, company, actual_price, weight_in_g, model=None,
                     tolerance=TOLERANCE, verbose=True):
    """Compare real price per kilo with model's expected price per kilo."""
    if actual_price is None or float(actual_price) <= 0:
        raise ValueError("actual_price must be greater than 0")
    if weight_in_g is None or float(weight_in_g) <= 0:
        raise ValueError("weight_in_g must be greater than 0")
    model = model or load_model()

    known, notes = _known_values(model), []
    tea_type = _match(tea_type or "Unknown", known["tea_type"], "tea_type", notes)
    country = _match(country or "Unknown", known["country"], "country", notes)
    company = _match(company, known["company"], "company", notes)

    row = pd.DataFrame([{
        "company": company, "name": name, "tea_type": tea_type, "country_of_origin": country,
        "weight": f"{float(weight_in_g):g} g", "price": f"€{float(actual_price):.2f}".replace(".", ","),
    }])
    
    # Raw prediction from pipeline
    raw_pred = float(model.predict(mt.clean_data(row)[mt.FEATURES])[0])

    # Convert inputs
    weight_kg = float(weight_in_g) / 1000.0
    actual_box = float(actual_price)
    actual_per_kg = actual_box / weight_kg

    # If model predicted price per kg directly:
    expected_per_kg = raw_pred
    expected_box_price = expected_per_kg * weight_kg

    diff_per_kg = actual_per_kg - expected_per_kg
    ratio = actual_per_kg / expected_per_kg if expected_per_kg > 0 else 1.0
    
    if abs(ratio - 1) <= tolerance:
        verdict = "IN LINE"
    elif ratio < 1:
        verdict = "UNDERVALUED"
    else:
        verdict = "OVERPRICED"
        
    result = {
        "actual_per_kg": actual_per_kg, 
        "expected_per_kg": expected_per_kg, 
        "difference_per_kg": diff_per_kg,
        "difference_pct": (ratio - 1) * 100, 
        "range_per_kg": (expected_per_kg * (1 - tolerance), expected_per_kg * (1 + tolerance)),
        "expected_box_price": expected_box_price, 
        "verdict": verdict, 
        "warnings": notes
    }

    if verbose:
        lo, hi = result["range_per_kg"]
        print("\n==========================================")
        print(" TEA PRICING EVALUATION REPORT")
        print("==========================================")
        print(f"• Product:      {name}  ({company})")
        print(f"• Category:     {tea_type}")
        print(f"• Origin:       {country}")
        print(f"• Weight:       {float(weight_in_g):g} g")
        print(f"• Actual price: €{actual_box:.2f} (€{actual_per_kg:.2f}/kg)")
        print(f"• Expected:     €{expected_per_kg:.2f}/kg (about €{expected_box_price:.2f} for this {float(weight_in_g):g}g box)")
        print(f"• In-line band: €{lo:.2f}–€{hi:.2f}/kg  (+/-{tolerance:.0%})")
        print("------------------------------------------")
        if verdict == "IN LINE":
            print(f"Status: IN LINE with market ({result['difference_pct']:+.0f}%, €{diff_per_kg:+.2f}/kg)")
        elif verdict == "UNDERVALUED":
            print(f"Status: UNDERVALUED by €{abs(diff_per_kg):.2f}/kg ({result['difference_pct']:+.0f}%)")
        else:
            print(f"Status: OVERPRICED by €{diff_per_kg:.2f}/kg ({result['difference_pct']:+.0f}%)")
        for w in notes:
            print(f"[warning] {w}")
        print("==========================================\n")
    return result


class TeaEvaluatorApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("Tea Price Evaluator")
        self.root.geometry("440x350")
        self.root.resizable(False, False)

        frame = ttk.Frame(root, padding=15)
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(frame, text="Tea Pricing Evaluator", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 15))

        fields = [
            ("Product Name:", "Uji Matcha Premium"),
            ("Tea Type:", "matcha"),
            ("Country of Origin:", "Japan"),
            ("Company / Shop:", "Mariage Frères"),
            ("Actual Price (€):", "25.00"),
            ("Weight (grams):", "30")
        ]

        self.entries = {}
        for idx, (label_text, default_val) in enumerate(fields, start=1):
            ttk.Label(frame, text=label_text).grid(row=idx, column=0, sticky=tk.W, pady=5)
            entry = ttk.Entry(frame, width=30)
            entry.grid(row=idx, column=1, sticky=(tk.W, tk.E), pady=5)
            entry.insert(0, default_val)
            self.entries[label_text] = entry

        eval_btn = ttk.Button(frame, text="Evaluate Tea", command=self.on_evaluate)
        eval_btn.grid(row=len(fields) + 1, column=0, columnspan=2, pady=(20, 5), sticky=(tk.W, tk.E))

    def on_evaluate(self):
        try:
            name = self.entries["Product Name:"].get()
            tea_type = self.entries["Tea Type:"].get()
            country = self.entries["Country of Origin:"].get()
            company = self.entries["Company / Shop:"].get()
            actual_price = float(self.entries["Actual Price (€):"].get())
            weight_in_g = float(self.entries["Weight (grams):"].get())

            res = evaluate_new_tea(
                name=name,
                tea_type=tea_type,
                country=country,
                company=company,
                actual_price=actual_price,
                weight_in_g=weight_in_g,
                model=self.model,
                verbose=True
            )

            lo, hi = res["range_per_kg"]
            report = (
                f"Verdict: {res['verdict']}\n\n"
                f"• Actual Price: €{actual_price:.2f} (€{res['actual_per_kg']:.2f}/kg)\n"
                f"• Expected Box Price: €{res['expected_box_price']:.2f} (€{res['expected_per_kg']:.2f}/kg)\n"
                f"• Difference: {res['difference_pct']:+.1f}% (€{res['difference_per_kg']:+.2f}/kg)\n"
                f"• In-line Band: €{lo:.2f} – €{hi:.2f}/kg\n"
            )
            if res["warnings"]:
                report += "\nWarnings:\n" + "\n".join(f"• {w}" for w in res["warnings"])

            messagebox.showinfo("Evaluation Results", report)

        except ValueError as e:
            messagebox.showerror("Input Error", f"Please check your numeric values:\n{e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")


if __name__ == "__main__":
    try:
        model = load_model()
    except FileNotFoundError as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Model Missing", str(e))
        sys.exit(1)

    root = tk.Tk()
    app = TeaEvaluatorApp(root, model)
    root.mainloop()