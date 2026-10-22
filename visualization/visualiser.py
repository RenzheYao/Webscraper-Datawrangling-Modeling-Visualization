import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

# -------------------------------------------------------------
# 1. Daten einladen & bereinigen
# -------------------------------------------------------------
df = pd.read_csv('../database/tea_price_list_19102026.csv')

df['price_numeric'] = (
    df['price']
    .astype(str)
    .str.replace('€', '')
    .str.replace(',', '.')
    .astype(float)
)
df['weight_numeric'] = df['weight'].astype(str).str.extract(r'(\d+)').astype(float)
df['price_per_kg'] = df['price_numeric'] / (df['weight_numeric'] / 1000.0)

sns.set_theme(style='whitegrid')

# -------------------------------------------------------------
# 2. Alle Visualisierungs-Charts generieren
# -------------------------------------------------------------

# Chart 1: Distribution of Tea Types
plt.figure(figsize=(10, 6))
tea_counts = df['tea_type'].value_counts()
sns.barplot(x=tea_counts.values, y=tea_counts.index, palette='viridis')
plt.title(
    'Distribution of Tea Types in the French Market',
    fontsize=14,
    fontweight='bold',
)
plt.xlabel('Number of Products', fontsize=12)
plt.ylabel('Tea Type', fontsize=12)
plt.tight_layout()
plt.savefig('tea_type_distribution.png', dpi=300)
plt.close()

# Chart 2: Average Price per Kilogram by Tea Type
plt.figure(figsize=(10, 6))
avg_price_kg = (
    df.groupby('tea_type')['price_per_kg'].mean().sort_values(ascending=False)
)
sns.barplot(x=avg_price_kg.values, y=avg_price_kg.index, palette='magma')
plt.title(
    'Average Price per Kilogram by Tea Type (€/kg)',
    fontsize=14,
    fontweight='bold',
)
plt.xlabel('Average Price (€/kg)', fontsize=12)
plt.ylabel('Tea Type', fontsize=12)
plt.tight_layout()
plt.savefig('avg_price_per_kg_by_tea_type.png', dpi=300)
plt.close()


def autolabel(ax, rects):
  for rect in rects:
    height = rect.get_height()
    ax.annotate(
        f'€{height:.1f}',
        xy=(rect.get_x() + rect.get_width() / 2, height),
        xytext=(0, 3),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
    )


# Chart 3: Black Tea Price Comparison (China vs India)
black_tea = df[df['tea_type'] == 'Black tea']
filtered_black = black_tea[
    black_tea['country_of_origin'].isin(['China', 'India'])
]
stats_black_kg = (
    filtered_black.groupby('country_of_origin')['price_per_kg']
    .agg(['mean', 'min', 'max'])
    .T
)

plt.figure(figsize=(9, 6))
x = np.arange(len(stats_black_kg.index))
width = 0.35
ax = plt.subplot()
rects1 = ax.bar(
    x - width / 2, stats_black_kg['China'], width, label='China', color='#e74c3c'
)
rects2 = ax.bar(
    x + width / 2, stats_black_kg['India'], width, label='India', color='#3498db'
)
ax.set_ylabel('Price (€/kg)', fontsize=12, fontweight='bold')
ax.set_title(
    'Black Tea Price per kg Comparison: China vs India',
    fontsize=14,
    fontweight='bold',
)
ax.set_xticks(x)
ax.set_xticklabels(
    ['Average Price', 'Minimum Price', 'Maximum Price'], fontsize=11
)
ax.legend(fontsize=11)
autolabel(ax, rects1)
autolabel(ax, rects2)
plt.tight_layout()
plt.savefig('black_tea_china_india_price_per_kg.png', dpi=300)
plt.close()

# Chart 4: Green Tea Price Comparison (China vs Japan)
green_tea = df[df['tea_type'] == 'Green tea']
filtered_green = green_tea[
    green_tea['country_of_origin'].isin(['China', 'Japan'])
]
stats_green_kg = (
    filtered_green.groupby('country_of_origin')['price_per_kg']
    .agg(['mean', 'min', 'max'])
    .T
)

plt.figure(figsize=(9, 6))
x_g = np.arange(len(stats_green_kg.index))
ax_g = plt.subplot()
rects_g1 = ax_g.bar(
    x_g - width / 2,
    stats_green_kg['China'],
    width,
    label='China',
    color='#2ecc71',
)
rects_g2 = ax_g.bar(
    x_g + width / 2,
    stats_green_kg['Japan'],
    width,
    label='Japan',
    color='#e67e22',
)
ax_g.set_ylabel('Price (€/kg)', fontsize=12, fontweight='bold')
ax_g.set_title(
    'Green Tea Price per kg Comparison: China vs Japan',
    fontsize=14,
    fontweight='bold',
)
ax_g.set_xticks(x_g)
ax_g.set_xticklabels(
    ['Average Price', 'Minimum Price', 'Maximum Price'], fontsize=11
)
ax_g.legend(fontsize=11)
autolabel(ax_g, rects_g1)
autolabel(ax_g, rects_g2)
plt.tight_layout()
plt.savefig('green_tea_china_japan_price_per_kg.png', dpi=300)
plt.close()

# Chart 5: Black Tea Country Proportion Pie Chart
black_tea_df = df[
    (df['tea_type'] == 'Black tea') & (df['country_of_origin'] != 'Unknown')
]
origin_counts = black_tea_df['country_of_origin'].value_counts()
main_counts = origin_counts[origin_counts >= 2]
other_sum = origin_counts[origin_counts < 2].sum()
if other_sum > 0:
  main_counts['Other'] = other_sum

plt.figure(figsize=(9, 7))
plt.pie(
    main_counts.values,
    labels=main_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=sns.color_palette('Set2'),
)
plt.title(
    'Country of Origin Proportion for Black Tea (Excluding Unknown)',
    fontsize=14,
    fontweight='bold',
)
plt.tight_layout()
plt.savefig('black_tea_origin_pie_chart_clean.png', dpi=300)
plt.close()


# Helper function for Annotated Boxplots
def create_annotated_boxplot(subset, title_str, filename, color_hex):
  prices = subset['price_per_kg'].values
  if len(prices) == 0:
    return
  q25, q50, q75 = np.percentile(prices, [25, 50, 75])
  minimum = prices.min()
  maximum = prices.max()

  plt.figure(figsize=(8, 6))
  ax = sns.boxplot(y=prices, color=color_hex, width=0.4)

  labels = [
      f'Max: €{maximum:.2f}',
      f'75% (Q3): €{q75:.2f}',
      f'50% (Median): €{q50:.2f}',
      f'25% (Q1): €{q25:.2f}',
      f'Min: €{minimum:.2f}',
  ]
  values = [maximum, q75, q50, q25, minimum]

  for label, val in zip(labels, values):
    ax.text(
        0.35,
        val,
        f'  {label}',
        va='center',
        ha='left',
        fontsize=11,
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            alpha=0.8,
            edgecolor=color_hex,
        ),
    )

  plt.title(title_str, fontsize=14, fontweight='bold')
  plt.ylabel('Price per kg (€/kg)', fontsize=12)
  plt.tight_layout()
  plt.savefig(filename, dpi=300)
  plt.close()


# Charts 6-9: Category Boxplots
chinese_black = df[
    (df['tea_type'] == 'Black tea') & (df['country_of_origin'] == 'China')
]
chinese_green = df[
    (df['tea_type'] == 'Green tea') & (df['country_of_origin'] == 'China')
]
matcha = df[df['tea_type'] == 'Matcha']
jasmine = df[df['tea_type'] == 'Jasmine tea']
earl_grey = df[df['tea_type'] == 'Earl Grey']

create_annotated_boxplot(
    chinese_black,
    'Box Plot of Price per kg for Chinese Black Tea',
    'chinese_black_tea_boxplot_annotated.png',
    '#e74c3c',
)
create_annotated_boxplot(
    chinese_green,
    'Box Plot of Price per kg for Chinese Green Tea',
    'chinese_green_tea_boxplot.png',
    '#2ecc71',
)
create_annotated_boxplot(
    matcha,
    'Box Plot of Price per kg for Japanese Matcha',
    'japanese_matcha_boxplot.png',
    '#1abc9c',
)
create_annotated_boxplot(
    jasmine,
    'Box Plot of Price per kg for Jasmine Tea',
    'jasmine_tea_boxplot.png',
    '#f1c40f',
)
create_annotated_boxplot(
    earl_grey,
    'Box Plot of Price per kg for Earl Grey',
    'earl_grey_boxplot.png',
    '#9b59b6',
)

print('All visualization charts successfully generated and saved.')

# -------------------------------------------------------------
# 3. Compile Statistical Results for LLM Prompt
# -------------------------------------------------------------


def calc_stats(subset):
  p = subset['price_per_kg']
  return {
      'count': int(len(p)),
      'min': float(p.min()),
      'q25': float(p.quantile(0.25)),
      'median': float(p.median()),
      'q75': float(p.quantile(0.75)),
      'max': float(p.max()),
      'mean': float(p.mean()),
  }


cb_stats = calc_stats(chinese_black)
cg_stats = calc_stats(chinese_green)
ib_stats = calc_stats(
    df[(df['tea_type'] == 'Black tea') & (df['country_of_origin'] == 'India')]
)
jg_stats = calc_stats(
    df[(df['tea_type'] == 'Green tea') & (df['country_of_origin'] == 'Japan')]
)
mat_stats = calc_stats(matcha)

summary_context = f"""
DATA SUMMARY & FINDINGS:
1. Market Volume Distribution:
   - Black Tea: 132 products (Largest market share)
   - Green Tea: 127 products
   - Top origins for Black Tea (excl Unknown): India (25.0%), Japan (14.6%), China/Sri Lanka Blend (14.6%), China (12.5%).

2. Price Benchmark per Kilogram (€/kg):
   - Chinese Black Tea (n={cb_stats['count']}):
     Min: €{cb_stats['min']:.2f}, 25% (Q1): €{cb_stats['q25']:.2f}, Median: €{cb_stats['median']:.2f}, 75% (Q3): €{cb_stats['q75']:.2f}, Max: €{cb_stats['max']:.2f}, Mean: €{cb_stats['mean']:.2f}.
   - Indian Black Tea (n={ib_stats['count']}):
     Min: €{ib_stats['min']:.2f}, Median: €{ib_stats['median']:.2f}, Max: €{ib_stats['max']:.2f}, Mean: €{ib_stats['mean']:.2f}.
   - Chinese Green Tea (n={cg_stats['count']}):
     Min: €{cg_stats['min']:.2f}, 25% (Q1): €{cg_stats['q25']:.2f}, Median: €{cg_stats['median']:.2f}, 75% (Q3): €{cg_stats['q75']:.2f}, Max: €{cg_stats['max']:.2f}, Mean: €{cg_stats['mean']:.2f}.
   - Japanese Green Tea (n={jg_stats['count']}):
     Min: €{jg_stats['min']:.2f}, Median: €{jg_stats['median']:.2f}, Max: €{jg_stats['max']:.2f}, Mean: €{jg_stats['mean']:.2f}.
   - Japanese Matcha (n={mat_stats['count']}):
     Median: €{mat_stats['median']:.2f}, Mean: €{mat_stats['mean']:.2f}.
"""

prompt_for_qwen = f"""
You are an expert market entry strategy consultant.
Based on the following data analysis results of the French tea market, write a comprehensive Market Entry Report.

{summary_context}

YOUR REPORT MUST INCLUDE:
1. Executive Summary & Market Findings
2. Category & Pricing Structure Analysis (€/kg)
3. Competitive Origin Analysis (China vs. India for Black Tea; China vs. Japan for Green Tea)
4. Specific Recommended Entry Pricing Strategies for Chinese Black Tea and Chinese Green Tea entering the French market (specify exact €/kg and recommended retail pack price per 100g for mass/entry, core specialty, and premium tiers).
"""
# -------------------------------------------------------------
# 4. Call Local Ollama Model (Qwen3-4B:latest) with Streaming
# -------------------------------------------------------------
model_name = 'Qwen3-4B:latest'
print(f"\nConnecting to Ollama & generating report with '{model_name}'...")
print("Note: On CPU-only, model loading may take 30–60 seconds before text appears.\n")

url = 'http://localhost:11434/api/generate'
payload = {
    'model': model_name,
    'prompt': prompt_for_qwen,
    'stream': True  # Stream response to keep connection alive
}

try:
    # Increased timeout to 600 seconds to allow model loading on CPU
    response = requests.post(url, json=payload, stream=True, timeout=600)
    if response.status_code == 200:
        print('=' * 60)
        print(f'GENERATED REPORT FROM LOCAL MODEL ({model_name}):')
        print('=' * 60)

        llm_report = ""
        for line in response.iter_lines():
            if line:
                body = json.loads(line.decode('utf-8'))
                token = body.get('response', '')
                print(token, end='', flush=True)
                llm_report += token

        print('\n' + '=' * 60)

        with open('french_market_entry_report.txt', 'w', encoding='utf-8') as f:
            f.write(llm_report)
        print("\n Report successfully saved to 'french_market_entry_report.txt'!")
    else:
        print(f"\n Error: Ollama API returned status code {response.status_code}. Response: {response.text}")
except Exception as e:
    print(f"\n Connection Error:\nDetails: {e}")