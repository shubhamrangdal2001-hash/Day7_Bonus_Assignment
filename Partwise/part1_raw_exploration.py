"""
Part 1 - Raw Data Exploration (Lists + NumPy Only)
Titanic Dataset Analysis
"""

import csv
import numpy as np

file_path = "train.csv"

rows = []
with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        rows.append(row)

print("Headers:", header)
print("Total rows:", len(rows))

# ── Task 1 & 2: Age list with missing values removed ──────────────────────────
age_index = header.index("Age")
age_list = []
for r in rows:
    if r[age_index] != '':
        age_list.append(float(r[age_index]))

print(f"\nTotal passengers with age data: {len(age_list)}")
print(f"Missing age entries removed: {len(rows) - len(age_list)}")

# ── Task 3: Mean, Median, Std Dev ─────────────────────────────────────────────
age_array = np.array(age_list)

mean_age   = np.mean(age_array)
median_age = np.median(age_array)
std_age    = np.std(age_array)

print(f"\nAge Statistics:")
print(f"  Mean Age   : {mean_age:.2f}")
print(f"  Median Age : {median_age:.2f}")
print(f"  Std Dev    : {std_age:.2f}")

# ── Task 4: Fare as NumPy array ───────────────────────────────────────────────
fare_index = header.index("Fare")
fare_list = []
for r in rows:
    if r[fare_index] != '':
        fare_list.append(float(r[fare_index]))

fare_array = np.array(fare_list)
print(f"\nFare Array (first 10): {fare_array[:10]}")

# ── Task 5: Top 10% and Bottom 10% fare passengers ────────────────────────────
top_10_threshold    = np.percentile(fare_array, 90)
bottom_10_threshold = np.percentile(fare_array, 10)

top_10_fares    = fare_array[fare_array >= top_10_threshold]
bottom_10_fares = fare_array[fare_array <= bottom_10_threshold]

print(f"\nFare Percentiles:")
print(f"  Top 10% threshold    : {top_10_threshold:.2f}")
print(f"  Bottom 10% threshold : {bottom_10_threshold:.2f}")
print(f"  Passengers in top 10%    : {len(top_10_fares)}")
print(f"  Passengers in bottom 10% : {len(bottom_10_fares)}")

# ── Task 6: Survival rate by age group ───────────────────────────────────────
survived_index = header.index("Survived")

ages_survived = []
for r in rows:
    if r[age_index] != '':
        ages_survived.append((float(r[age_index]), int(r[survived_index])))

ages_survived = np.array(ages_survived)  # shape: (n, 2) => [age, survived]

age_col      = ages_survived[:, 0]
survived_col = ages_survived[:, 1]

mask_child  = age_col < 15
mask_adult  = (age_col >= 15) & (age_col <= 60)
mask_senior = age_col > 60

def survival_rate(mask):
    group = survived_col[mask]
    return np.mean(group) * 100 if len(group) > 0 else 0

sr_child  = survival_rate(mask_child)
sr_adult  = survival_rate(mask_adult)
sr_senior = survival_rate(mask_senior)

print(f"\nSurvival Rates by Age Group:")
print(f"  Children (< 15)   : {sr_child:.1f}%  (n={mask_child.sum()})")
print(f"  Adults  (15-60)   : {sr_adult:.1f}%  (n={mask_adult.sum()})")
print(f"  Seniors (> 60)    : {sr_senior:.1f}%  (n={mask_senior.sum()})")

# ── Insight: Is age linearly related to survival? ────────────────────────────
correlation = np.corrcoef(age_col, survived_col)[0, 1]
print(f"\nPearson Correlation (Age vs Survival): {correlation:.4f}")
print("Interpretation: Weak/no linear relationship between age and survival.")
print("Age group analysis shows non-linear patterns (children had higher survival).")
