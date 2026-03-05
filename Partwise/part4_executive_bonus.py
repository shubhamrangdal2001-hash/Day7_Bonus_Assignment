"""
Part 4 - Executive Challenge + Bonus Challenge
Titanic Dataset - Final Analysis
"""

import csv
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PART 4 - Executive Challenge
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("PART 4 - Executive Challenge")
print("=" * 60)

# Q1: Who should be prioritized for rescue?
print("""
Q1: Rescue Priority Based on Model

If a Titanic-like disaster happened today, rescue priorities based
on our model would be:
  1. Women (especially from 1st/2nd class) -- highest survival need
  2. Children (age < 15) -- vulnerable group, lower survival odds
     without help
  3. Elderly passengers (> 60) -- lower mobility, need faster rescue
  4. Passengers traveling alone -- no family support on board

Our survival score shows gender and class as the top two factors,
so priority order: Children and Women first (regardless of class),
then elderly passengers, then anyone traveling solo.
""")

# Q2: Ethical concerns
print("""
Q2: Three Ethical Concerns in Automated Survival Prediction

1. Gender & Class Bias: Training on historical data embeds old
   social hierarchies (e.g., "women and children first"). Using
   this for real rescue decisions could unfairly disadvantage
   certain groups.

2. Accountability: An automated system deciding who to rescue
   removes human judgment. In edge cases (e.g., a healthy male
   nurse who could help others), the model could give wrong
   priority.

3. Data Fairness: Lower-class passengers had fewer recorded
   details (missing cabin info, group tickets). The model is
   inherently less accurate for groups with sparse historical
   data, amplifying existing inequality.
""")

# Q3: Insurance underwriting context
print("""
Q3: Changes for Insurance Underwriting

If this were an insurance dataset:
  - "Survival" maps to claim risk -- we'd predict claim probability.
  - FamilySize and travel class would be replaced by health history
    and occupational risk factors.
  - Gender-based pricing is legally restricted in many countries
    (EU Unisex Directive), so gender would need to be dropped or
    handled carefully.
  - The weight of 'Fare' would shift to 'Income/Assets' to measure
    moral hazard and coverage amount.
  - We'd add a regulatory review step before deploying any scoring
    model, and run fairness audits across protected demographics.
""")

# ─────────────────────────────────────────────────────────────────────────────
# BONUS CHALLENGE - NumPy Only, No Loops, No Pandas groupby
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("BONUS - NumPy Only (Vectorized)")
print("=" * 60)

# Load CSV manually
rows = []
with open("train.csv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        rows.append(row)

data = np.array(rows)

# Extract columns as numpy arrays
survived = data[:, header.index('Survived')].astype(int)
pclass   = data[:, header.index('Pclass')].astype(int)
sex      = data[:, header.index('Sex')]

# ── Survival by Class (vectorized, no loops) ─────────────────────────────────
print("\n--- Survival by Class (NumPy Vectorized) ---")
for cls in np.unique(pclass):
    mask = pclass == cls
    rate = np.mean(survived[mask])
    print(f"  Class {cls}: {rate:.4f}  ({rate*100:.1f}%)")

# ── Survival by Gender (vectorized, no loops) ─────────────────────────────────
print("\n--- Survival by Gender (NumPy Vectorized) ---")
for gender in np.unique(sex):
    mask = sex == gender
    rate = np.mean(survived[mask])
    print(f"  {gender.capitalize()}: {rate:.4f}  ({rate*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# predict_survival() function
# ─────────────────────────────────────────────────────────────────────────────

def predict_survival(passenger_dict):
    """
    Predicts survival for a single passenger using the handcrafted score.

    Parameters:
        passenger_dict (dict): Keys - Sex, Pclass, Age, Fare, FamilySize

    Returns:
        dict: score (float), prediction (int), label (str)
    """
    # Weights
    w1, w2, w3, w4, w5 = 0.40, 0.25, 0.20, 0.10, 0.05

    # Gender encoding
    gender_score = 1.0 if passenger_dict.get('Sex', 'male') == 'female' else 0.0

    # Pclass: 1 is best, 3 is worst → invert and normalize
    pclass_score = (4 - passenger_dict.get('Pclass', 3)) / 2.0  # range [0,1]

    # Fare
    fare = passenger_dict.get('Fare', 15.0)
    fare_score = np.clip((fare - 7.0) / (512.0 - 7.0), 0, 1)

    # Age group
    age = passenger_dict.get('Age', 30.0)
    if age < 15:
        age_score = 1.0
    elif age <= 60:
        age_score = 0.5
    else:
        age_score = 0.0

    # Family size (too large or alone = lower survival)
    family_size = passenger_dict.get('FamilySize', 1)
    family_score = np.clip((family_size - 1) / 6.0, 0, 1)
    if family_size == 1:
        family_score = 0.2

    raw_score = (w1 * gender_score +
                 w2 * pclass_score +
                 w3 * fare_score +
                 w4 * age_score +
                 w5 * family_score)

    prediction = 1 if raw_score >= 0.5 else 0
    label = "Survived" if prediction == 1 else "Did Not Survive"

    return {
        "score": round(raw_score, 4),
        "prediction": prediction,
        "label": label
    }


# ── Test predict_survival ─────────────────────────────────────────────────────
print("\n--- predict_survival() Test Cases ---")

test_cases = [
    {"Sex": "female", "Pclass": 1, "Age": 28, "Fare": 100.0, "FamilySize": 2},
    {"Sex": "male",   "Pclass": 3, "Age": 35, "Fare": 7.5,   "FamilySize": 1},
    {"Sex": "female", "Pclass": 3, "Age": 12, "Fare": 15.0,  "FamilySize": 4},
    {"Sex": "male",   "Pclass": 2, "Age": 45, "Fare": 30.0,  "FamilySize": 3},
]

for i, p in enumerate(test_cases, 1):
    result = predict_survival(p)
    print(f"\n  Passenger {i}: {p}")
    print(f"  → Score: {result['score']}, Prediction: {result['label']}")
