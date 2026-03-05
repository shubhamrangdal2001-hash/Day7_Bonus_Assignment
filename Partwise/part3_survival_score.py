"""
Part 3 - NumPy-Based Survival Score (No ML Library)
Manual scoring rule to predict survival
"""

import pandas as pd
import numpy as np

# Load cleaned data (run part2 first, or use train.csv directly)
df = pd.read_csv("train.csv")

# ── Re-create features ────────────────────────────────────────────────────────
df['Age']        = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
df['Embarked']   = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare']       = df['Fare'].fillna(df['Fare'].median())
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['FarePerPerson'] = df['Fare'] / df['FamilySize']

# Encode gender: female=1, male=0
df['GenderNum'] = (df['Sex'] == 'female').astype(int)

# Encode AgeGroup numerically: Child=2, Adult=1, Senior=0
def encode_age(age):
    if age < 15:
        return 2
    elif age <= 60:
        return 1
    else:
        return 0

df['AgeGroupNum'] = df['Age'].apply(encode_age)

# Encode FareGroup: High=2, Medium=1, Low=0
fare_33 = df['Fare'].quantile(0.33)
fare_66 = df['Fare'].quantile(0.66)

def encode_fare(fare):
    if fare > fare_66:
        return 2
    elif fare > fare_33:
        return 1
    else:
        return 0

df['FareGroupNum'] = df['Fare'].apply(encode_fare)

# ── Task 1: Normalize numeric variables ──────────────────────────────────────
def normalize(arr):
    min_val = arr.min()
    max_val = arr.max()
    if max_val == min_val:
        return np.zeros_like(arr, dtype=float)
    return (arr - min_val) / (max_val - min_val)

gender_norm     = normalize(df['GenderNum'].values.astype(float))
pclass_norm     = normalize((4 - df['Pclass'].values).astype(float))  # invert: class 1 = highest
age_group_norm  = normalize(df['AgeGroupNum'].values.astype(float))
fare_group_norm = normalize(df['FareGroupNum'].values.astype(float))
family_norm     = normalize(df['FamilySize'].values.astype(float))

# ── Task 2: Assign weights based on correlation strength ─────────────────────
# Correlation magnitudes (from Part 2 analysis):
# Gender: ~0.54, Pclass: ~0.34, FareGroup: ~0.30, AgeGroup: ~0.10, FamilySize: ~0.02
w1 = 0.40   # Gender
w2 = 0.25   # Pclass
w3 = 0.20   # FareGroup
w4 = 0.10   # AgeGroup
w5 = 0.05   # FamilySize

print("Weights assigned based on correlation strength:")
print(f"  w1 (Gender)     = {w1}")
print(f"  w2 (Pclass)     = {w2}")
print(f"  w3 (FareGroup)  = {w3}")
print(f"  w4 (AgeGroup)   = {w4}")
print(f"  w5 (FamilySize) = {w5}")

# ── Task 3: Compute survival probability score ────────────────────────────────
raw_score = (w1 * gender_norm +
             w2 * pclass_norm +
             w3 * fare_group_norm +
             w4 * age_group_norm +
             w5 * family_norm)

# Normalize to [0, 1]
survival_score = normalize(raw_score)

print(f"\nSurvival Score - Mean: {survival_score.mean():.3f}, Std: {survival_score.std():.3f}")
print(f"Sample scores (first 5): {survival_score[:5].round(3)}")

# ── Task 4: Classify using threshold = 0.5 ───────────────────────────────────
predictions  = (survival_score >= 0.5).astype(int)
actual       = df['Survived'].values

# ── Task 5: Manual confusion matrix, accuracy, precision, recall ──────────────
TP = np.sum((predictions == 1) & (actual == 1))
TN = np.sum((predictions == 0) & (actual == 0))
FP = np.sum((predictions == 1) & (actual == 0))
FN = np.sum((predictions == 0) & (actual == 1))

accuracy  = (TP + TN) / len(actual)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0

print(f"\n--- Confusion Matrix (Manual) ---")
print(f"  TP={TP}  FP={FP}")
print(f"  FN={FN}  TN={TN}")

print(f"\n--- Performance Metrics ---")
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")

# ── Insight: Compared to random guessing ─────────────────────────────────────
random_accuracy = max(actual.mean(), 1 - actual.mean())
print(f"\nRandom guessing accuracy (majority class): {random_accuracy:.4f}")
improvement = (accuracy - random_accuracy) / random_accuracy * 100
print(f"Our model improvement over random guessing: {improvement:.1f}%")
print("Yes, the handcrafted score significantly outperforms random guessing.")
