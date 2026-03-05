"""
Part 2 - Advanced Pandas Engineering
Titanic Dataset Feature Engineering
"""

import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")

print("Original shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ── Task 1: Handle Missing Values ─────────────────────────────────────────────
# Age → fill with median per passenger class
df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))

# Embarked → fill with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Fare → fill with median (just in case)
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().sum().sum() > 0 else "None")

# ── Task 2: Feature Engineering ──────────────────────────────────────────────
df['FamilySize']    = df['SibSp'] + df['Parch'] + 1
df['IsAlone']       = (df['FamilySize'] == 1).astype(int)
df['FarePerPerson'] = df['Fare'] / df['FamilySize']

print("\nNew features sample:")
print(df[['FamilySize', 'IsAlone', 'FarePerPerson']].head(5))

# ── Task 3: Categorical Bins ─────────────────────────────────────────────────
df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[0, 14, 60, 100],
    labels=['Child', 'Adult', 'Senior']
)

df['FareGroup'] = pd.cut(
    df['Fare'],
    bins=[-1, 15, 50, df['Fare'].max()],
    labels=['Low', 'Medium', 'High']
)

print("\nAge Group distribution:")
print(df['AgeGroup'].value_counts())
print("\nFare Group distribution:")
print(df['FareGroup'].value_counts())

# ── Task 4: Pivot Tables ──────────────────────────────────────────────────────
print("\n--- Pivot: Survival by Gender & Class ---")
pivot1 = pd.pivot_table(df, values='Survived', index='Sex', columns='Pclass', aggfunc='mean')
print(pivot1.round(3))

print("\n--- Pivot: Survival by FareGroup & Embarked ---")
pivot2 = pd.pivot_table(df, values='Survived', index='FareGroup', columns='Embarked', aggfunc='mean')
print(pivot2.round(3))

# ── Task 5: Correlation Matrix ───────────────────────────────────────────────
num_cols = ['Survived', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'FarePerPerson', 'Pclass']
corr_matrix = df[num_cols].corr()

print("\n--- Correlation Matrix ---")
print(corr_matrix.round(3))

# ── Task 6: Top 5 Features by Survival Correlation ───────────────────────────
survival_corr = corr_matrix['Survived'].drop('Survived').abs().sort_values(ascending=False)
print("\n--- Top 5 Features Correlated with Survival ---")
print(survival_corr.head(5))

# ── Insight: Does wealth dominate gender in predicting survival? ───────────────
print("\n--- Survival by Gender ---")
print(df.groupby('Sex')['Survived'].mean().round(3))

print("\n--- Survival by Pclass ---")
print(df.groupby('Pclass')['Survived'].mean().round(3))

print("\n--- Survival by Gender & FareGroup ---")
print(df.groupby(['Sex', 'FareGroup'])['Survived'].mean().round(3))

print("""
Insight: Gender is the strongest single predictor (women ~74%, men ~19%).
Wealth (FareGroup/Pclass) plays a secondary but still significant role.
Even among males, high fare passengers survived at a higher rate (~40%).
So gender dominates, but wealth clearly amplifies survival chances.
""")

df.to_csv("titanic_cleaned.csv", index=False)
print("Cleaned dataset saved to titanic_cleaned.csv")
