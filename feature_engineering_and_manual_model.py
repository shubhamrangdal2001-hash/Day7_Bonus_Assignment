
import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)
df['FarePerPerson'] = df['Fare'] / df['FamilySize']

df['AgeGroup'] = pd.cut(df['Age'], bins=[0,15,60,100], labels=['Child','Adult','Senior'])
df['FareGroup'] = pd.qcut(df['Fare'], q=3, labels=['Low','Medium','High'])

df['Gender'] = df['Sex'].map({'male':0,'female':1})
df['ClassScore'] = 4 - df['Pclass']
df['AgeScore'] = df['AgeGroup'].map({'Child':2,'Adult':1,'Senior':0})
df['FareScore'] = df['FareGroup'].map({'Low':0,'Medium':1,'High':2})

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

norm_gender = normalize(df['Gender'].values)
norm_class = normalize(df['ClassScore'].values)
norm_age = normalize(df['AgeScore'].values)
norm_fare = normalize(df['FareScore'].values)
norm_family = normalize(df['FamilySize'].values)

score = (
    0.35 * norm_gender +
    0.25 * norm_class +
    0.15 * norm_fare +
    0.15 * norm_age +
    0.10 * norm_family
)

prob = score / score.max()
prediction = np.where(prob >= 0.5, 1, 0)

print("Manual survival model ready.")
