import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

dataset = pd.read_csv("tested.csv")
print(dataset.head(10))

print("number of passengers in original data:" + str(len(dataset.index)))

sns.countplot(x="Survived", data=dataset)
plt.show()

sns.countplot(x="Survived", hue="Sex", data=dataset)
plt.show()

sns.countplot(x="Survived", hue="Pclass", data=dataset)
plt.show()

dataset["Age"].plot.hist()
plt.show()

dataset["Fare"].plot.hist(bins=20, figsize=(10,5))
plt.show()

sns.countplot(x="SibSp", data=dataset)
plt.show()

sns.countplot(x="Parch", data=dataset)
plt.show()

## Performing data wrangling

print(dataset.isnull())

print(dataset.isnull().sum())

sns.heatmap(dataset.isnull(), yticklabels=False, cmap="viridis")
plt.show()

sns.boxplot(x="Pclass", y="Age", data=dataset)
plt.show()

dataset.drop("Cabin", axis=1, inplace=True)
print(dataset.head(5))

dataset.dropna(inplace=True)
sns.heatmap(dataset.isnull(), yticklabels=False, cmap="viridis")
plt.show()

print(dataset.isnull().sum())

sex = pd.get_dummies(dataset["Sex"], drop_first=True)
print(sex.head(5))

embark = sex = pd.get_dummies(dataset["Embarked"], drop_first=True)
print(embark.head(5))

pcl = sex = pd.get_dummies(dataset["Pclass"], drop_first=True)
print(pcl.head(5))

dataset = pd.concat([dataset, sex, embark, pcl], axis=1)
print(dataset.head(10))

dataset.drop(["Sex", "Pclass", "Embarked", "PassengerId", "Name", "Ticket"], axis=1, inplace=True)
print(dataset.head(5))
