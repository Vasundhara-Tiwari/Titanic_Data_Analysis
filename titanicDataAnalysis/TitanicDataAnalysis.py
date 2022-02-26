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

