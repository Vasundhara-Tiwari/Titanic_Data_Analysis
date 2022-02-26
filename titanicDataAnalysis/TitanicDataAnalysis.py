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