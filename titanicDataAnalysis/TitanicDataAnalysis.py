import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# importing data fom csv file
dataset = pd.read_csv("tested.csv")
print(dataset.head(10))

#printing length of dataset
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

# heatmap of the dataset
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


##Training and testing data

X=dataset.drop("Survived", axis=1)
y=dataset["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
lgmodel = LogisticRegression()
lgmodel.fit(X_train, y_train)

predictions = lgmodel.predict(X_test)
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))
print(str(accuracy_score(y_test, predictions))+"%")