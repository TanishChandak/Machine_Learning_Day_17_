import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()

print(dir(iris))
print(iris.feature_names )

# Conveting the iris datas into the dataset:
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

# appending the target value from the iris data in the (df):
df['target'] = iris.target
print(df.head())

# appending the flowers name to specifiy them
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
print(df.head())

# data visualization:
df0 = df[df.target == 0]
print(df0.head())
df1 = df[df.target == 1]
print(df1.head())
df2 = df[df.target == 2]
print(df2.head())

# plotting a graph using the scatter plot:
# this is for (sepal):
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='green', marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='blue', marker='.')
plt.show()

# this is for (petal):
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='green', marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='blue', marker='.')
plt.show()

# Training and testing the dataset:
from sklearn.model_selection import train_test_split

# independent variables:
X = df.drop(['target', 'flower_name'], axis='columns')
print(X.head())

# dependent variables:
y = df.target
print(y.head( ))

# traning the data:
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print("The length of the X_train:",len(X_train))
print("The length of the X_test:",len(X_test))

# Creating an KNN classifier:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

# Confusion matrix:
from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Classification reports:
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))