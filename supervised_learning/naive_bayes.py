import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.naive_bayes import GaussianNB
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)

# Scatter plot of the data points
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='RdBu')
#plt.show()

# Fit the Gaussian Naive Bayes model to the data
model = GaussianNB()
model.fit(X, y)

# generate new data and predict their labels

rng = np.random.RandomState(24)
Xnew = [-6, -14] + [14,18] * rng.rand(2000,2)
ynew = model.predict(Xnew)
print(ynew)

plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:,0], Xnew[:,1], c=ynew, s=20, cmap='RdBu', alpha=0.09)
#plt.show()

#Classification of the Iris dataset

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X.shape)
print(y.shape)

model = GaussianNB()
model.fit(X_train,y_train)
y_model = model.predict(X_test)
print(accuracy_score(y_test, y_model))