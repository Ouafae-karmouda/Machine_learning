import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Loading the data
iris = load_iris()

#Getting the features and the target
X,y = iris.data, iris.target

#Instantiate the model
model = KNeighborsClassifier(n_neighbors=5)

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=np.random.seed(1), test_size=0.5)

#train the model
model.fit(X_train, y_train)

#Get the prediction
y_pred = model.predict(X_test)

#Get the accuracy of the model
score = accuracy_score(y_test, y_pred)

print(f"\n Score of Kneighbor model {score}")

