import numpy as np 

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Load the data
iris = load_iris()

# Get the features and the target
X, y = iris.data, iris.target

#Split the data
X1, X2, y1, y2 = train_test_split(X,y, test_size=0.5, random_state=1)

#instantiate the model
model = KNeighborsClassifier(n_neighbors=5)

#fit and predict
ypred1 = model.fit(X1,y1).predict(X2)
ypred2 = model.fit(X2,y2).predict(X1)

list_scores = [accuracy_score(y2, ypred1), accuracy_score(y1, ypred2)]
print(list_scores)
print(f"The mean score is {np.mean(list_scores)} with standard deviation {np.std(list_scores)}")