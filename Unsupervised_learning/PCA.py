from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 


import matplotlib
# This line is for making the plot using sns possible by setting the backend

matplotlib.use('TkAgg')  # or 'Qt5Agg' or 'Qt4Agg' depending on your backend


iris = load_iris()
X_iris, y_iris = iris.data, iris.target
model = PCA(n_components=2)
model.fit(X_iris)
X_2D = model.transform(X_iris)


iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

print(iris.head())
iris['PCA1'] = X_2D[:,0]
iris['PCA2'] = X_2D[:,1]
sns.lmplot(x="PCA1", y="PCA2", hue='target', data=iris, fit_reg=False)
plt.show()
#ssns.show()


# print(X_2D)
# plt.scatter(X_2D[:,0], X_2D[:,1], c=y_iris)
# plt.show()
