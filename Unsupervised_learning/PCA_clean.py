import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

matplotlib.use('TkAgg')  # or 'Qt5Agg' or 'Qt4Agg' depending on your backend


# Load the iris dataset
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Perform PCA on the iris dataset
model = PCA(n_components=2)
model.fit(X_iris)
X_2D = model.transform(X_iris)

# Create a pandas DataFrame to store the iris dataset with the PCA components
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Add the PCA components to the iris DataFrame
iris_df['PCA1'] = X_2D[:, 0]
iris_df['PCA2'] = X_2D[:, 1]

# Plot the PCA components using seaborn
sns.lmplot(x='PCA1', y='PCA2', hue='target', data=iris_df, fit_reg=False)

# Display the plot
plt.show()


# Alternate way to plot the PCA components using matplotlib
# plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y_iris)
# plt.show()
