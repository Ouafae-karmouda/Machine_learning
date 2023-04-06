import matplotlib.pyplot as plt

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# Assign the features to X and the target to y
X = iris.iloc[:, :-1]
y = iris.iloc[:, -1]

# Instantiate the scaler
scaler = StandardScaler()

# Fit and transform the data
X_normalized = scaler.fit_transform(X)


# Plot the distribution of each normalized feature using a histogram
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].hist(X_normalized[:, 0], bins=20)
axs[0, 0].set_title('Sepal Length')
axs[0, 1].hist(X_normalized[:, 1], bins=20)
axs[0, 1].set_title('Sepal Width')
axs[1, 0].hist(X_normalized[:, 2], bins=20)
axs[1, 0].set_title('Petal Length')
axs[1, 1].hist(X_normalized[:, 3], bins=20)
axs[1, 1].set_title('Petal Width')
plt.show()



# Plot the normalized features
plt.plot(X_normalized)
plt.xlabel('Sample Index')
plt.ylabel('Normalized Feature Value')
plt.show()
