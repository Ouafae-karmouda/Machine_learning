import matplotlib.pyplot as plt 
import numpy as np
from sklearn.linear_model import LinearRegression


rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y)
#plt.show()

model = LinearRegression(fit_intercept=True)

# Turn the features into the right size

X = x[:,np.newaxis]
print(X.shape)

#Apply our model to data
model.fit(X,y)

# print the parameters learnt
print(model.coef_)
print(model.intercept_)

# predict labels of new data
xfit = np.linspace(-1, 11)

Xfit = xfit[:,np.newaxis]
yfit = model.predict(Xfit)

#plot data and the fitted line
plt.scatter(X,y)
plt.plot(Xfit,yfit)
plt.show()