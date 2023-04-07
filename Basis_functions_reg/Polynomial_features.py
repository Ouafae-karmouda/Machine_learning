import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

poly_model = make_pipeline(PolynomialFeatures(7),
                           LinearRegression())


#generate a sine wave
N = 50
rng = np.random.RandomState(1)
x = 10 * rng.rand(N)
y = np.sin(x) + 0.1 * rng.randn(N)


# split data on train ad test
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
#fit a poly model
model = poly_model.fit(x_train[:,np.newaxis], y_train)
#predict new labels
xfit = np.linspace(0, 10, 1000)
y_pred = model.predict(xfit[:, np.newaxis])
plt.figure()
plt.scatter(x_train, y_train)
plt.plot(xfit, y_pred, color='red')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.show()
plt.plot()




         
