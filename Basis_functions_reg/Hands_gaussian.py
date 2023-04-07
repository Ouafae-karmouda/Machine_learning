import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma):

    return np.exp(  -np.power(x-mu, 2)/ ( 2 * np.power(sigma, 2) ))


# generate the x values
x = np.linspace(-5,5, 1000)

mu = 0.5
sigma = 0.1
# compute the y values
y = gaussian(x, mu = mu, sigma = sigma)

# plot the  gaussian fct

plt.figure()
plt.plot(x,y)
plt.xlabel('x values')
plt.ylabel('f(x)')
plt.title(f'Gaussian fct with {mu} AND {sigma}')
plt.show()


import numpy as np

def gauss_basis(x, mu, sigma):
    arg = (x - mu) / sigma
    return np.exp(-0.5 * arg ** 2)

# Generate input data
x = np.linspace(0, 4, 1000)

# Generate centers for the Gaussian basis functions
centers = np.array([1, 2, 3])

# Generate the width of the Gaussian basis functions
width = 0.5

# Compute the value of the Gaussian basis functions for each input data point
phi = np.zeros((len(x), len(centers)))
for i in range(len(centers)):
    phi[:, i] = gauss_basis(x, centers[i], width)

# Plot the Gaussian basis functions
import matplotlib.pyplot as plt

for i in range(len(centers)):
    plt.plot(x, phi[:, i], label='Basis function {}'.format(i+1))

plt.xlabel('x')
plt.ylabel('Basis function value')
plt.legend()
plt.show()
