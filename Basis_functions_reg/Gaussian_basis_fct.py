import numpy as np
import matplotlib.pyplot as plt

# Define the function
def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2) / (2 * np.power(sigma, 2)))

# Generate the x values
x = np.linspace(-5, 5, 1000)

# Compute the y values for the Gaussian function
y = gaussian(x, 0, 1)

# Plot the Gaussian function
plt.plot(x, y)
plt.title('Gaussian Function with mu=0 and sigma=1')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
