import numpy as np
import matplotlib.pyplot as plt

# Create 1-dimensional arrays of x and y coordinates
x = np.linspace(-1, 1, num=3)
y = np.linspace(-1, 1, num=3)

# Create a meshgrid of points that covers the entire range of x and y
xx, yy = np.meshgrid(x, y)

# Plot the meshgrid of points
plt.scatter(xx, yy)

# Show the plot
plt.show()
