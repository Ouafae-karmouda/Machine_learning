import random
import numpy as np
import matplotlib.pyplot as plt


list_integers = []

for i in range(100):
    list_integers.append(random.randint(1,1000))


plt.hist(list_integers, bins = 30)
plt.show()
print("The sum of numbers", sum(list_integers))
print("The mean of numbers", np.mean(list_integers))
print("The median of numbers", np.median(list_integers))
print("The std of numbers", np.std(list_integers))





