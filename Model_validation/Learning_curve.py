
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve


fig, axs= plt.subplots(1, 2,figsize=(16,6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for i, n in enumerate(([2, 9])):
    N, train_lc, val_lc= learning_curve(Polyn)