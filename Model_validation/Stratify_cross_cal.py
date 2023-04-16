from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np

X, y = np.ones((50,1)), np.hstack(([0]*45, [1]*5))
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
    print(f'train - {np.bincount(y[train])} | test {np.bincount(y[test])}')