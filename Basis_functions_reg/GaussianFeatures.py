import numpy as np
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""
    
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
    
    def _gauss_basis(self, x, y, width):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis=1))
    
    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
    
    def transform(self, X):
        return self._gauss_basis(X, self.centers_[:, np.newaxis], self.width_)
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def get_params(self, deep=True):
        return {'N': self.N, 'width_factor': self.width_factor}
    
    def set_params(self, **params):
        if 'N' in params:
            self.N = params['N']
        if 'width_factor' in params:
            self.width_factor = params['width_factor']
        return self



class GaussianFeatures1(BaseEstimator, TransformerMixin):
    
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor

    @staticmethod
    def _gauss_basis(x, y, width, axis = None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
    
    def fit(self, X, y = None):
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width = self.width_factor * (self.centers_[1] - self.centers_[1])
        return self
    
    def trasnsform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis()], self.centers_,
                                 self.width_, axis=1)
    
#generate a sine wave
N = 50
rng = np.random.RandomState(1)
X = 10 * rng.rand(N)
y = np.sin(X) + 0.1 * rng.randn(N)



# Define the pipeline
gauss_model = make_pipeline(
    GaussianFeatures(50),
    LinearRegression()
)

a = X.reshape(-1,1)
print(a)
# Fit the pipeline to the data
gauss_model.fit(X.reshape(-1,1), y.reshape(-1,1))




xfit = np.linspace(0, 10, 1000)
yfit = gauss_model.predict(xfit[:,np.newaxis])



plt.scatter(x,y)
plt.plot(xfit, yfit)
plt.xlim(0,10)
plt.show()