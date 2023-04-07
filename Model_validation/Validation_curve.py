import numpy as np
import seaborn
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))


# le t us generta esome data

def make_data(N, err= 1.0, seed=1):
    """_summary_

    Args:
        N (strictly positive integer): number data points
        err (int, optional): _description_. Defaults to 0.
        seed (int, optional): _description_. Defaults to 1.
    """
    rng = np.random.RandomState(seed)
    X = rng.rand(N,1)
    y = 10 - 1./(X.ravel() +0.1) 

    if err >0:
        y += err*rng.randn(N)
    return X, y

X, y= make_data(N=40)
#plot the data
X_test = np.linspace(-0.1, 1.1, 500)[:, None]
#plt.scatter(X.ravel(), y, color = 'black')
axis = plt.axis()

for degree in [1, 3, 5]:
    y_test = PolynomialRegression(degree).fit(X,y).predict(X_test)
    #plt.plot(X_test.ravel(), y_test, label ='degree={0}'.format(degree))
    
#plt.xlim(-0.1, 1.0)

#plt.ylim(-2, 12)
#plt.legend(loc='best');
#plt.show()


#Validation curve

degree = np.arange(0,21)
#train_score, val_score = validation_curve(PolynomialRegression(), X, y,
#                                          'polynomialfeatures__degree',
#                                          degree, cv=7)


train_score, val_score = validation_curve(PolynomialRegression(), X, y,
                                          param_name='polynomialfeatures__degree',
                                          param_range=degree, cv=7)


plt.plot(degree, np.median(train_score, 1), color='blue', label = 'training_score')
plt.plot(degree, np.median(val_score, 1), color = 'red', label = "Validation_score" )
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')
plt.show();

# Thr optimal trade off between variance and bias  is found for a third order polynomial

plt.scatter(X.ravel(),y)
lim =plt.axis()
y_test = PolynomialRegression(3).fit(X,y).predict(X_test)
plt.plot(X_test.ravel(), y_test)
plt.axis(lim)
plt.show()



fig, ax= plt.subplots(1, 2,figsize=(16,6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for i, degree in enumerate(([2, 9])):
    N, train_lc, val_lc= learning_curve(PolynomialRegression(degree), 
                                        X, y, cv=7,
                                        train_sizes=np.linspace(0.3, 1, 25))
    print(N)
    ax[i].plot(N, np.mean(train_lc, 1), color='blue', label='training_score')
    ax[i].plot(N, np.mean(val_lc, 1), color='red', label='val_score')
    ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1], color='gray',
                   linestyle='dashed')
    
    ax[i].set_ylim(0, 1)
    ax[i].set_xlim(N[0], N[-1])
    ax[i].set_xlabel('training size')
    ax[i].set_ylabel('score')
    ax[i].set_title('degree = {0}'.format(degree), size=14)
    ax[i].legend(loc='best')
plt.show()

# One aspect of model complexity The optimal model depend on the training size

X2, y2 = make_data(200)

# Make a plot of accurcay of the model w.r.t model complexity for 200 data

plt.figure(figsize=(7,4))
degree = np.arange(21)

train_score2, val_score2 = validation_curve(PolynomialRegression(),X2, y2,
                                    param_range=degree,
                                    param_name='polynomialfeatures__degree',
                                    cv=7)
print(val_score2)
plt.plot(degree, np.median(train_score2, 1), color='blue', label='Training_score' )
plt.plot(degree, np.median(val_score2,1), color='red', label='Validation_score')

plt.plot(degree, np.median(train_score, 1), color='blue', label='Training_score',
            linestyle = 'dashed' )
plt.plot(degree, np.median(val_score,1), color='red', label='Validation_score',
            linestyle='dashed')

plt.ylim(0,1)
plt.show()