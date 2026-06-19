import numpy as np
""" 
class LinearRegressionClosed:

    def __init__(self):
        self.coef = None
        self.intercept = 0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        Xb = np.c_[np.ones((X.shape[0], 1)), X]

    def predict(self, X):
        pass """

x = np.array([1,23,4,5])
y = np.ones((4,1))
print(np.c_[y,x])
