import numpy as np


class LinearRegressionClosed:
    def __init__(self):
        self.coef = None
        self.intercept = 0
        self.theta = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        Xb = np.c_[np.ones((X.shape[0], 1)), X]

        # theta from solving the gradient of the jacobian
        theta = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y

        self.coef = theta[1:,]
        self.intercept = theta[0,]
        self.theta = theta

    def predict(self, X):
        X = np.array(X)
        Xb = np.c_[np.ones((X.shape[0], 1)), X]

        if self.theta is not None:
            return Xb @ self.theta

    def loss(self, X, y):
        y_pred = self.predict(X)
        if self.theta is not None:
            return np.mean((y_pred - y) ** 2) / 2

    def rmse(self, X_test, y_test):
        prediction = self.predict(X_test)
        return np.sqrt(np.mean((prediction - y_test) ** 2))

    def r_score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        r_sq = 1 - np.sum((y_test - y_pred) ** 2) / np.sum(
            (y_test - np.mean(y_test)) ** 2
        )

        return r_sq
