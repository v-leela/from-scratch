import numpy as np


class LinearRegression:
    def __init__(self, alpha=0.1, iterations=100, tol=1e-7):
        self.coef = None
        self.intercept = 0
        self.theta = None

        self.alpha = alpha
        self.iterations = iterations
        self.tol = tol

    def calculate_gradient(self, X, y, theta):
        m = len(X)
        return X.T @ (X @ theta - y) / m

    def gradient_descent(self, X, y):
        Xb = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.zeros(Xb.shape[1])
        diff = 100

        for _ in range(self.iterations):
            gradient = self.calculate_gradient(Xb, y, theta)
            new_theta = theta - self.alpha * gradient
            diff = np.linalg.norm(new_theta - theta)
            theta = new_theta
            if diff < self.tol:
                break

        return theta

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        theta = self.gradient_descent(X, y)

        self.coef = theta[1:,]
        self.intercept = theta[0,]
        self.theta = theta

    def predict(self, X):
        X = np.array(X)

        if self.theta is not None:
            return X @ self.coef + self.intercept

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
