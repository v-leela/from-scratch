import numpy as np

class LogisticRegression:
    def __init__(self, alpha = 0.1, iterations = 100, tol = 1e-7, threshold = 0.5):
        self.coef = None
        self.intercept = 0
        self.theta = None

        self.alpha = alpha
        self.iterations = iterations
        self.tol = tol
        self.threshold = threshold

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def calculate_gradient(self, theta, X, y):
        m = len(X)
        return (1/m) * (X.T @ (sigmoid(X @ theta) - y))

    def gradient_descent(self, X, y): 
        Xb = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.zeros(Xb.shape[1])
        diff = 100

        for _ in range(self.iterations):
            gradient = calculate_gradient(theta, Xb, y)
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

    def loss(self, X, y):
        m = len(X)
        y_pred = self.predict(X)
        if self.theta is not None:
            return (-1 / m) * (np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))

    def predict_prob(self, X):
        if self.theta is not None:
            return sigmoid(X @ self.coef + self.intercept)

    def predict(self, X):
        return (predict_prob(X, self.theta) >= self.threshold).astype(int)
