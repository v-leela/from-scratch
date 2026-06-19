import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def calculate_gradient(theta, X, y):
    m = len(X)
    return (1/m) * (X.T @ (sigmoid(theta @ X) - y))

def gradient_descent(X, y, alpha = 0.1, iterations = 100, tol = 1e-7):
    Xb = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(1, Xb.shape[0])
    diff = 100

    while diff > tol:
        gradient = calculate_gradient(theta, Xb, y)
        new_theta = theta - alpha * gradient
        diff = np.abs(new_theta - theta)

    return theta

def predict_prob(X, theta):
    Xb = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(theta @ X)

def predict(X, theta, threshold = 0.5):
    return (predict_prob(X, theta) >= 0.5).astype(int)