import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def calculate_gradient(theta, X, y):
    m = len(X)
    return (1/m) * (X.T @ (sigmoid(X @ theta) - y))

def gradient_descent(X, y, alpha = 0.1, iterations = 100, tol = 1e-7): 
    Xb = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(Xb.shape[1])
    diff = 100

    for _ in range(iterations):
        gradient = calculate_gradient(theta, Xb, y)
        new_theta = theta - alpha * gradient
        diff = np.linalg.norm(new_theta - theta)
        theta = new_theta
        if diff < tol:
            break

    return theta

def predict_prob(X, theta):
    Xb = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(Xb @ theta)

def predict(X, theta, threshold = 0.5):
    return (predict_prob(X, theta) >= threshold).astype(int)
