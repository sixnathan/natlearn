import numpy as np

class LogisticRegression:
    def __init__(self):
        self.theta = None

    def sig(self, z):
        h = 1 / (1 + np.exp(-z))
        return h

    def hypo(self, X):
        return self.sig(X @ self.theta)

    def cost(self, X, y):
        h = self.hypo(X)
        cost = (-1/X.shape[0]) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
        return cost

    def graddec(self, X, y, rate, iterations):
        self.theta = np.zeros((X.shape[1], 1))
        for i in range(iterations):
            h = self.hypo(X)
            newgrad = (1/X.shape[0]) * X.T @ (h - y)
            self.theta = self.theta - rate * newgrad
