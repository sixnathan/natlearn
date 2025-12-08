import numpy as np

class LinearRegression:
    def __init__(self):
        self.theta = None

    def hypo(self, X):
        return X @ self.theta

    def cost(self, X, y):
        h = self.hypo(X)
        loss = h - y
        cost = (loss.T @ loss)[0,0] / (2 * X.shape[0])
        return cost

    def graddec(self, X, y, rate, iterations):
        self.theta = np.zeros((X.shape[1], 1))
        for i in range(iterations):
            h = self.hypo(X)
            newgrad = (1/X.shape[0]) * X.T @ (h - y)
            self.theta = self.theta - rate * newgrad