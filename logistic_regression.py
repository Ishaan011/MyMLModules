import numpy as np

class max_likelihood:
    def __init__(self, learning_rate, epochs):
        self . lr = learning_rate
        self . epochs = epochs
        

    def fit(self, X, y):
        X = np . insert(X, 0, 1, axis = 1)
        self . weights = np . ones(X . shape[1])

        for i in range(self . epochs):
            y_hat = self . sigmoid(np . dot(X, self . weights))
            m = X . shape[0]
            self . weights += self . lr *(np . dot((y - y_hat), X) / m)

        return self . weights[0], self . weights[1:]

    def predict(self, X_test):
        y_pred = self . sigmoid(np . dot(X_test, self . weights))
        return np . where(y_pred<0.5, 0, 1)
    
    def sigmoid(self, z):
        return 1 / (1 + np . exp(-z))