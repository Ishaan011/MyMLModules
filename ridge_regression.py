import numpy as np

class RidgeRegressor_1D:

    def __init__(self, alpha = 0.1):
        self . alpha = alpha
        self . m = None
        self . b = None

    def fit(self, X_train, y_train):
        
        num = np . sum((y_train - np . mean(y_train)) * (X_train - np .mean(X_train)), axis = 0)
        den = np . sum((X_train - np . mean(X_train)) * (X_train - np .mean(X_train)), axis = 0) + self . alpha

        self . m = num / den
        self . b = np . mean(y_train) - (self . m * np . mean(X_train))
        print("m: ", self . m,"\nb: ", self . b)
    
    def predict(X_test):
        pass


class RidgeRegressorOLS:

    def __init__(self, alpha = 0.1):
        self . alpha = alpha
        self . coef_  = None
        self . intercept_ = None
        

    def fit(self, X_train, y_train):
        X_train = np . insert(X_train, 0, 1, axis = 1)
        I = np . identity(X_train . shape[1])
        I[0][0] = 0 # Only the coefficients are supposed to be regularised
        weights = np . linalg . inv(np . dot(X_train . T, X_train) - self . alpha * I) . dot(X_train . T) . dot(y_train)
        self . intercept_ = weights[0]
        self . coef_ = weights[1:]

        print("Coeff: ", self . coef_, "\nIntercept: ", self . intercept_)

    def predict(self, X_test):
        return np . dot(X_test, self . coef_) + self . intercept_

class RidgeRegressorGradDes:

    def __init__(self, epochs, learning_rate, alpha):
        self  . lr = learning_rate
        self . epochs = epochs
        self . alpha = alpha
        self . coef_ = None
        self . intercept_ = None

    def fit(self, X_train, y_train):
        self . coef_ = np . ones(X_train.shape[1]) 
        self . intercept_ = 0
        theta = np . insert(self . coef, 0, self . intercept_)

        X_train = np . insert(X_train, 0, 1, axis = 1)

        for i in range(self . epochs):
            theta_der = np . dot(X_train . T, X_train) . dot(theta) - np . dot(X_train . T, y_train) + self . alpha * theta
            theta -= self . lr * theta_der

        self . coef_ = theta[1:]
        self . intercept_ = theta[0]

    def predict(self, X_test):
        return np . dot(X_test, self . coef_) + self . intercept_