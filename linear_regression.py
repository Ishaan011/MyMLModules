import numpy as np

class multiLinearReg:

    def __init__(self):
        self . coef_ = None
        self . intercept_ = None

    def fit(self, X_train, y_train):
        X_train =np . insert(X_train, 0, 1, axis = 1)
        #calculate the coefficients
        try:
            betas = np . linalg . inv(np . dot(X_train . T, X_train)) . dot(X_train . T) . dot(y_train)
        except:
            X_train[3,1] *= 0.9999999999999
            betas = np . linalg . inv(np . dot(X_train . T, X_train)) . dot(X_train . T) . dot(y_train)

        self . intercept_ = betas[0]
        self . coef_ = betas[1:]

        return self . coef_, self . intercept_
        
    def predict(self, X_test):
        y_pred = np . dot(X_test, self . coef_) + self . intercept_
        return y_pred