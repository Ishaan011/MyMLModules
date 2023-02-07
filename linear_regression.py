import numpy as np

class multiLinearReg:
    #initializing the variables
    def __init__(self):
        self . coef_ = None
        self . intercept_ = None

    def fit(self, X_train, y_train):
        # Inserting the bias column to the training data
        X_train =np . insert(X_train, 0, 1, axis = 1)
        
        #calculating the coefficients
        try:
            betas = np . linalg . inv(np . dot(X_train . T, X_train)) . dot(X_train . T) . dot(y_train)
        except:
            X_train[3,1] *= 0.9999999999999
            betas = np . linalg . inv(np . dot(X_train . T, X_train)) . dot(X_train . T) . dot(y_train)

        # First column is the intercept
        self . intercept_ = betas[0]
        
        # Rest of the columns are the coefficients
        self . coef_ = betas[1:]

        return self . coef_, self . intercept_
        
    def predict(self, X_test):
        # Prediction on the input data
        y_pred = np . dot(X_test, self . coef_) + self . intercept_
        return y_pred