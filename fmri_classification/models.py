# -*- coding: utf-8 -*-
"""
Created on Fri May 25 13:57:49 2018

@author: hgazula
"""
import numpy as np
import scipy.io as sio
from polyssifier import poly

## Loading X data
#f = h5py.File('data/X.mat')
#X = np.array(f["MixingMatrix_Weights"])
#print('Actual shape of X is: ', X.shape)
#
## Reshaping the X matrix
#X = np.reshape(X, (869, -1), order='C')
#print('After reshaping :', X.shape)

# Loading y data
X = sio.loadmat('data/X.mat')
X = X['X']
print('Shape of X is: ', X.shape)


## Zelected from RFE
index = [48,  170,  260,  290,  312,  345,  353,  361,  418,  428,  503,
         516,  646,  656,  690,  722,  880,  923,  947,  951,  984,  989,
         991, 1035, 1154, 1165, 1205, 1280, 1301, 1375, 1430, 1432, 1465,
        1527, 1568, 1574, 1582, 1590, 1625, 1626, 1674, 1686, 1704, 1714,
        1736, 1742, 1751, 1764, 1799, 1849]

X_ = X[:, index]

# Loading y data
y = sio.loadmat('data/labels.mat')
y = y['labels']
print('Shape of y is: ', y.shape)
y = np.reshape(y, (869,))

# Polyssifier
report = poly(X_, y, n_folds=10)

# Logistic Regression

# Support Vector Machines

# Decision Trees

# Random Forests

# XGBoost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#seed = 7
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=test_size)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#### Recursive Feature Estimator
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#seed = 7
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

estimator = SVC(kernel="rbf")
