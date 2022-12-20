# -*- coding: utf-8 -*-
"""
Created on Fri May 25 13:57:49 2018

@author: hgazula
"""

from load_data_from_mat import return_X_and_y
from polyssifier import poly
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = return_X_and_y()
# Polyssifier
report = poly(X, y, n_folds=10)

# XGBoost
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
