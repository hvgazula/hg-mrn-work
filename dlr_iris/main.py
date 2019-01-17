#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 09:49:23 2019

@author: Harshvardhan
"""
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
x = iris.data
y = iris.target

x_new = x[y < 2, :]
y_new = y[y < 2]

# Pooled
logreg = LogisticRegression()
logreg.fit(x_new, y_new)
print (logreg.coef_)

# Gradient Descent