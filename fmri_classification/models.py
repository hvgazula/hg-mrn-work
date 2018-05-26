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

# Loading y data
y = sio.loadmat('data/labels.mat')
y = y['labels']
print('Shape of y is: ', y.shape)
y = np.reshape(y, (869,))

# Polyssifier
report = poly(X.T, y)

# Logistic Regression

# Support Vector Machines

# Decision Trees

# Random Forests

# XGBoost

#