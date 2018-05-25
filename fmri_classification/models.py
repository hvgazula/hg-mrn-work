# -*- coding: utf-8 -*-
"""
Created on Fri May 25 13:57:49 2018

@author: hgazula
"""
import h5py
import numpy as np
import scipy.io as sio

# Loading X data
f = h5py.File('data/MixingMatrix_Weights.mat')
X = np.array(f["MixingMatrix_Weights"])
print('Actual shape of X is: ', X.shape)

# Reshaping the X matrix
X = np.reshape(X, (869, -1))
print('After reshaping :', X.shape)

# Loading y data
y = sio.loadmat('data/labels.mat')
y = y['labels']
print('Shape of y is: ', y.shape)

# Logistic Regression

# Support Vector Machines

# Decision Trees

# Random Forests

# XGBoost

#