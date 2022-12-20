# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:42:53 2019

@author: hgazula
"""
import numpy as np
from sklearn import datasets

from numba import jit, prange


#@jit(nopython=True)
def gottol(vector, tol=1e-5):
    """Check if the gradient meets the tolerances"""
    return np.sum(np.square(vector)) <= tol


#@jit(nopython=True)
def gradient(weights, X, y, lamb=0.0):
    """Computes the gradient"""
    hthetaofx = 1 / (1 + np.exp(-np.dot(X, weights.reshape(-1, 1))))
    bac = hthetaofx - y
    bac1 = np.dot(bac.T, X)
    bac2 = (1 / len(X)) * bac1 + lamb * weights
    return bac2


#@jit(nopython=True)
def multishot_gd(X1, y1, X2, y2):

    params = np.zeros(X1.shape[1])

    # Initialize at remote
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    wp = np.zeros(X1.shape[1])
    mt = np.zeros(X1.shape[1])
    vt = np.zeros(X1.shape[1])

    tol = 1e-4
    eta = 1e-2

    count = 0
    while True:
        count = count + 1

        # At local
        grad_local1 = gradient(wp, X1, y1, lamb=0)
        grad_local2 = gradient(wp, X2, y2, lamb=0)

        # at remote
        grad_remote = grad_local1 + grad_local2

        mt = beta1 * mt + (1 - beta1) * grad_remote
        vt = beta2 * vt + (1 - beta2) * (grad_remote**2)

        m = mt / (1 - beta1**count)
        v = vt / (1 - beta2**count)

        wc = wp - eta * m / (np.sqrt(v) + eps)
        print(wc)

        if np.linalg.norm(wc - wp) <= tol:
            break

        wp = wc

    avg_beta_vector = wc
    params = avg_beta_vector

    return (params)


iris = datasets.load_iris()
x = iris.data
y = iris.target

# Decentralized Logistic Regression
X1 = x[y == 0, :]
y1 = y[y == 0].reshape(-1, 1)
X2 = x[y == 1, :]
y2 = y[y == 1].reshape(-1, 1)

multishot_gd(X1, y1, X2, y2)
