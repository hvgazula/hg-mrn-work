#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 09:49:23 2019

@author: Harshvardhan
"""
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def gottol(v, tol=1e-5):
    return np.sum(np.square(v)) <= tol


def gradient(weights, X, y, lamb=0.0):
    """Computes the gradient"""
    hthetaofx = 1/(1 + np.exp(-1 * np.dot(X, weights.reshape(-1, 1))))
    bac = hthetaofx - y
    bac1 = np.dot(bac.T, X)
    bac2 = (1 / len(X)) * bac1 + lamb * weights
    return bac2


def pooled(X, y):
    logreg = LogisticRegression()
    logreg.fit(X, y.ravel())
    print (logreg.coef_)
    return None

    
def gd_for(winit, X, y, steps=1000, eta=0.01, tol=1e-6):
    wc = winit

    for i in range(steps):
        Gradient = gradient(wc, X, y)
        wc = wc - eta * gradient(wc, X, y)
        if gottol(Gradient, tol):
            break
    return wc, i


def adam(X, y):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    wp = np.zeros(X.shape[1])
    mt = np.zeros(X.shape[1])
    vt = np.zeros(X.shape[1])
    
    tol = 1e-4
    eta = 1e-2
    
    count = 0
    while True:
        count = count + 1
    
        # At local
        grad_remote = gradient(wp, X, y, lamb=0)        
    
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
    print(avg_beta_vector)
    
    return None

def nadam(X, y):
    beta1 = 0.99 # same as mu
    beta2 = 0.999 # same as v
    eps = 1e-8
    wp = np.zeros(X.shape[1])
    mt = np.zeros(X.shape[1])
    vt = np.zeros(X.shape[1])
    
    tol = 1e-4
    eta = 1e-2
    
    count = 0
    beta1t = beta1 * (1 - 0.5 * .96 ** (count/250))
    while True:
        count = count + 1
    
        # At local
        grad_remote = gradient(wp, X, y, lamb=0)
        grad_remote = grad_remote/(1 - sum(beta1t))
    
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
    print(avg_beta_vector)
    
    return None

if __name__ == "__main__":
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    
    x_new = x[y < 2, :]
    y_new = y[y < 2]

    w = np.array([0, 0, 0, 0])
    
    pooled(x_new, y_new)
    gd_for(w, x_new, y_new)
#    adam(x_new, y_new.reshape(-1,1))