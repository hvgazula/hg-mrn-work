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
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    hthetaofx = 1/(1 + np.exp(-1 * np.dot(X, weights.reshape(-1, 1))))
    bac = hthetaofx - y
    bac1 = np.dot(bac.T, X)
    bac2 = (1 / len(X)) * bac1 + lamb * weights
    return bac2


def pooled(X, y):
    logreg = LogisticRegression()
    logreg.fit(X, y.ravel())
    return logreg.coef_

    
def gd_for(winit, X, y, steps=100000, eta=0.01, tol=1e-6):
    wc = winit

    for i in range(steps):
        Gradient = gradient(wc, X, y)
        wc = wc - eta * gradient(wc, X, y)
        if gottol(Gradient, tol):
            break
    return wc, i


def gd_while(winit, X, y, steps=100, eta=0.01, tol=1e-6):
    wc = winit

    count = 0
    Gradient = gradient(wc, X, y)
    while not gottol(Gradient, tol):
        count = count + 1
        wc = wc - eta * Gradient
        Gradient = gradient(wc, X, y)
    return wc, count


def adam(w, X, y, steps=10000, eta=1e-2, tol=1e-8):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    wp = w
    mt, vt = 0, 0

    i = 0
    while True:
        i = i + 1
        grad = gradient(wp, X, y)

        mt = beta1 * mt + (1 - beta1) * grad
        vt = beta2 * vt + (1 - beta2) * (grad**2)

        m = mt / (1 - beta1**i)
        v = vt / (1 - beta2**i)

        wc = wp - eta * m / (np.sqrt(v) + eps)

        if np.linalg.norm(wp - wc) <= tol:
            break

        wp = wc

    return wc, i


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
    
    print("Pooled:", pooled(x_new, y_new))
    print("GD_For:", gd_for(w, x_new, y_new)[0])
    print("GD_While:", gd_while(w, x_new, y_new)[0])
    print("Adam: ", adam(w, x_new, y_new)[0])