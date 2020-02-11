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
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fmin_tnc
from sklearn import linear_model


warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df


def gottol(v, tol=1e-6):
    return np.sum(np.square(v)) <= tol


def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))


def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(theta, x))


def cost_function(theta, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x)))
    return total_cost


def fit(x, y, theta):
    opt_weights = fmin_tnc(func=cost_function, x0=theta,
                   fprime=gradient,args=(x, y.flatten()))
    return opt_weights[0]


def fit1(theta, x, y, n_iterations=10000, w_=None, eta=0):
    m = x.shape[0]
    cost_ = 0
    for _ in range(n_iterations):
        y_pred = np.dot(x, w_)
        residuals = y_pred - y
        gradient_vector = np.dot(x.T, residuals)
        w_ -= (eta / m) * gradient_vector
        cost = np.sum((residuals ** 2)) / (2 * m)
        cost_.append(cost)
    return w_


# def gradient1(weights, X, y, lamb=0.0):
#     """Computes the gradient"""
#     if y.ndim == 1:
#         y = y.reshape(-1, 1)
    
#     hthetaofx = 1/(1 + np.exp(-1 * np.dot(X, weights.reshape(-1, 1))))
#     bac = hthetaofx - y
#     bac1 = np.dot(bac.T, X)
#     bac2 = (1 / len(X)) * bac1 + lamb * weights
#     return bac2


def net_input(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)


def gradient(theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,   x)) - y)


def pooled(X, y):
    logreg = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1e7, tol=1e-6)
    logreg.fit(X, y.ravel())
    return logreg.coef_


def pooled1(X, y):
    logreg = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=0, learning_rate='constant', eta0=0.01, tol=1e-6, max_iter=1e7)
    logreg.fit(X, y.ravel())
    return logreg.coef_
    
def gd_for(winit, X, y, steps=1e7, eta=0.01, tol=1e-6):
    wc = winit

    for i in range(int(steps)):
        Gradient = gradient(wc, X, y)
        wc = wc - eta * gradient(wc, X, y)
        if gottol(Gradient, tol):
            break
    return wc, i


# def gd_for1(winit, X, y, steps=1e7, eta=0.01, tol=1e-6):
#     wc = winit

#     for i in range(int(steps)):
#         Gradient = gradient1(wc, X, y)
#         wc = wc - eta * gradient1(wc, X, y)
#         if gottol(Gradient, tol):
#             break
#     return wc, i


def gd_while(winit, X, y, steps=1e7, eta=0.01, tol=1e-6):
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


def plot_data(admitted, not_admitted):
    # plots
    if isinstance(admitted, pd.DataFrame) and isinstance(not_admitted, pd.DataFrame):
        plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
        plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
    else:
        plt.scatter(admitted[:, 0], admitted[:, 1], s=10, label='Admitted')
        plt.scatter(not_admitted[:, 0], not_admitted[:, 1], s=10, label='Not Admitted')
        
    plt.legend()
    plt.show()
    

def clean_data():
    # load the data from the file
    data = load_data("data/marks.txt", None)

    # X = feature values, all the columns except the last column
    X = data.iloc[:, :-1]

    # y = target values, last column of the data frame
    y = data.iloc[:, -1]

    # filter out the applicants that got admitted
    admitted = data.loc[y == 1]

    # filter out the applicants that din't get admission
    not_admitted = data.loc[y == 0]
    
    plot_data(admitted, not_admitted)
    
    return X, y
  
    
def augment_data(X, y):
        
    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = y[:, np.newaxis]
    theta = np.zeros((X.shape[1], 1))
    
    return X, y, theta


if __name__ == "__main__":
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    
    x_new = x[y < 2, :2]
    y_new = y[y < 2]
    
    admitted = x_new[y_new == 0, :]
    not_admitted = x_new[y_new == 1, :]
    
    plot_data(admitted, not_admitted)
    
    x_new = np.c_[np.ones((x_new.shape[0], 1)), x_new]

    parameters = fit(x_new, y_new, theta)
    
    w = np.array([0., 0., 0., 0., 0.])
    
    print("Pooled:", pooled(x_new, y_new))
    print("Pooled:", pooled1(x_new, y_new))
    print("GD_For:", gd_for(w, x_new, y_new)[0])
    # print("GD_While:", gd_while(w, x_new, y_new)[0])
    # print("Adam: ", adam(w, x_new, y_new)[0])