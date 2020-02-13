#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 10:23:38 2020

@author: hgazula
"""

from sklearn import datasets
import numpy as np

np.seterr(all='ignore')


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient(X, y, theta):
    z = np.dot(X, theta)
    h = sigmoid(z)
    return np.dot(X.T, (h - y)) / y.size


def loss(X, y, theta):
    z = np.dot(X, theta)
    h = sigmoid(z)
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)
    
iris = datasets.load_iris()
X = iris.data[:, :2]
X = add_intercept(X)
y = (iris.target != 0) * 1

num_iter = 300000
lr = 0.1

# for loop
theta = np.zeros((X.shape[1], 1))
for i in range(num_iter):
    z = np.dot(X, theta)
    h = sigmoid(z)
    grad = np.dot(X.T, (h - y)) / y.size
    prev_loss = loss(X, y, theta)
    theta -= lr * grad
    curr_loss = loss(X, y, theta)
print(theta)
        
# while loop
theta = np.zeros(X.shape[1])
i = 0
while True and i < num_iter:
    i = i + 1
    z = np.dot(X, theta)
    h = sigmoid(z)
    grad = np.dot(X.T, (h - y)) / y.size
    prev_loss = loss(X, y, theta)
    theta -= lr * grad
    curr_loss = loss(X, y, theta)
    
    ldiff = abs(curr_loss-prev_loss)
    # print(ldiff)
    if ldiff <= 1e-8 or np.isnan(ldiff):
        break
print(theta)
    
# adam
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
mt, vt = 0, 0
tol = 2e-8
lr = 0.01
i = 0
theta = np.zeros(X.shape[1])
while True and i < num_iter:
    i = i + 1

    grad = gradient(X, y, theta)
    mt = beta1 * mt + (1 - beta1) * grad
    vt = beta2 * vt + (1 - beta2) * (grad**2)
    m = mt / (1 - beta1**i)
    v = vt / (1 - beta2**i)
    prev_loss = loss(X, y, theta)
    theta = theta - lr * m / (np.sqrt(v) + eps)
    curr_loss = loss(X, y, theta) 
    print(curr_loss)
    
    ldiff = abs(curr_loss-prev_loss)
    if ldiff <= 1e-8 or np.isnan(ldiff):
        break
print(theta)