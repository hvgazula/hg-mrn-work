# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:42:53 2019

@author: hgazula
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adam optimizer with no hot-start
"""
from numba import jit, prange
import numpy as np
from sklearn import datasets


#@jit(nopython=True)
def gottol(vector, tol=1e-5):
    """Check if the gradient meets the tolerances"""
    return np.sum(np.square(vector)) <= tol


#@jit(nopython=True)
def objective(weights, X, y, lamb=0.0):
    """calculates the Objective function value"""
    return (1 / 2 * len(X)) * np.sum(
        (np.dot(X, weights) - y)**2) + lamb * np.linalg.norm(weights) / 2.


#@jit(nopython=True)
def gradient(weights, X, y, lamb=0.0):
    """Computes the gradient"""
#    return (1 / len(X)) * np.dot(X.T, np.dot(X, weights) - y) + lamb * weights
    hthetaofx = 1/(1 + np.exp(-1 * np.dot(X, weights.reshape(-1, 1))))
    return (1 / len(X)) * np.dot(X.T, hthetaofx - y) + lamb * weights


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

    tol = 1e-8
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


#folder_index = input('Enter the name of the folder to save results: ')
#folder_name = folder_index.replace(' ', '_')
#if not os.path.exists(folder_name):
#    os.makedirs(folder_name)
#
#X1, site_01_y1, X2, site_02_y1, X3, site_03_y1, X4, site_04_y1, column_name_list = load_data(
#)
#
#(params, sse, tvalues, rsquared, dof_global) = multishot_gd(
#    X1, site_01_y1, X2, site_02_y1, X3, site_03_y1, X4, site_04_y1)
#
#ps_global = 2 * sp.stats.t.sf(np.abs(tvalues), dof_global)
#pvalues = pd.DataFrame(ps_global.transpose(), columns=column_name_list)
#sse = pd.DataFrame(sse.transpose(), columns=['sse'])
#params = pd.DataFrame(params.transpose(), columns=column_name_list)
#tvalues = pd.DataFrame(tvalues.transpose(), columns=column_name_list)
#rsquared = pd.DataFrame(rsquared.transpose(), columns=['rsquared_adj'])
#
## %% Write to a file
#print('Writing data to a shelve file')
#results = shelve.open(
#    os.path.join(folder_name, 'multishotAdam_results'))
#results['params'] = params
#results['sse'] = sse
#results['pvalues'] = pvalues
#results['tvalues'] = tvalues
#results['rsquared'] = rsquared
#results.close()