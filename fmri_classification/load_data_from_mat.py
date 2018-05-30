#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:02:23 2018

@author: Harshvardhan
"""

import numpy as np
import scipy.io as sio


def return_X_and_y():

    # =============================================================================
    #     ## Loading X data
    #     f = h5py.File('data/X.mat')
    #     X = np.array(f["MixingMatrix_Weights"])
    #     print('Actual shape of X is: ', X.shape)
    #
    #     # Reshaping the X matrix
    #     X = np.reshape(X, (869, -1), order='C')
    #     print('After reshaping :', X.shape)
    # =============================================================================

    # Loading X data
    X = sio.loadmat('data/X.mat')
    X = X['X']
    print('Shape of X is: ', X.shape)

    # Loading y data
    y = sio.loadmat('data/labels.mat')
    y = y['labels']
    print('Shape of y is: ', y.shape)
    y = np.reshape(y, (869, ))

    return X, y
