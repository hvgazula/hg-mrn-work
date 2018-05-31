#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:02:23 2018

@author: Harshvardhan
"""
import numpy as np
import scipy.io as sio


def return_X_and_y():

    # Loading X data
    X = sio.loadmat('data/X.mat')
    X = X['X']

    # Loading y data
    y = sio.loadmat('data/labels.mat')
    y = y['labels']
    y = np.reshape(y, (869, ))

    return X, y
