#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:02:23 2018

@author: Harshvardhan
"""
import numpy as np
import os
import scipy.io as sio


def return_X_and_y(folder_name):
    """Loads the data relevant to the folder specified"""
    if folder_name == 'Part_01':
        X = sio.loadmat(
            os.path.join('data', folder_name, 'X.mat'))['X']  # Loading X data
        y = sio.loadmat(os.path.join('data', folder_name, 'labels.mat'))[
            'labels']  # Loading y data
    elif folder_name == 'Part_02':
        data = sio.loadmat(
            os.path.join('data', folder_name, 'FNC_residual.mat'))
        X = data['residuals']  # Loading X data
        y = data['diagnosis']  # Loading y data
    else:
        print('Folder not found.')

    y = np.reshape(y, (869, ))
    return X, y
