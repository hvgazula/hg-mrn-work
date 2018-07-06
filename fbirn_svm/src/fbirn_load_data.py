#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:32:34 2018

@author: hgazula
"""

import pickle
import pandas as pd
import statsmodels.api as sm


def select_and_drop_cols(site_dummy, site_data):
    """Select and crop columns"""
#    select_column_list = ['age', 'diagnosis']  # for singleshot

    # Below two lines for pooled and multishot
    select_column_list = [
        'age', 'diagnosis', 'site_07', 'site_09', 'site_10', 'site_12',
        'site_13', 'site_18'
    ]
    site_data = site_dummy.merge(site_data, on='site', how='right')
    site_data = site_data.drop('site', axis=1)
    site_X = site_data[select_column_list]
    site_y = site_data.drop(select_column_list, axis=1)
    return site_X, site_y


def get_dummies_and_augment(site_X):
    """Add a constant column and get dummies for categorical values"""
    X = pd.get_dummies(site_X, drop_first='True')
    X = sm.add_constant(X, has_constant='add')
    return X


def load_data():

    with open("final_data.pkl", "rb") as f:
        demographics, voxels = pickle.load(f)

    FinalData = pd.concat([demographics, voxels], axis=1)
    # 3 7 9 10 12 13 18

    site_01 = FinalData[FinalData['site'].str.match('03')]
    site_02 = FinalData[FinalData['site'].str.match('07')]
    site_03 = FinalData[FinalData['site'].str.match('09')]
    site_04 = FinalData[FinalData['site'].str.match('10')]
    site_05 = FinalData[FinalData['site'].str.match('12')]
    site_06 = FinalData[FinalData['site'].str.match('13')]
    site_07 = FinalData[FinalData['site'].str.match('18')]

    # send the total number of sites information to each site (Remote)
    unique_sites = FinalData['site'].unique()
    unique_sites.sort()
    site_dummy = pd.get_dummies(unique_sites, drop_first=True)
    site_dummy.set_index(unique_sites, inplace=True)
    site_dummy = site_dummy.add_prefix('site_')
    site_dummy['site'] = site_dummy.index

    site_01_X, site_01_y = select_and_drop_cols(site_dummy, site_01)
    site_02_X, site_02_y = select_and_drop_cols(site_dummy, site_02)
    site_03_X, site_03_y = select_and_drop_cols(site_dummy, site_03)
    site_04_X, site_04_y = select_and_drop_cols(site_dummy, site_04)
    site_05_X, site_05_y = select_and_drop_cols(site_dummy, site_05)
    site_06_X, site_06_y = select_and_drop_cols(site_dummy, site_06)
    site_07_X, site_07_y = select_and_drop_cols(site_dummy, site_07)

    site_01_y1 = site_01_y.values
    site_02_y1 = site_02_y.values
    site_03_y1 = site_03_y.values
    site_04_y1 = site_04_y.values
    site_05_y1 = site_05_y.values
    site_06_y1 = site_06_y.values
    site_07_y1 = site_07_y.values

    X1 = get_dummies_and_augment(site_01_X)
    X2 = get_dummies_and_augment(site_02_X)
    X3 = get_dummies_and_augment(site_03_X)
    X4 = get_dummies_and_augment(site_04_X)
    X5 = get_dummies_and_augment(site_05_X)
    X6 = get_dummies_and_augment(site_06_X)
    X7 = get_dummies_and_augment(site_07_X)

    column_name_list = X1.columns.tolist()

    X1 = X1.values
    X2 = X2.values
    X3 = X3.values
    X4 = X4.values
    X5 = X5.values
    X6 = X6.values
    X7 = X7.values

    site_01_y1 = site_01_y1.astype('float64')
    site_02_y1 = site_02_y1.astype('float64')
    site_03_y1 = site_03_y1.astype('float64')
    site_04_y1 = site_04_y1.astype('float64')
    site_05_y1 = site_05_y1.astype('float64')
    site_06_y1 = site_06_y1.astype('float64')
    site_07_y1 = site_07_y1.astype('float64')

    return (X1, site_01_y1, X2, site_02_y1, X3, site_03_y1, X4, site_04_y1,
            X5, site_05_y1, X6, site_06_y1, X7, site_07_y1,
            column_name_list)
