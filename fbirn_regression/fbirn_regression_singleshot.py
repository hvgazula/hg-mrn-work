#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 10:12:49 2017

@author: Harshvardhan Gazula
@notes : Contains a variant of the single-shot regression proposed by Eswar.
         Drop site specific columns at each site and use global_beta_vector
         to calculate SSE at each local site.
@modified: 01/14/2018 weighted average single shot regression
"""

import os
import shelve
import numpy as np
from numba import jit, prange
import pandas as pd
import scipy as sp
from fbirn_load_data import load_data


@jit(nopython=True)
def singleshot_regression(X1, site_01_y1, X2, site_02_y1, X3, site_03_y1, X4,
                          site_04_y1, X5, site_05_y1, X6, site_06_y1, X7,
                          site_07_y1):

    size_y = site_01_y1.shape[1]

    params = np.zeros((X1.shape[1], size_y))
    sse = np.zeros(size_y)
    tvalues = np.zeros((X1.shape[1], size_y))
    rsquared = np.zeros(size_y)

    for voxel in prange(size_y):
        print(voxel)

        y1 = site_01_y1[:, voxel]
        y2 = site_02_y1[:, voxel]
        y3 = site_03_y1[:, voxel]
        y4 = site_04_y1[:, voxel]
        y5 = site_05_y1[:, voxel]
        y6 = site_06_y1[:, voxel]
        y7 = site_07_y1[:, voxel]

        # Start of single shot regression
        beta1 = np.linalg.inv(X1.T @ X1) @ (X1.T @ y1)
        beta2 = np.linalg.inv(X2.T @ X2) @ (X2.T @ y2)
        beta3 = np.linalg.inv(X3.T @ X3) @ (X3.T @ y3)
        beta4 = np.linalg.inv(X4.T @ X4) @ (X4.T @ y4)
        beta5 = np.linalg.inv(X5.T @ X5) @ (X5.T @ y5)
        beta6 = np.linalg.inv(X6.T @ X6) @ (X6.T @ y6)
        beta7 = np.linalg.inv(X7.T @ X7) @ (X7.T @ y7)

        # PART 02 - Aggregating parameter values at the remote
        count_y_local = np.array(
            [len(y1),
             len(y2),
             len(y3),
             len(y4),
             len(y5),
             len(y6),
             len(y7)])

        # Weighted Average
        count_y_local_float = count_y_local.astype(np.float64)
        sum_params = np.column_stack((beta1, beta2, beta3, beta4, beta5, beta6,
                                      beta7))
        avg_beta_vector = sum_params @ count_y_local_float / np.sum(
            count_y_local)

        # =============================================================================
        #         # Simple Average
        #         avg_beta_vector = (beta1 + beta2 + beta3 + beta4) / 4
        # =============================================================================

        params[:, voxel] = avg_beta_vector

        # PART 03 - SSE at each local site
        y1_estimate = np.dot(avg_beta_vector, X1.T)
        y2_estimate = np.dot(avg_beta_vector, X2.T)
        y3_estimate = np.dot(avg_beta_vector, X3.T)
        y4_estimate = np.dot(avg_beta_vector, X4.T)
        y5_estimate = np.dot(avg_beta_vector, X5.T)
        y6_estimate = np.dot(avg_beta_vector, X6.T)
        y7_estimate = np.dot(avg_beta_vector, X7.T)

        sse1 = np.linalg.norm(y1 - y1_estimate)**2
        sse2 = np.linalg.norm(y2 - y2_estimate)**2
        sse3 = np.linalg.norm(y3 - y3_estimate)**2
        sse4 = np.linalg.norm(y4 - y4_estimate)**2
        sse5 = np.linalg.norm(y5 - y5_estimate)**2
        sse6 = np.linalg.norm(y6 - y6_estimate)**2
        sse7 = np.linalg.norm(y7 - y7_estimate)**2

        # At Local
        mean_y_local = np.array([
            np.mean(y1),
            np.mean(y2),
            np.mean(y3),
            np.mean(y4),
            np.mean(y5),
            np.mean(y6),
            np.mean(y7)
        ])

        # At Remote
        mean_y_global = np.sum(
            mean_y_local * count_y_local) / np.sum(count_y_local)

        # At Local
        sst1 = np.sum(np.square(y1 - mean_y_global))
        sst2 = np.sum(np.square(y2 - mean_y_global))
        sst3 = np.sum(np.square(y3 - mean_y_global))
        sst4 = np.sum(np.square(y4 - mean_y_global))
        sst5 = np.sum(np.square(y5 - mean_y_global))
        sst6 = np.sum(np.square(y6 - mean_y_global))
        sst7 = np.sum(np.square(y7 - mean_y_global))

        cov1 = X1.T @ X1
        cov2 = X2.T @ X2
        cov3 = X3.T @ X3
        cov4 = X4.T @ X4
        cov5 = X5.T @ X5
        cov6 = X6.T @ X6
        cov7 = X7.T @ X7

        # PART 05 - Finding rsquared (global)
        SSE_global = sse1 + sse2 + sse3 + sse4 + sse5 + sse6 + sse7
        sse[voxel] = SSE_global
        SST_global = sst1 + sst2 + sst3 + sst4 + sst5 + sst6 + sst7
        r_squared_global = 1 - (SSE_global / SST_global)
        rsquared[voxel] = r_squared_global

        # PART 04 - Finding p-value at the Remote
        varX_matrix_global = cov1 + cov2 + cov3 + cov4 + cov5 + cov6 + cov7

        dof_global = np.sum(count_y_local) - len(avg_beta_vector)

        MSE = SSE_global / dof_global
        var_covar_beta_global = MSE * np.linalg.inv(varX_matrix_global)
        se_beta_global = np.sqrt(np.diag(var_covar_beta_global))
        ts_global = avg_beta_vector / se_beta_global

        tvalues[:, voxel] = ts_global

    return (params, sse, tvalues, rsquared, dof_global)


folder_index = input('Enter the name of the folder to save results: ')
folder_name = folder_index.replace(' ', '_')
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

X1, site_01_y1, X2, site_02_y1, X3, site_03_y1, X4, site_04_y1, X5, site_05_y1, X6, site_06_y1, X7, site_07_y1, column_name_list = load_data(
)

(params, sse, tvalues, rsquared, dof_global) = singleshot_regression(
    X1, site_01_y1, X2, site_02_y1, X3, site_03_y1, X4, site_04_y1, X5,
    site_05_y1, X6, site_06_y1, X7, site_07_y1)

ps_global = 2 * sp.stats.t.sf(np.abs(tvalues), dof_global)
pvalues = pd.DataFrame(ps_global.transpose(), columns=column_name_list)
sse = pd.DataFrame(sse.transpose(), columns=['sse'])
params = pd.DataFrame(params.transpose(), columns=column_name_list)
tvalues = pd.DataFrame(tvalues.transpose(), columns=column_name_list)
rsquared = pd.DataFrame(rsquared.transpose(), columns=['rsquared_adj'])

# %% Writing to a file
print('Writing data to a shelve file')
results = shelve.open(os.path.join(folder_name, 'singleshot'))
results['params'] = params
results['sse'] = sse
results['pvalues'] = pvalues
results['tvalues'] = tvalues
results['rsquared'] = rsquared
results.close()
