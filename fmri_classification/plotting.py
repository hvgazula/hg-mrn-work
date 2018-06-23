#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 11:43:51 2018

@author: Harshvardhan
"""
import seaborn as sns
import shelve
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.backends.backend_pdf

sns.set()


#pdf = matplotlib.backends.backend_pdf.PdfPages(
#    "Plots_DataSet_02_Features_100.pdf")

pdf = matplotlib.backends.backend_pdf.PdfPages(
    "test.pdf")

classifier_dict = {
    'knn': 'K Nearest Neighbors (K = 5)',
    'lda': 'Linear Discriminant Analysis',
    'logistic': 'Logistic Regression',
    'simple_svm': 'Linear SVM',
    'radial_svm': 'RBF SVM',
}

for key in classifier_dict.keys():
    try:
        file_list = sorted(
            [os.path.splitext(x)[0] for x in glob.glob(key + '/*.dat')])

        df_list = [shelve.open(file)[key] for file in file_list]

        main_list = []
        for index, df in enumerate(df_list):
            curr_series = df['feature_idx']
            my_list = []
            my_list.append(curr_series[1][0])
            for i in range(1, len(curr_series)):
                curr_elem = list(set(curr_series[i + 1]) - set(curr_series[i]))
                my_list.append(curr_elem[0])
            main_list.append(my_list)

        index_selection_order_fig = plt.figure()
        index_df = pd.DataFrame(main_list)
        for col in index_df.columns:
            y = index_df[col]
            x = np.ones(y.shape) * (col + 1)
            plt.scatter(x, y, s=5)
            plt.xlabel('Feature Selection Step')
            plt.ylabel('Feature Index')
            plt.title(classifier_dict[key] + ' (Features selected = 100)')

        pdf.savefig(index_selection_order_fig)
        plt.close(index_selection_order_fig)

        # Print Performance Criterion Plots
        performance_criterion = ['accuracy', 'f1_score']
        for criterion in performance_criterion:
            fig = plt.figure()
            curr_df = [df[criterion] for df in df_list]
            concat_df = pd.concat(curr_df, axis=1)
            concat_df.columns = list(range(1, len(concat_df.columns) + 1))

            for col in concat_df.columns:
                y = concat_df[col]
                x = np.ones(y.shape) * col
                plt.scatter(x, y)
                plt.xlabel('Run Index')
                plt.ylabel(criterion)
                plt.title(classifier_dict[key] + ' (Features selected = 100)')

            pdf.savefig(fig)
            plt.close(fig)

        # Print index at which maximum accuracy is achieved
            index_fig = plt.figure()
            y = concat_df.idxmax(axis=0)
            x = np.arange(1, len(y)+1)
            plt.scatter(x, y)
            plt.xlabel('Run Index')
            plt.ylabel('Step Max ' + criterion + ' was achieved')
            plt.title(classifier_dict[key] + ' (Features selected = 100)')
            pdf.savefig(index_fig)
            plt.close(index_fig)

    except Exception:
        pass

pdf.close()
