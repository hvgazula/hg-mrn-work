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


def extract_feature_idx_order(df):
    main_list = []
    for index, df in enumerate(df_list):
        curr_series = df['feature_idx']
        my_list = []
        my_list.append(curr_series[1][0])
        for i in range(1, len(curr_series)):
            curr_elem = list(set(curr_series[i + 1]) - set(curr_series[i]))
            my_list.append(curr_elem[0])
        main_list.append(my_list)

    return main_list


if __name__ == '__main__':

    sns.set()

    output_file = 'test.pdf'
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_file)

    classifier_dict = {
        'knn': 'K Nearest Neighbors (K = 5)',
        'lda': 'Linear Discriminant Analysis',
        'logistic': 'Logistic Regression',
        'simple_svm': 'Linear SVM',
        'radial_svm': 'RBF SVM',
    }

    # Folder List
    select_dir = os.path.join('/Users/Harshvardhan/Downloads', 'Results*')
    dir_contents = glob.glob(select_dir)
    results_folders = filter(os.path.isdir, dir_contents)

    for folder in results_folders:
        results_code = folder.split('_')[1]
        data_set, features_selected, runs = results_code.split('-')

        for key in classifier_dict.keys():
            try:
                sub_directory = os.path.join(folder, key, '*.dat')
                sub_dir_files = glob.glob(sub_directory)

                if not sub_dir_files:
                    continue

                files_without_ext = map(os.path.splitext, sub_dir_files)
                file_list = sorted([x[0] for x in files_without_ext])

                df_list = [shelve.open(file)[key] for file in file_list]

                feature_idx_order = extract_feature_idx_order(df_list)

                feature_idx_order_fig = plt.figure()
                index_df = pd.DataFrame(feature_idx_order)
                for col in index_df.columns:
                    y = index_df[col]
                    x = np.ones(y.shape) * (col + 1)
                    plt.scatter(x, y, s=5)
                    plt.xlabel('Feature Selection Step')
                    plt.ylabel('Feature Index')
                    plt.title(
                        '{} (Dataset = {}, Features selected = {})'.format(
                            classifier_dict[key], data_set, features_selected))

                pdf.savefig(feature_idx_order_fig)
                plt.close(feature_idx_order_fig)

                # Print Performance Criterion Plots
                performance_criterion = ['accuracy', 'f1_score']
                for criterion in performance_criterion:
                    fig = plt.figure()
                    curr_df = [df[criterion] for df in df_list]
                    concat_df = pd.concat(curr_df, axis=1)
                    concat_df.columns = list(
                        range(1, len(concat_df.columns) + 1))

                    for col in concat_df.columns:
                        y = concat_df[col]
                        x = np.ones(y.shape) * col
                        plt.scatter(x, y)
                        plt.xlabel('Run Index')
                        plt.ylabel(criterion)
                        plt.title('{} (Dataset = {}, Features selected = {})'.
                                  format(classifier_dict[key], data_set,
                                         features_selected))

                    pdf.savefig(fig)
                    plt.close(fig)

                    # Print index at which maximum performance is achieved
                    index_fig = plt.figure()
                    y = concat_df.idxmax(axis=0)
                    x = np.arange(1, len(y) + 1)
                    plt.scatter(x, y)
                    plt.xlabel('Run Index')
                    plt.ylabel('Step Max ' + criterion + ' was achieved')
                    plt.title(
                        '{} (Dataset = {}, Features selected = {})'.format(
                            classifier_dict[key], data_set, features_selected))
                    pdf.savefig(index_fig)
                    plt.close(index_fig)

            except Exception as e:
                print(e)
                pass

    pdf.close()
