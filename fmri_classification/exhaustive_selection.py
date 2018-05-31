#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 08:50:26 2018

@author: Harshvardhan
"""
import os
import pandas as pd
import shelve
import sys
from load_data_from_mat import return_X_and_y
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS


def make_predictions_on_test(efs_model, curr_model, X_train, X_test, y_train,
                             y_test, curr_idx):
    X_train_sfs = X_train[:, curr_idx]
    X_test_sfs = X_test[:, curr_idx]

    # Fit the estimator using the new feature subset
    # and make a prediction on the test data
    curr_model.fit(X_train_sfs, y_train)
    y_pred = curr_model.predict(X_test_sfs)

    # Compute the accuracy of the prediction
    test_acc = float((y_test == y_pred).sum()) / y_pred.shape[0]

    return test_acc


def perform_efs(curr_model, X, y, min_cols, max_cols):

    efs1 = EFS(
        curr_model,
        min_features=min_cols,
        max_features=max_cols,
        print_progress=True,
        scoring='accuracy',
        cv=5,
        n_jobs=-1)

    efs1 = efs1.fit(X, y)

    df = pd.DataFrame.from_dict(efs1.get_metric_dict()).T
#    df['test_acc'] = df['feature_idx'].apply(
#        lambda x: make_predictions_on_test(efs1, curr_model, X_train, X_test, y_train, y_test, x)
#    )

    return df


if __name__ == '__main__':
    X, y = return_X_and_y()

    min_features = int(sys.argv[1])
    max_features = int(sys.argv[2])

    file_name = 'Results_{:04}_to_{:04}'.format(min_features, max_features)

#    test_size = 0.3
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=test_size)

    logreg = LogisticRegression()
    logreg_efs = perform_efs(logreg, X, y, min_features, max_features)

    simple_svm = LinearSVC()
    simple_svm_efs = perform_efs(simple_svm, X, y, min_features, max_features)

    radial_svm = SVC()
    radial_svm_efs = perform_efs(radial_svm, X, y, min_features, max_features)

    # %% Write to a file
    results_folder = 'Results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    print('Writing data to a shelve file')
    results = shelve.open(os.path.join('Results', file_name))
    results['logistic'] = logreg_efs
    results['linear_svm'] = simple_svm_efs
    results['radial_svm'] = radial_svm_efs
    results.close()
