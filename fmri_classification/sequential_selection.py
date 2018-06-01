#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:50:07 2018

@author: Harshvardhan
"""
import pandas as pd
import shelve
from load_data_from_mat import return_X_and_y
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def perform_sfs(main_model, X_train, X_test, y_train, y_test):
    sfs1 = SFS(
        logreg,
        k_features=10,
        verbose=1,
        forward=True,
        floating=False,
        scoring='accuracy',
        cv=0,
        n_jobs=-1)

    sfs1 = sfs1.fit(X_train, y_train)

    print('Selected features:', sfs1.k_feature_idx_)

    # Generate the new subsets based on the selected features
    # Note that the transform call is equivalent to
    # X_train[:, sfs1.k_feature_idx_]

    X_train_sfs = sfs1.transform(X_train)
    X_test_sfs = sfs1.transform(X_test)

    # Fit the estimator using the new feature subset
    # and make a prediction on the test data
    main_model.fit(X_train_sfs, y_train)
    y_pred = main_model.predict(X_test_sfs)

    # Compute the accuracy of the prediction
    acc = float((y_test == y_pred).sum()) / y_pred.shape[0]
    print('Test set accuracy: %.2f %%' % (acc * 100))

    return sfs1


if __name__ == '__main__':
    X, y = return_X_and_y()

    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.33, random_state=1)

    logreg = LogisticRegression()
    logreg_sfs = perform_sfs(logreg, X_train, X_test, y_train, y_test)

    simple_svm = LinearSVC()
    simple_svm_sfs = perform_sfs(simple_svm, X_train, X_test, y_train, y_test)

    radial_svm = SVC()
    radial_svm_sfs = perform_sfs(radial_svm, X_train, X_test, y_train, y_test)

    lda = LinearDiscriminantAnalysis()
    lda_sfs = perform_sfs(lda, X_train, X_test, y_train, y_test)

    print('Writing data to a shelve file')
    results = shelve.open(os.path.join('Results', file_name))
    results['logistic'] = logreg_sfs
    results['linear_svm'] = simple_svm_sfs
    results['radial_svm'] = radial_svm_sfs
    results.close()
