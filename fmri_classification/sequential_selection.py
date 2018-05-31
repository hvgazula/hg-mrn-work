#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:50:07 2018

@author: Harshvardhan
"""
from load_data_from_mat import return_X_and_y
from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def perform_sfs(main_model, X_train, X_test, y_train, y_test):
    sfs1 = SFS(
        main_model,
        k_features=X.shape[1],
        verbose=1,
        forward=True,
        floating=False,
        scoring='accuracy',
        cv=5,
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


X, y = return_X_and_y()
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

#knn = KNeighborsClassifier(n_neighbors=3)
#knn_sfs = perform_sfs(knn, X_train, X_test, y_train, y_test)

logreg = LogisticRegression()
logreg_sfs = perform_sfs(logreg, X_train, X_test, y_train, y_test)

simple_svm = LinearSVC()
simple_svm_sfs = perform_sfs(simple_svm, X_train, X_test, y_train, y_test)

radial_svm = SVC()
radial_svm_sfs = perform_sfs(radial_svm, X_train, X_test, y_train, y_test)

lda = LinearDiscriminantAnalysis()
lda_sfs = perform_sfs(lda, X_train, X_test, y_train, y_test)