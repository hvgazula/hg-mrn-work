#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 08:50:26 2018

@author: Harshvardhan
"""
import pandas as pd
from load_data_from_mat import return_X_and_y
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS


def perform_efs(curr_model, X, y):

    efs1 = EFS(curr_model,
               min_features=1,
               max_features=X.shape[1],
               scoring='accuracy',
               print_progress=True,
               cv=5,
               n_jobs=-1)

    efs1 = efs1.fit(X, y)

    print('Best accuracy score: %.2f' % efs1.best_score_)
    print('Best subset:', efs1.best_idx_)

    df = pd.DataFrame.from_dict(efs1.get_metric_dict()).T
    df.sort_values('avg_score', inplace=True, ascending=False)

    return efs1


X, y = return_X_and_y()

knn = KNeighborsClassifier(n_neighbors=3)
knn_efs = perform_efs(knn, X, y)

logreg = LogisticRegression()
logreg_efs = perform_efs(logreg, X, y)

simple_svm = LinearSVC()
simple_svm_efs = perform_efs(simple_svm, X, y)

radial_svm = SVC()
radial_svm_efs = perform_efs(radial_svm, X, y)
