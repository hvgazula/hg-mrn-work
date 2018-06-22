#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:50:07 2018

@author: Harshvardhan
"""
import pandas as pd
import socket
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from utils import return_X_and_y, write_results
import logging


def get_test_score(X_train, X_test, y_train, y_test, curr_feature_set,
                   main_model):
    X_train_sfs = X_train[:, curr_feature_set]
    X_test_sfs = X_test[:, curr_feature_set]

    # Fit the estimator using the new feature subset
    # and make a prediction on the test data
    main_model.fit(X_train_sfs, y_train)
    y_pred = main_model.predict(X_test_sfs)

    keys = {'accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix'}
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    return dict(zip(keys, [accuracy, precision, recall, f1score, confusion]))


def perform_sfs(curr_classifier, X_train, X_test, y_train, y_test):
    sfs1 = SFS(
        curr_classifier,
        k_features=100,
        verbose=0,
        forward=True,
        floating=False,
        scoring='accuracy',
        cv=5,
        n_jobs=8)

    sfs1 = sfs1.fit(X_train, y_train)
    df = pd.DataFrame.from_dict(sfs1.get_metric_dict(), orient='index')
    df[[
        'accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix'
    ]] = df['feature_idx'].apply(
        lambda x: get_test_score(X_train, X_test, y_train, y_test, x, curr_classifier)
    ).apply(pd.Series)

    return df


if __name__ == '__main__':
    data_folder = int(sys.argv[1])
    run_number = int(sys.argv[2])

    logging.basicConfig(
        level=logging.DEBUG,
        filename="Results_Data_Part_02_20180622.log",
        filemode="a+",
        format="%(asctime)-15s %(levelname)-8s %(message)s")

    folder_tag = '{:02}-{:03}'.format(data_folder, run_number)
    host = socket.gethostname()
    
#    logging.info("Running with 100 features, includes kNN and excludes LDA")
#    logging.info("Results are in folder named Results_20180622")

    logging.info('Run:{:03}, Host:{:10} Loading data'.format(
        run_number, host))
    X, y = return_X_and_y('Part_{:02}'.format(data_folder))

    logging.info('Run:{:03}, Host:{:10} Splitting data'.format(
        run_number, host))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    logging.info('Run:{:03}, Host:{:10} Running Logistic Regression'.format(run_number, host))
    logreg = LogisticRegression()
    logreg_sfs = perform_sfs(logreg, X_train, X_test, y_train, y_test)
    logging.info('Run:{:03}, Host:{:10} Writing output Logistic Regression'.format(
        run_number, host))
    write_results(logreg_sfs, 'logistic', folder_tag)

    logging.info('Run:{:03}, Host:{:10} Running Linear SVM'.format(run_number, host))
    simple_svm = LinearSVC()
    simple_svm_sfs = perform_sfs(simple_svm, X_train, X_test, y_train, y_test)
    logging.info(
        'Run:{:03}, Host:{:10} Writing output Linear SVM'.format(run_number, host))
    write_results(simple_svm_sfs, 'simple_svm', folder_tag)

    logging.info('Run:{:03}, Host:{:10} Running Radial SVM'.format(run_number, host))
    radial_svm = SVC()
    radial_svm_sfs = perform_sfs(radial_svm, X_train, X_test, y_train, y_test)
    logging.info(
        'Run:{:03}, Host:{:10} Writing ouput Radial SVM'.format(run_number, host))
    write_results(radial_svm_sfs, 'radial_svm', folder_tag)

    logging.info('Run:{:03}, Host:{:10} Running kNN'.format(run_number, host))
    knn = KNeighborsClassifier()
    knn_sfs = perform_sfs(knn, X_train, X_test, y_train, y_test)
    logging.info(
        'Run:{:03}, Host:{:10} Writing ouput kNN'.format(run_number, host))
    write_results(knn_sfs, 'knn', folder_tag)
