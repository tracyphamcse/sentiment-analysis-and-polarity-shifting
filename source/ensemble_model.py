import sys
import os

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

import pickle
from nltk.tokenize import word_tokenize

from file_io import load_files_in_directory, write_file_by_utf8
from config import *
from utils import display_general_help, display_general_domain_not_supported

def generate_label(n_class_1, n_class_2):
    return [0] * n_class_1 + [1] * n_class_2

def read_data(file_path_pos, file_path_neg):
    pos = load_files_in_directory(file_path_pos)
    neg = load_files_in_directory(file_path_neg)
    return pos, neg

def train(data, target, target_names, file_path):
    X_train = data
    y_train = target

    classifier = MODEL
    classifier.fit(X_train,y_train)

    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(file_path, 'wb') as f:
        pickle.dump(classifier, f)
    print 'Store model at ', file_path

def test_stacking(data, target, target_names, file_path):
    with open(file_path, 'rb') as f:
        classifier = pickle.load(f)

        y_pred = classifier.predict(data)

        acc = accuracy_score(target, y_pred)
        report = classification_report(target, y_pred, target_names=target_names)

        print 'Accuracy: ' , acc
        print report

def test_product_rule(data, target, target_names):
    y_pred = []
    for d in data:
        if (d[0] * d[2] * d[4] > d[1] * d[3] * d[5]):
            y_pred.append(0)
        else:
            y_pred.append(1)

    acc = accuracy_score(target, y_pred)
    report = classification_report(target, y_pred, target_names=target_names)

    print 'Accuracy: ' , acc
    print report

def extract(file_path_pos, file_path_neg, tfidf_pkl, classifier_pkl):

    pos, neg = read_data(file_path_pos, file_path_neg)
    n_pos = len(pos)
    n_neg = len(neg)

    data = pos + neg
    label = generate_label(n_pos, n_neg)

    tfidf = pickle.load(open(tfidf_pkl, 'rb'))
    tfidf_data = tfidf.transform(data)

    with open(classifier_pkl, 'rb') as f:
        classifier = pickle.load(f)
        predict = classifier.predict_proba(tfidf_data)
        return predict, label

def main(domains, types):
    for domain in domains:
        print domain

        target_names = ['positive', 'negative']

        print 'Training...'

        print 'Extract feature...'

        print 'Baseline feature...'
        baseline_feature_X_train, y_train = extract(DATA_TRAIN_FOLDER(domain, 'positive'), DATA_TRAIN_FOLDER(domain, 'negative'), TFIDF_BASE_PKL_FILE(domain), CLASSIFIER_BASE_PKL_FILE(domain))

        print 'Shift feature...'
        shift_feature_X_train, _ = extract(SHIFT_PART_TRAIN_FOLDER(domain, 'positive'), SHIFT_PART_TRAIN_FOLDER(domain, 'negative'), TFIDF_SHIFT_PKL_FILE(domain), CLASSIFIER_SHIFT_PKL_FILE(domain))

        print 'Unshift feature...'
        unshift_feature_X_train, _ = extract(UNSHIFT_PART_TRAIN_FOLDER(domain, 'positive'), UNSHIFT_PART_TRAIN_FOLDER(domain, 'negative'), TFIDF_UNSHIFT_PKL_FILE(domain), CLASSIFIER_UNSHIFT_PKL_FILE(domain,))

        X_train = []
        for i in range(len(y_train)):
            x = []
            x.extend(baseline_feature_X_train[i])
            x.extend(shift_feature_X_train[i])
            x.extend(unshift_feature_X_train[i])
            X_train.append(x)

        train(X_train, y_train,target_names , CLASSIFIER_ENSEMBLE_PKL_FILE(domain))

        print 'Testing...'

        print 'Extract feature...'

        print 'Baseline feature...'
        baseline_feature_X_test, y_test = extract(DATA_TEST_FOLDER(domain, 'positive'), DATA_TEST_FOLDER(domain, 'negative'), TFIDF_BASE_PKL_FILE(domain), CLASSIFIER_BASE_PKL_FILE(domain))

        print 'Shift feature...'
        shift_feature_X_test, _ = extract(SHIFT_PART_TEST_FOLDER(domain, 'positive'), SHIFT_PART_TEST_FOLDER(domain, 'negative'), TFIDF_SHIFT_PKL_FILE(domain), CLASSIFIER_SHIFT_PKL_FILE(domain))

        print 'Unshift feature...'
        unshift_feature_X_test, _ = extract(UNSHIFT_PART_TEST_FOLDER(domain, 'positive'), UNSHIFT_PART_TEST_FOLDER(domain, 'negative'), TFIDF_UNSHIFT_PKL_FILE(domain), CLASSIFIER_UNSHIFT_PKL_FILE(domain))

        X_test= []
        for i in range(len(y_test)):
            x = []
            x.extend(baseline_feature_X_test[i])
            x.extend(shift_feature_X_test[i])
            x.extend(unshift_feature_X_test[i])
            X_test.append(x)

        print 'Ensemble using Product Rule'
        test_product_rule (X_test, y_test, target_names)

        print 'Ensemble using Stacking Medthod'
        test_stacking(X_test, y_test, target_names, CLASSIFIER_ENSEMBLE_PKL_FILE(domain))

if __name__ == '__main__':
    set_args = set(sys.argv[1:])
    if not set_args:
        display_general_help(file_name='tfidf_model')
        sys.exit()

    # Choose the domain
    # -all to get all domains
    domains = ALL_DOMAINS
    if '-all' not in set_args:
        domains = list(set_args & set(ALL_DOMAINS))
    if not domains:
        display_general_domain_not_supported()
        sys.exit()

    main(domains, 'sentiment')
    print 'Done!'
