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

def read_sentiment_data(domain):
    pos = load_files_in_directory(SHIFT_PART_TRAIN_FOLDER(domain, 'positive'))
    neg = load_files_in_directory(SHIFT_PART_TRAIN_FOLDER(domain, 'negative'))
    return pos, neg

def read_sentiment_test_data(domain):
    pos = load_files_in_directory(SHIFT_PART_TEST_FOLDER(domain, 'positive'))
    neg = load_files_in_directory(SHIFT_PART_TEST_FOLDER(domain, 'negative'))
    return pos, neg

def train(data, target, target_names, file_path):

    # X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1234)

    X_train = data
    y_train = target

    classifier = MODEL
    classifier.fit(X_train,y_train)

    # y_pred = classifier.predict(X_test)
    #
    # acc = accuracy_score(y_test, y_pred)
    # report = classification_report(y_test, y_pred, target_names=target_names)
    #
    # print 'Accuracy: ' , acc
    # print report

    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(file_path, 'wb') as f:
        pickle.dump(classifier, f)
    print 'Store model at ', file_path

def test(data, target, target_names, file_path):
    with open(file_path, 'rb') as f:
        classifier = pickle.load(f)

        y_pred = classifier.predict(data)

        acc = accuracy_score(target, y_pred)
        report = classification_report(target, y_pred, target_names=target_names)

        print 'Accuracy: ' , acc
        print report

def main(domains):
    for domain in domains:
        print domain
        print 'Build classifier model for shift of {}'.format(domain)

        print 'Training...'
        pos, neg = read_sentiment_data(domain)
        n_pos = len(pos)
        n_neg = len(neg)

        data = pos + neg
        tfidf = pickle.load(open(TFIDF_SHIFT_PKL_FILE(domain), 'rb'))
        tfidf_data = tfidf.transform(data)
        target = generate_label(n_pos, n_neg)
        target_names = ['positive', 'negative']

        train(tfidf_data, target, target_names, CLASSIFIER_SHIFT_PKL_FILE(domain))

        print 'Testing...'
        pos, neg = read_sentiment_test_data(domain)
        n_pos = len(pos)
        n_neg = len(neg)

        data = pos + neg
        tfidf = pickle.load(open(TFIDF_SHIFT_PKL_FILE(domain), 'rb'))
        tfidf_data = tfidf.transform(data)
        target = generate_label(n_pos, n_neg)

        test(tfidf_data, target, target_names, CLASSIFIER_SHIFT_PKL_FILE(domain))


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

    main(domains)
    print 'Done!'
