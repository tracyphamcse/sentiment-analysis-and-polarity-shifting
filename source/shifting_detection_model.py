import sys
import os

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

import pickle
from nltk.tokenize import word_tokenize, sent_tokenize

from file_io import load_files_in_directory, write_file_by_utf8, read_file_by_utf8
from config import *
from utils import display_general_help, display_general_domain_not_supported
from generate_shifting_training_data import ClauseSplitter
from dictionaries import clause_indicator

def generate_label(n_class_1, n_class_2):
    return [0] * n_class_1 + [1] * n_class_2

def read_shifting_data(domain):
    shift_pos = read_file_by_utf8(SHIFT_FILE(domain,'positive')).split('\n')
    shift_neg = read_file_by_utf8(SHIFT_FILE(domain,'negative')).split('\n')

    unshift_pos = read_file_by_utf8(UNSHIFT_FILE(domain,'positive')).split('\n')
    unshift_neg = read_file_by_utf8(UNSHIFT_FILE(domain,'negative')).split('\n')

    return shift_pos + shift_neg, unshift_pos + unshift_neg

def train(data, target, target_names, file_path):

    X_train, _, y_train, _= train_test_split(data, target, test_size=0, random_state=1234)

    X_test = data
    y_test = target

    classifier = MODEL
    classifier.fit(X_train,y_train)

    y_pred = classifier.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    print 'Accuracy: ' , acc
    print report

    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(file_path, 'wb') as f:
        pickle.dump(classifier, f)
    print 'Store model at ', file_path

def test(data, file_path):
    with open(file_path, 'rb') as f:
        classifier = pickle.load(f)
        y_pred = classifier.predict(data)
        return y_pred

def init_data(folder_path):
    data = []
    file_contents = load_files_in_directory(folder_path)
    for content in file_contents:
        sentences = sent_tokenize(content)
        temp = []
        for sent in sentences:
            temp.extend(ClauseSplitter().split_sentence(sent,clause_indicator))
        data.append(temp)
    return data

def detect_polarity_train(domain, polarity, file_path, tfidf):
    data = init_data(DATA_TRAIN_FOLDER(domain, polarity))
    for i in range(len(data)):
        doc = data[i]

        shift = ''
        unshift = ''

        tfidf_data = tfidf.transform(doc)
        y_pred = test(tfidf_data, file_path)

        for j in range(len(y_pred)):
            if y_pred[j] == 0:
                shift += doc[j]
            else:
                unshift += doc[j]

        write_file_by_utf8(SHIFT_PART_TRAIN_FILE(domain, polarity, i), shift)
        write_file_by_utf8(UNSHIFT_PART_TRAIN_FILE(domain, polarity, i), unshift)

def detect_polarity_test(domain, polarity, file_path, tfidf):
    data = init_data(DATA_TEST_FOLDER(domain, polarity))
    for i in range(len(data)):
        doc = data[i]

        shift = ''
        unshift = ''

        tfidf_data = tfidf.transform(doc)
        y_pred = test(tfidf_data, file_path)

        for j in range(len(y_pred)):
            if y_pred[j] == 0:
                shift += doc[j]
            else:
                unshift += doc[j]

        write_file_by_utf8(SHIFT_PART_TEST_FILE(domain, polarity, i), shift)
        write_file_by_utf8(UNSHIFT_PART_TEST_FILE(domain, polarity, i), unshift)

def main(domains):
    for domain in domains:
        print domain
        print 'Build classifier model for polarity shifting detection of {}'.format(domain)

        print 'Training...'
        shift, unshift = read_shifting_data(domain)
        n_shift = len(shift)
        n_unshift = len(unshift)

        data = shift + unshift
        tfidf = pickle.load(open(TFIDF_DETECT_PKL_FILE(domain), 'rb'))
        tfidf_data = tfidf.transform(data)
        target = generate_label(n_shift, n_unshift)
        target_names = ['shift', 'unshift']

        train(tfidf_data, target, target_names, CLASSIFIER_DETECT_PKL_FILE(domain))

        print 'Polarity Detecting....'

        detect_polarity_train(domain,'positive',CLASSIFIER_DETECT_PKL_FILE(domain),tfidf)
        detect_polarity_train(domain,'negative',CLASSIFIER_DETECT_PKL_FILE(domain),tfidf)

        detect_polarity_test(domain,'positive',CLASSIFIER_DETECT_PKL_FILE(domain),tfidf)
        detect_polarity_test(domain,'negative',CLASSIFIER_DETECT_PKL_FILE(domain),tfidf)


if __name__ == '__main__':
    set_args = set(sys.argv[1:])
    if not set_args:
        display_general_help(file_name='shifting_detection_model')
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
