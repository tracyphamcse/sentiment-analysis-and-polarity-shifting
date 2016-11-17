import numpy as np
import pickle
import sys
import time
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

from file_io import load_files_in_directory, write_file_by_utf8, read_file_by_utf8
from config import *
from utils import display_general_help, display_general_domain_not_supported

def generate_tfidf(data, file_path):
    print 'Extracting feature...'
    vectorizer = TfidfVectorizer(decode_error='ignore',binary=True, ngram_range=N_GRAM)
    vectorizer.fit(data)

    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(file_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print 'Store at ', file_path


def read_shifting_data(domain):
    shift_pos = read_file_by_utf8(SHIFT_FILE(domain,'positive')).split('\n')
    shift_neg = read_file_by_utf8(SHIFT_FILE(domain,'negative')).split('\n')
    unshift_pos = read_file_by_utf8(UNSHIFT_FILE(domain,'positive')).split('\n')
    unshift_neg = read_file_by_utf8(UNSHIFT_FILE(domain,'negative')).split('\n')

    print len(shift_pos), len(shift_neg), len(unshift_pos), len(unshift_neg)
    return shift_pos + shift_neg + unshift_pos + unshift_neg

def read_sentiment_data(domain):
    pos = load_files_in_directory(DATA_TRAIN_FOLDER(domain, 'positive'))
    neg = load_files_in_directory(DATA_TRAIN_FOLDER(domain, 'negative'))
    return pos + neg

def main(domains, types):
    for domain in domains:
        print domain
        print 'Build tfidf model for {} of {}'.format(types, domain)

        if (types == 'detect'):
            data = read_shifting_data(domain)
            generate_tfidf(data, TFIDF_DETECT_PKL_FILE(domain))
        else:
            data = read_sentiment_data(domain)
            generate_tfidf(data, TFIDF_BASE_PKL_FILE(domain))



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

    # Choose the data to build TFID model: shift-unshift or sentiment data
    # defaul: sentiment
    if '-detect' in set_args:
        types = 'detect'
    else:
        types = 'base'

    main(domains, types)
    print 'Done!'
