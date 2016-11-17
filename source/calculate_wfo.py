from __future__ import division
import sys
import operator
import math

from nltk.tokenize import sent_tokenize, word_tokenize

from config import ALL_DOMAINS, LAMDA, DATA_TRAIN_FOLDER, RANKING_FILE
from utils import display_general_help, display_general_domain_not_supported
from document import load_documents_for_train, load_documents_for_test
from file_io import load_files_in_directory, write_file_by_utf8
from dictionaries import set_adjectives, stopwords


def create_dict(documents):
    dictionary = {}
    for document in documents:
        words = extract_word_in_document(document)
        words = [word.lower() for word in words if (word.lower() not in stopwords)]
        words = set(words)
        # words = set(words) & set(set_adjectives)
        # Add new word to dict or increase value of word
        for word in words:
            if dictionary.has_key(word):
                dictionary[word] += 1
            else:
                dictionary[word] = 1
    return dictionary

def extract_word_in_document(document):
    sentences = sent_tokenize(document)
    words = []
    for sentence in sentences:
        words.extend(word_tokenize(sentence))
    return words

def WFO(prob_in_pos, prob_in_neg):
    return pow(prob_in_pos,LAMDA) * pow(max(0,math.log(prob_in_pos/prob_in_neg)),1-LAMDA)

def save_ranking(file_path, dict):
    content = ''
    for word in dict:
        content += word[0] + ' ' + str(word[1]) + '\n'
    write_file_by_utf8(file_path, content)

def main(domains):
    for domain in set(domains):
        print domain
        print 'Creating dictionary...'
        # Read each file in positive training data
        training_data_pos = load_files_in_directory(DATA_TRAIN_FOLDER(domain,'positive'))
        number_of_pos = len(training_data_pos) * 1.0
        dict_pos = create_dict(training_data_pos)

        # Read each file in negative training data
        training_data_neg = load_files_in_directory(DATA_TRAIN_FOLDER(domain,'negative'))
        number_of_neg = len(training_data_neg) * 1.0
        dict_neg = create_dict(training_data_neg)

        print 'Calculating WFO...'
        top_ranked_pos = {}
        top_ranked_neg = {}
        for word in dict_pos:
            if dict_neg.has_key(word):
                freq_word_in_pos = dict_pos[word]
                freq_word_in_neg = dict_neg[word]
                if freq_word_in_pos + freq_word_in_neg >= 3:
                    freq_word_in_pos /= number_of_pos
                    freq_word_in_neg /= number_of_neg
                    top_ranked_pos[word] = WFO(freq_word_in_pos, freq_word_in_neg)
                    top_ranked_neg[word] = WFO(freq_word_in_neg, freq_word_in_pos)

        ranking_pos = sorted(top_ranked_pos.items(), key=operator.itemgetter(1), reverse=True)
        ranking_neg = sorted(top_ranked_neg.items(), key=operator.itemgetter(1), reverse=True)

        print 'Writing results ...'
        save_ranking(RANKING_FILE(domain,'positive'), ranking_pos)
        save_ranking(RANKING_FILE(domain,'negative'), ranking_neg)
        print

if __name__ == '__main__':
    set_args = set(sys.argv[1:])
    if not set_args:
        display_general_help(file_name='calculate_wfo')
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
