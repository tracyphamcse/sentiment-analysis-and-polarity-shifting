import sys

from nltk.tokenize import sent_tokenize, word_tokenize

from config import ALL_DOMAINS, N_MAX, RANKING_FILE, DATA_TRAIN_FOLDER, SHIFT_FILE, UNSHIFT_FILE
from utils import display_general_help, display_general_domain_not_supported
from file_io import read_file_by_utf8, write_file_by_utf8, load_files_in_directory
from dictionaries import clause_indicator

import random

class RankedWord(object):
    """docstring for RankedWord"""
    def __init__(self, w, r):
        super(RankedWord, self).__init__()
        self.word = w
        self.rank = float(r)

def getRankedWord(line):
    temp = line.split(' ')
    return RankedWord(temp[0], temp[1])

def getRankedFeature(domain):
    rankingpos = read_file_by_utf8(RANKING_FILE(domain, 'positive')).split('\n')
    rankingneg = read_file_by_utf8(RANKING_FILE(domain, 'negative')).split('\n')

    rankingpos = [ranking for ranking in rankingpos if ranking]
    rankingneg = [ranking for ranking in rankingneg if ranking]

    tPosCurrent = getRankedWord(rankingpos[0])
    tNegCurrent = getRankedWord(rankingneg[0])

    rankedfeatures = []
    curpos = 0
    curneg = 0
    while tPosCurrent.rank > 0 or tNegCurrent.rank > 0:
        if tPosCurrent.rank >= tNegCurrent.rank:
            rankedfeatures.append(tPosCurrent.word)
            curpos += 1
            tPosCurrent = getRankedWord(rankingpos[curpos])
        else:
            rankedfeatures.append(tNegCurrent.word)
            curneg += 1
            tNegCurrent = getRankedWord(rankingneg[curneg])
    return rankedfeatures

def getSentenceContain(word, listSentences):
    sentences = []
    for sentence in listSentences:
        if word in sentence:
            sentences.append(sentence)
            listSentences.remove(sentence)
    return sentences

class ClauseSplitter():

    @staticmethod
    def split_sentence(sentence, set_clause_indicator):
        sentences = ClauseSplitter.split_sentence_by_clause_indicator(sentence, set_clause_indicator)
        return sentences

    @staticmethod
    def split_sentence_by_clause_indicator(sentence, set_clause_indicator):
        set_words = set(word_tokenize(sentence))
        clause_indicator = ClauseSplitter.find_intersection_word(set_words, set_clause_indicator)

        if not clause_indicator:
            return [sentence]

        if ClauseSplitter.is_begining_of_sentence(clause_indicator, sentence):
            return [sentence]

        indicator_index = sentence.index(clause_indicator)
        clauses = [sentence[:indicator_index], sentence[indicator_index:]]

        return clauses

    @staticmethod
    def is_begining_of_sentence(word, sentence):
        words = word_tokenize(sentence)
        if words.index(word) in [0,1,2]:
            return True
        return False

    @staticmethod
    def find_intersection_word(set1, set2):
        words = list(set1 & set2)
        return words[0] if words else ''


def init_data(folder_path):
    data = []
    file_contents = load_files_in_directory(folder_path)
    for content in file_contents:
        sentences = sent_tokenize(content)
        for sent in sentences:
            data.extend(ClauseSplitter().split_sentence(sent,clause_indicator))
    return data

def main(domains):
    for domain in domains:
        print domain
        print 'Generating polarity shifting sentences for {}'.format(domain)
        # Create sentence array in positive training data
        sentencepos = init_data(DATA_TRAIN_FOLDER(domain, 'positive'))
        sentenceneg = init_data(DATA_TRAIN_FOLDER(domain, 'negative'))

        rankedfeatures = getRankedFeature(domain)
        sShiftFromPos = []
        sShiftFromNeg = []
        sUnshiftFromPos = []
        sUnshiftFromNeg = []

        for t in rankedfeatures:
            posSentenceContainT = getSentenceContain(t, sentencepos)
            negSentenceContainT = getSentenceContain(t, sentenceneg)

            if len(posSentenceContainT) > len(negSentenceContainT):
                sShiftFromNeg.extend(negSentenceContainT)
                sUnshiftFromPos.extend(posSentenceContainT)
            else:
                sShiftFromPos.extend(posSentenceContainT)
                sUnshiftFromNeg.extend(negSentenceContainT)

        sUnshiftFromPos.extend(sentencepos)
        sUnshiftFromNeg.extend(sentenceneg)

        print len(sShiftFromPos)
        print len(sShiftFromNeg)
        print len(sUnshiftFromPos)
        print len(sUnshiftFromNeg)

        random.shuffle(sShiftFromPos)
        random.shuffle(sShiftFromNeg)
        random.shuffle(sUnshiftFromPos)
        random.shuffle(sUnshiftFromNeg)

        sShiftFromPos = sShiftFromPos[:N_MAX]
        sShiftFromNeg = sShiftFromNeg[:N_MAX]
        sUnshiftFromPos = sUnshiftFromPos[:N_MAX]
        sUnshiftFromNeg = sUnshiftFromNeg[:N_MAX]

        write_file_by_utf8(SHIFT_FILE(domain, 'positive'), '\n'.join(sShiftFromPos))
        write_file_by_utf8(UNSHIFT_FILE(domain, 'positive'), '\n'.join(sUnshiftFromPos))

        write_file_by_utf8(SHIFT_FILE(domain, 'negative'), '\n'.join(sShiftFromNeg))
        write_file_by_utf8(UNSHIFT_FILE(domain, 'negative'), '\n'.join(sUnshiftFromNeg))

        print

if __name__ == '__main__':
    set_args = set(sys.argv[1:])
    if not set_args:
        display_general_help(file_name='generate_shifting_training_data')
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
