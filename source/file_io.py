import codecs
import glob
import os
from config import RESOURCE_DIR


def read_file(file_path):
    input = open(file_path, 'r')
    file_content = input.read()
    input.close()
    return file_content

def read_file_by_utf8(file_path):
    input = codecs.open(file_path, 'r', encoding='utf-8', errors='ignore')
    file_content = input.read()
    input.close()
    return file_content

def load_files_in_directory(directory):
    file_paths = glob.glob(directory + '*.txt')
    return [read_file_by_utf8(file_path) for file_path in file_paths]

def write_file(file_path, content):
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    output = open(file_path, 'w')
    output.write(content)
    output.close()

def write_file_by_utf8(file_path, content):
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    output = codecs.open(file_path, 'w', encoding='utf-8')
    output.write(content)
    output.close()

def read_shifting_data(domain):
    shift_pos = read_file_by_utf8(SHIFT_FILE(domain,'positive')).split('\n')
    shift_neg = read_file_by_utf8(SHIFT_FILE(domain,'negative')).split('\n')
    unshift_pos = read_file_by_utf8(UNSHIFT_FILE(domain,'positive')).split('\n')
    unshift_neg = read_file_by_utf8(UNSHIFT_FILE(domain,'negative')).split('\n')
    return shift_pos + shift_neg + unshift_pos + unshift_neg

def read_sentiment_data(domain):
    pos = load_files_in_directory(DATA_TRAIN_FOLDER(domain, 'positive'))
    neg = load_files_in_directory(DATA_TRAIN_FOLDER(domain, 'negative'))
    return pos + neg
