from file_io import read_file
from config import ADJ_FILE, STOPWORD_FILE, CLAUSE_INDICATOR_FILE

def init_set_in_file(file_path):
    words = read_file(file_path).split('\n')
    return set([word.replace('\r', '') for word in words if word])

set_adjectives = init_set_in_file(ADJ_FILE)
stopwords = init_set_in_file(STOPWORD_FILE)
clause_indicator = init_set_in_file(CLAUSE_INDICATOR_FILE)
