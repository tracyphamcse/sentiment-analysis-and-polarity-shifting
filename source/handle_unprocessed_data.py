import sys
from bs4 import BeautifulSoup as Soup

from config import ALL_DOMAINS, DATA_TRAIN_FILE, DATA_TEST_FILE, UNPROCESSED_FILE
from utils import display_general_help, display_general_domain_not_supported
from file_io import read_file_by_utf8, write_file_by_utf8



def parse_to_review_contents(review_data):
    soup = Soup(review_data, 'lxml')
    reviews = soup.find_all('review_text')
    review_contents = [review.contents[0].strip() for review in reviews]
    return review_contents

def handle_unprocessed_data(domain, polarity):
    review_data = read_file_by_utf8(UNPROCESSED_FILE(domain,polarity))
    review_contents = parse_to_review_contents(review_data)

    data_for_train = [review_content for i, review_content in enumerate(review_contents) if i % 2 is not 0]
    data_for_test = [review_content for i, review_content in enumerate(review_contents) if i % 2 is 0]

    for i, data in enumerate(data_for_train):
        write_file_by_utf8(DATA_TRAIN_FILE(domain, polarity, i), data)
    for i, data in enumerate(data_for_test):
        write_file_by_utf8(DATA_TEST_FILE(domain, polarity, i), data)

    return len(data_for_train) + len(data_for_test)

def main(domains):
    for domain in domains:
        print domain
        print 'Handling domain {}, polarity {}...'.format(domain, 'positive')
        number_of_handled_files = handle_unprocessed_data(domain, 'positive')
        print 'Handled {} file(s)'.format(number_of_handled_files)

        print 'Handling domain {}, polarity {}...'.format(domain, 'negative')
        number_of_handled_files = handle_unprocessed_data(domain, 'negative')
        print 'Handled {} file(s)'.format(number_of_handled_files)

        print

if __name__ == '__main__':
    set_args = set(sys.argv[1:])
    if not set_args:
        display_general_help(file_name='handle_unprocessed_data')
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
