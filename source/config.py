from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

RESOURCE_DIR = '../dataset/'
ALL_DOMAINS = ['books', 'dvd', 'electronics', 'kitchen']
ADJ_FILE = '../dictionaries/adj.txt'
STOPWORD_FILE = '../dictionaries/stopwords.txt'
CLAUSE_INDICATOR_FILE = '../dictionaries/clause_indicator.txt'

LAMDA = 0
N_MAX = 1000

# N_GRAM = [1,1]	#Unigram
N_GRAM = [1,2]	#Bigram

# MODEL = SVC(random_state=1234, kernel='linear', probability=True)				#SVM MODEL
MODEL = LogisticRegression(random_state=1234) 		#LOGISTIC REGRESSION MODEL

def UNPROCESSED_FILE(domain,polarity):
	return '../dataset/unprocessed/{}/{}.review'.format(domain, polarity)

def DATA_TRAIN_FOLDER(domain,polarity):
	return '../dataset/data_train/{}/{}/'.format(domain, polarity)

def DATA_TRAIN_FILE(domain,polarity,i):
	return '../dataset/data_train/{}/{}/{:03d}.txt'.format(domain, polarity,i)

def DATA_TEST_FOLDER(domain,polarity):
	return '../dataset/data_test/{}/{}/'.format(domain, polarity)

def DATA_TEST_FILE(domain,polarity,i):
	return '../dataset/data_test/{}/{}/{:03d}.txt'.format(domain, polarity,i)

def RANKING_FILE(domain,polarity):
	return '../result/ranking/{}/{}/ranking.txt'.format(domain, polarity)

def SHIFT_FILE(domain,polarity):
	return '../result/data_train_shifting/{}/{}/shift.txt'.format(domain, polarity)

def UNSHIFT_FILE(domain,polarity):
	return '../result/data_train_shifting/{}/{}/unshift.txt'.format(domain, polarity)

def TFIDF_DETECT_PKL_FILE(domain):
	return '../result/pickle/{}/detect_tfidf.pkl'.format(domain)

def TFIDF_BASE_PKL_FILE(domain):
	return '../result/pickle/{}/base_tfidf.pkl'.format(domain)

def TFIDF_SHIFT_PKL_FILE(domain):
	return '../result/pickle/{}/shift_tfidf.pkl'.format(domain)

def TFIDF_UNSHIFT_PKL_FILE(domain):
	return '../result/pickle/{}/unshift_tfidf.pkl'.format(domain)

def CLASSIFIER_DETECT_PKL_FILE(domain):
	return '../result/pickle/{}/detect_classifier.pkl'.format(domain)

def CLASSIFIER_BASE_PKL_FILE(domain):
	return '../result/pickle/{}/base_classifier.pkl'.format(domain)

def CLASSIFIER_SHIFT_PKL_FILE(domain):
	return '../result/pickle/{}/shift_classifier.pkl'.format(domain)

def CLASSIFIER_UNSHIFT_PKL_FILE(domain):
	return '../result/pickle/{}/unshift_classifier.pkl'.format(domain)

def CLASSIFIER_ENSEMBLE_PKL_FILE(domain):
	return '../result/pickle/{}/ensemble_classifier.pkl'.format(domain)

def SHIFT_PART_TRAIN_FILE(domain, polarity, i):
	return '../result/data_train_shifting/{}/{}/shift/{:03d}.txt'.format(domain, polarity, i)

def SHIFT_PART_TRAIN_FOLDER(domain, polarity):
	return '../result/data_train_shifting/{}/{}/shift/'.format(domain, polarity)

def UNSHIFT_PART_TRAIN_FILE(domain, polarity, i):
	return '../result/data_train_shifting/{}/{}/unshift/{:03d}.txt'.format(domain, polarity, i)

def UNSHIFT_PART_TRAIN_FOLDER(domain, polarity):
	return '../result/data_train_shifting/{}/{}/unshift/'.format(domain, polarity)

def SHIFT_PART_TEST_FILE(domain, polarity, i):
	return '../result/data_test_shifting/{}/{}/shift/{:03d}.txt'.format(domain, polarity, i)

def SHIFT_PART_TEST_FOLDER(domain, polarity):
	return '../result/data_test_shifting/{}/{}/shift/'.format(domain, polarity)

def UNSHIFT_PART_TEST_FILE(domain, polarity, i):
	return '../result/data_test_shifting/{}/{}/unshift/{:03d}.txt'.format(domain, polarity, i)

def UNSHIFT_PART_TEST_FOLDER(domain, polarity):
	return '../result/data_test_shifting/{}/{}/unshift/'.format(domain, polarity)
