import gensim
from gensim import corpora
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.logentropy_model import LogEntropyModel
import numpy as np
import pickle
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold, cross_val_score
from scipy.sparse import csr_matrix, csc_matrix, hstack, coo_matrix
import scipy
from gensim.matutils import Sparse2Corpus,Scipy2Corpus,corpus2csc
from sklearn import svm
import sys
import operator
from operator import itemgetter

L1_LABEL_SET = ['ARA','GER','FRE','HIN','ITA','JPN','KOR','SPA','TEL','TUR','CHI']

def encode_label(label):
	"""
	encode label str as int
	"""
	return L1_LABEL_SET.index(label)

def get_y(filename):
	"""
	given data file (containing all examples in train, dev or test set), 
	return 1d array containing label indicator 
	"""
	idxs = []
	with open(filename,'r') as f:
		for line in f:
			label = (line.split('\t')[1]).strip()
			idx = encode_label(label)
			idxs.append(idx)
	return np.array(idxs,dtype=np.int)

def get_line_as_str(filename):
	"""
	read lines in file and return as generator
	"""
	with open(filename,'r') as f:
		for line in f:
			essay = line.split('\t')[0]
			yield str(essay)

def get_separate_vocab(target = ['word', 'lemma', 'error'],ngram_range = (1,3),
			analyzer = 'word', pattern = r'\b\w+\b',
			word_file = None, lemma_file = None):
	if 'word' in target and word_file is not None:
		vec1 = CountVectorizer(decode_error= 'ignore',
										ngram_range=ngram_range,
										lowercase=True,
										analyzer = analyzer,
										token_pattern= pattern,
										min_df=2)

		vec1.fit_transform(get_line_as_str(word_file))
		vocab_word = vec1.vocabulary_.keys()
	else:
		vocab_word = []

	if 'lemma' in target and lemma_file is not None:
		vec2 = CountVectorizer(decode_error= 'ignore',
										ngram_range=ngram_range,
										lowercase=True,
										analyzer = analyzer,
										token_pattern= pattern,
										min_df=2)

		vec2.fit_transform(get_line_as_str(lemma_file))
		vocab_lemma = vec2.vocabulary_.keys()
	else:
		vocab_lemma = []

	if 'error_icle' in target:
		with open('../data/typo_dict_icle.pickle', 'rb') as handle:
			typo_dict = pickle.load(handle)
		vocab_typo = typo_dict.keys()
	elif 'error_ets' in target:
		with open('../data/typo_dict_ets.pickle', 'rb') as handle:
			typo_dict = pickle.load(handle)
		vocab_typo = typo_dict.keys()
	elif 'errors' in target:
		with open('../data/typo_dict_all.pickle', 'rb') as handle:
			typo_dict = pickle.load(handle)
		vocab_typo = typo_dict.keys()
	else:
		vocab_typo = []

	return vocab_typo,vocab_word,vocab_lemma


def get_full_vocab(word_file = None, lemma_file = None, ngram_range = (1,3), 
	analyzer = 'word', pattern = r'\b\w+\b',
	target = ['lemma', 'word']):
	vocab_typo,vocab_word,vocab_lemma = get_separate_vocab(target = target, 
											ngram_range = ngram_range,
											analyzer = analyzer,
											pattern = pattern,
											word_file = word_file,
											lemma_file = lemma_file)
	return list(set(vocab_typo + vocab_word + vocab_lemma))

def get_char_vocab(infile, ngram_range):
	char_vectorizer = CountVectorizer(decode_error= 'ignore',
									ngram_range=ngram_range,
									lowercase=True,
									analyzer = 'char_wb')
	X = char_vectorizer.fit_transform(get_line_as_str(infile))
	return char_vectorizer.vocabulary_.keys()

def get_char_ngram(inflie, n):
	"""
	given the input file and ngram range n,
	extract the ngram up to size n

	Return:
	vocab_all : a list of n lists, each one represents
				the vocabulary related to that ngram size
	"""
	X_all = []
	vocab_all = []
	for i in range(1,n+1):
		char_vectorizer = CountVectorizer(decode_error= 'ignore',
									ngram_range=(i,i),
									lowercase=True,
									analyzer = 'char_wb')
		X = char_vectorizer.fit_transform(get_line_as_str(inflie))
		sorted_vocab = sorted(char_vectorizer.vocabulary_.items(), key=operator.itemgetter(1))
		sorted_keys =  list(sorted_vocab[i][0] for i in range(len(sorted_vocab)))
		vocab_all.append(sorted_keys)

		s = X.sum(axis = 1)
		frequency_X = coo_matrix(np.nan_to_num(X/s))
		X_all.append(frequency_X)
	X_stacked = hstack(X_all)
	y = get_y(inflie)

	return X_stacked,y,vocab_all

def get_char_ngram_with_vocab(inflie, n, vocab):
	"""
	given a file, ngram range n and vocabulary
	extract the ngram in file up to size n according to vocab
	"""
	X_all = []
#	vocab_all = []
#	for item in vocab:
#		vocab_all += item
	for i in range(1,len(vocab)+1):
		char_vectorizer = CountVectorizer(decode_error= 'ignore',
									ngram_range=(i,i),
									lowercase=True,
									analyzer = 'char_wb',
#									vocabulary = vocab_all)
									vocabulary = vocab[i-1])
		X = char_vectorizer.fit_transform(get_line_as_str(inflie))
		
		s = X.sum(axis = 1)
		frequency_X = coo_matrix(np.nan_to_num(X/s))
		X_all.append(frequency_X)
	X_stacked = hstack(X_all)
	y = get_y(inflie)

	return X_stacked,y

def transform_to_sparse(inflie, N, feature_size,vectorizer = None):
	"""
	N: the number of instances in the file 
	"""
	if vectorizer is not None:
		X = Scipy2Corpus(vectorizer.fit_transform(get_line_as_str(inflie)))
		# tfidf = TfidfModel(X)
		# train_X = tfidf[X]
		logen = LogEntropyModel(X)
		x = logen[X]
		y = get_y(inflie)
		data = []
		rows = []
		cols = []
		line_count = 0
		for line in x:
			for elem in line:
				rows.append(line_count)
				cols.append(elem[0])
				data.append(elem[1])
			line_count += 1
	# return csr_matrix((data,(rows,cols)),shape=mat_shape), y
	return csr_matrix((data,(rows,cols)),shape=(N, feature_size)), y								

# def get_error_ngram(char_n =3, file_pattern = 'data'):
# 	#file_pattern = 'data'
# 	#file_pattern = 'data_without_typo_icle7'
# 	train_file = '../data/train+dev_'+file_pattern+'.txt'
# 	dev_file = '../data/dev_'+file_pattern+'.txt'
# 	test_file = '../data/test_'+file_pattern+'.txt'
	
# 	vocab_file = '../data/train+dev_typo_icle7.txt'

# 	#use char (1-3) grams of errors 
# 	__, _, vocab_char = get_char_ngram(vocab_file,char_n)	
# #	for i in range(len(vocab_char)):
# #	 	print(len(vocab_char[i]))
	
# 	train_X_error, train_y = get_char_ngram_with_vocab(train_file,char_n,vocab_char)
# 	dev_X_error, dev_y = get_char_ngram_with_vocab(dev_file,char_n, vocab_char)
# 	test_X_error, test_y = get_char_ngram_with_vocab(test_file,char_n,vocab_char)
# 	print(train_X_error.shape)
# 	print(dev_X_error.shape)
# 	print(test_X_error.shape)
# 	# print(train_X_error.tocsr()[0])
# 	return train_X_error, train_y,dev_X_error, dev_y,test_X_error, test_y, vocab_char
	
def get_word_ngram(ngram_range = (1,3), file_pattern = 'data'):

	train_file_w = '../data/train_'+file_pattern+'.txt'
	dev_file_w = '../data/dev_'+file_pattern +'.txt'
	# test_file_w = '../data/test_'+file_pattern +'.txt'

	word_file = '../data/train_data.txt'
	# lemma_file = '../data/train_data.txt'

	vocab_word = get_full_vocab(word_file = word_file,
			analyzer = 'word',
			ngram_range = ngram_range, target = ['word'])

	word_vectorizer = CountVectorizer(decode_error= 'ignore',
									ngram_range=ngram_range,
									lowercase=True,
									analyzer = 'word',
									token_pattern=r'\b\w+\b',
									vocabulary=vocab_word)

	#use word ngram (with lemma or error)
	train_X_sparse, train_y = transform_to_sparse(inflie = train_file_w,
							N = 11000, feature_size = len(vocab_word),
							vectorizer = word_vectorizer)
	dev_X_sparse, dev_y = transform_to_sparse(inflie = dev_file_w, 
							N = 1100, feature_size = len(vocab_word),
							vectorizer = word_vectorizer)
	# test_X_sparse, test_y = transform_to_sparse(inflie = test_file_w, 
	# 						N = 1100, feature_size = len(vocab_word),
	# 						vectorizer = word_vectorizer)

	print(train_X_sparse.shape)
	print(dev_X_sparse.shape)
	# print(test_X_sparse.shape)
	# print(train_X_sparse.tocsr()[0])
	return train_X_sparse, train_y,dev_X_sparse, dev_y#,test_X_sparse, test_y

def main():
	train_X_sparse_matrix, train_y,dev_X_sparse_matrix, dev_y =get_word_ngram()

	# train_X_error, train_y,dev_X_error, dev_y,test_X_error, test_y, vocab_char= get_error_ngram()
	# train_X_sparse_matrix = hstack([train_X_error, train_X_sparse])
	# dev_X_sparse_matrix = hstack([dev_X_error,dev_X_sparse])
	# test_X_sparse_matrix = hstack([test_X_error,test_X_sparse])

	print(train_X_sparse_matrix.shape)
	print(dev_X_sparse_matrix.shape)
	# print(test_X_sparse_matrix.shape)

	clf_svc = svm.LinearSVC(C=100, multi_class ='ovr')
	clf_svc.fit(train_X_sparse_matrix,train_y)

	pred_val = clf_svc.predict(dev_X_sparse_matrix)
	acc_val = accuracy_score(dev_y,pred_val)
	print("val acc: {:f}".format(acc_val)) #will be 1.0

	# pred_test = clf_svc.predict(test_X_sparse_matrix)
	# proba_test = clf_svc.decision_function(test_X_sparse_matrix)
	# acc_test = accuracy_score(test_y,pred_test)
	# print("test acc: {:f}".format(acc_test))

if __name__ == "__main__":
	main()
