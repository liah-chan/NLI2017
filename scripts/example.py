import gensim
from gensim import corpora
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.logentropy_model import LogEntropyModel
import numpy as np
import pickle
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, cross_val_score
from scipy.sparse import csr_matrix, csc_matrix, hstack, coo_matrix
import scipy
from gensim.matutils import Sparse2Corpus,Scipy2Corpus,corpus2csc
from sklearn import svm
import sys
import operator
from operator import itemgetter
from itertools import izip

L1_LABEL_SET = ['ARA','GER','FRE','HIN','ITA','JPN','KOR','SPA','TEL','TUR','CHI']
#L1_LABEL_SET = ['ARA','DEU','FRA','HIN','ITA','JPN','KOR','SPA','TEL','TUR','ZHO']

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

def get_line_as_str(word_file = None, lemma_file = None):
	"""
	read lines in file and return as generator
	"""
	if word_file is not None and lemma_file is not None:
		with open(word_file, 'r') as wordf, open(lemma_file,'r') as lemmaf:
			for word, lemma in izip(wordf, lemmaf):
				word = word.split('\t')[0].strip()
				lemma = lemma.split('\t')[0].strip()
				yield str(word+' '+lemma)
	else:
		filename = word_file if word_file is not None else lemma_file
		with open(filename,'r') as f:
			for line in f:
				essay = line.split('\t')[0]
				yield str(essay)

def get_char_ngram(file_pattern, ngram_range):
	"""
	given the input file and ngram range n,
	extract the ngram up to size n.
	
	Param
	------
	file_pattern: str, file pattern to determine the train, dev and test file
	ngram_range: int, character n-gram size

	Return
	------
	train_X_stacked: sparse matrix, representing train set
	train_y: 1d array, indicators of labels in train set
	dev_X_stacked: sparse matrix, representing dev set
	dev_y: 1d array, indicators of labels in dev set
	
	"""
	
	train_file = '../data/train_data.word.'+file_pattern+'txt'
	dev_file = '../data/dev_data.word.'+file_pattern+'txt'
	
	train_X_all = []
	vocab_all = []
	#train_file
	for i in range(1,ngram_range+1):
		char_vectorizer = CountVectorizer(decode_error= 'ignore',
									ngram_range=(i,i),
									lowercase=True,
									analyzer = 'char_wb')
		X = char_vectorizer.fit_transform(get_line_as_str(word_file=train_file))
		sorted_vocab = sorted(char_vectorizer.vocabulary_.items(), key=operator.itemgetter(1))
		sorted_keys =  list(sorted_vocab[i][0] for i in range(len(sorted_vocab)))
		vocab_all.append(sorted_keys)

		s = X.sum(axis = 1)
		frequency_X = coo_matrix(np.nan_to_num(X/s))
		train_X_all.append(frequency_X)
	train_X_stacked = hstack(train_X_all)
	train_y = get_y(train_file)
	
	#dev_file:
	dev_X_all = []
	for i in range(1,len(vocab_all)+1):
		char_vectorizer = CountVectorizer(decode_error= 'ignore',
									ngram_range=(i,i),
									lowercase=True,
									analyzer = 'char_wb',
									vocabulary = vocab_all[i-1])
		X = char_vectorizer.fit_transform(get_line_as_str(dev_file))
		
		s = X.sum(axis = 1)
		frequency_X = coo_matrix(np.nan_to_num(X/s))
		dev_X_all.append(frequency_X)
	dev_X_stacked = hstack(dev_X_all)
	dev_y = get_y(dev_file)
	return train_X_stacked,train_y,dev_X_stacked,dev_y
	
def get_base_feature(file_pattern = '', ngram_range = (1,3), min_df = 2,feature_weight = 'logent',
	analyzer = 'word', target = ['lemma', 'word'],	token_pattern = r'\b\w+\b',
	train_size = 11000, dev_size = 1100, test_size = 1100):
	"""
	getting word/lemma n-grams as base feature
	
	Param
	------
	file_pattern: str, file pattern to determine the train, dev and test file
	ngram_range: int, character n-gram size
	min_df: int, minimum document frequency
	feature_weight: {'logent', 'tfidf', 'binary'}, weighting scheme
	analyzer: str, no need to change it here
	target: list, base feature type	
	token_pattern: str, regex representing allowed pattern
	train_size: int
	dev_size: int
	test_size: int
	
	Return
	------
	train_X_sparse: sparse matrix, representing train set
	train_y: 1d array, indicators of labels in train set
	dev_X_sparse: sparse matrix, representing dev set
	dev_y: 1d array, indicators of labels in dev set
	
	"""
	
	train_word_file = '../data/train_data.word.'+file_pattern+'txt'
	dev_word_file = '../data/dev_data.word.'+file_pattern+'txt'
	train_lemma_file = '../data/train_data.lemma.'+file_pattern+'txt'	
	dev_lemma_file = '../data/dev_data.lemma.'+file_pattern+'txt'
	
	if 'word' in target:
		vec1 = CountVectorizer(decode_error= 'ignore',
										ngram_range=ngram_range,
										lowercase=True,
										analyzer = analyzer,
										token_pattern= token_pattern,
										min_df=min_df)

		vec1.fit_transform(get_line_as_str(word_file = train_word_file))
		vocab_word = vec1.vocabulary_.keys()
	else:
		vocab_word = []

	if 'lemma' in target:
		vec2 = CountVectorizer(decode_error= 'ignore',
										ngram_range=ngram_range,
										lowercase=True,
										analyzer = analyzer,
										token_pattern= token_pattern,
										min_df=min_df)

		vec2.fit_transform(get_line_as_str(lemma_file = train_lemma_file))
		vocab_lemma = vec2.vocabulary_.keys()
	else:
		vocab_lemma = []
	vocab_all = list(set(vocab_word + vocab_lemma))

	vec_all = CountVectorizer(decode_error= 'ignore',
									ngram_range=ngram_range,
									lowercase=True,
									analyzer = 'word',
									token_pattern= token_pattern,
									vocabulary=vocab_all)
									
	infiles_train = {'word_file': train_word_file, 'lemma_file': train_lemma_file}
	infiles_dev = {'word_file': dev_word_file, 'lemma_file': dev_lemma_file}
	
	train_X_sparse, train_y = transform_to_sparse(infiles = infiles_train,
							N = train_size, feature_size = len(vocab_all),
							feature_weight = feature_weight,
							vectorizer = vec_all)
	dev_X_sparse, dev_y = transform_to_sparse(infiles = infiles_dev, 
							N = dev_size, feature_size = len(vocab_all),
							feature_weight = feature_weight,
							vectorizer = vec_all)
	# test_X_sparse, test_y = transform_to_sparse(inflie = test_file_w, 
	# 						N = 1100, feature_size = len(vocab_all),
	# 						vectorizer = word_vectorizer)

	print(train_X_sparse.shape)
	print(dev_X_sparse.shape)
	# print(test_X_sparse.shape)
	# print(train_X_sparse.tocsr()[0])
	return train_X_sparse, train_y,dev_X_sparse, dev_y#,test_X_sparse, test_y

def transform_to_sparse(infiles, N, feature_size,vectorizer = None, 
	feature_weight = 'logent'):
	"""
	Param
	------
	infiles: dict, in the form {word_file: "path", lemma_file: "path"}
	N: the number of instances in the file
	feature_size: int
	vectorizer: sklearn vectorizer
	feature_weight: {'logent', 'tfidf', 'binary'}, weighting scheme
	
	Return
	------
	X: sparse matrix, feature representation of infiles with specific weighting scheme
	y: 1d array,  indicators of labels in infiles
	
	"""
	infile = infiles['word_file'] if infiles['word_file'] is not None else infiles['lemma_file']
	if vectorizer is not None:
		if feature_weight == 'binary':
			vectorizer = CountVectorizer(decode_error= 'ignore',
										ngram_range=(1,3),
										lowercase=True,
										analyzer = 'word',
										binary = True,
										token_pattern=r'\b\w+\b')
										token_pattern=r'\b[[\w]+|\,|\;|\"|]')
			X = vectorizer.fit_transform(get_line_as_str(**infiles))
			y = get_y(infile)
		else:
			X = Scipy2Corpus(vectorizer.fit_transform(get_line_as_str(**infiles)))
			if feature_weight == 'tfidf':
				weighting_scheme = TfidfModel(X)
			elif feature_weight == 'logent':			
				weighting_scheme = LogEntropyModel(X)
			x = weighting_scheme[X]
			y = get_y(infile)
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
			print(len(data))
			print(len(rows))
			print(len(cols))
			print(N)
			print(feature_size)
			X = csr_matrix((data,(rows,cols)),shape=(N, feature_size))
	return X, y	

def get_other_feature():
	"""
	function to extract other features
	
	Param
	------
	

	Return
	------
	in a similar format as other functions for feature extraction
	
	"""

	pass
	
def main():
	file_pattern = ''
	token_pattern = r'[\w!#$%&()*+\-./:<=>?@[\]^_`{|}~]+'
	
	print('extracting base features...')
	train_X_base, train_y,dev_X_base, dev_y = get_base_feature(file_pattern = file_pattern,
					 feature_weight = 'binary',
					 # token_pattern = token_pattern,	#uncomment this one to use the pattern including punctuations				  
					 train_size = 11000, 
					 dev_size = 1100,
					 target = ['word','lemma'])
	
	print('extracting character n-grams...')
	train_X_char, train_y, dev_X_char, dev_y = get_char_ngram(file_pattern = file_pattern,ngram_range = 3)
	
	#you can add other features in the sequece to pass to hstact() function
	
	train_X_matrix = hstack([train_X_base, train_X_char], format='csr')
	dev_X_matrix = hstack([dev_X_base,dev_X_char], format='csr')
#	test_X_matrix = hstack([test_X_error,test_X_sparse], format='csr')
	
	print(train_X_matrix.shape)
	print(dev_X_matrix.shape)
	# print(test_X_sparse_matrix.shape)

	clf_svc = svm.LinearSVC(C=100, multi_class ='ovr')
	clf_svc.fit(train_X_matrix,train_y)

	pred_val = clf_svc.predict(dev_X_matrix)
	acc_val = accuracy_score(dev_y,pred_val)
	print("val acc: {:f}".format(acc_val)) #will be 1.0

	# pred_test = clf_svc.predict(test_X_sparse_matrix)
	# proba_test = clf_svc.decision_function(test_X_sparse_matrix)
	# acc_test = accuracy_score(test_y,pred_test)
	# print("test acc: {:f}".format(acc_test))

if __name__ == "__main__":
	main()
