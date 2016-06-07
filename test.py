#!/usr/bin/python

########################################################
__author__ = "Surajit Dasgupta"
__copyright__ = "Copyright (c) 2016, JU NLP MWE Project"
__credits__ = ["Surajit Dasgupta", "Chandan Prakash"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Surajit Dasgupta"
__email__ = "surajit.techie@gmail.com"
__status__ = "Development"
########################################################                                       

from collections import defaultdict
from nltk.corpus import stopwords
import string
import glove
import re

window = 4

text = ''.join(open('paulgraham.txt').readlines())
sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
stop = stopwords.words('english')

wordset = set()
bigrams = list()
wordhash = dict()
cooccur = {}
vector = list()

for i in range(len(sentences)):
	sentences[i] = re.sub('[%s]' % re.escape(string.punctuation), '', sentences[i])
	sentences[i] = sentences[i].lower()
	sentences[i] = sentences[i].replace("\n"," ")
	sentences[i] = [word for word in sentences[i].split() if word not in stop]

for i in range(len(sentences)):
		wordset.update(sentences[i])
		for j in range(len(sentences[i]) - 1):
			bigrams.append(sentences[i][j] + ' ' + sentences[i][j + 1])
		wordset.update(bigrams)

wordset = list(wordset)

for i in range(len(wordset)):
	wordhash[wordset[i]] = i

for i in range(len(wordhash)):
	cooccur[i] = cooccur.get(i, {})

#inv_wordhash = dict((v, k) for k, v in wordhash.iteritems())

for i in range(len(sentences)):					#Unigram cooccurences
	for j in range(len(sentences[i])):
		left = max(j - window, 0)
		right = min(j + 1+window, len(sentences[i]))
		for k in range(left, j):
			center_id = wordhash[sentences[i][j]]
			context_id = wordhash[sentences[i][k]]
			cooccur[center_id][context_id] = cooccur[center_id].get(context_id, 0) + 1

for i in range(len(sentences)):					#Bigram cooccurences
	for j in range(len(sentences[i]) - 1):
		left = max(j - window, 0)
		right = min(j + 1 + window, len(sentences[i]))
		bigram_word = sentences[i][j] + ' ' + sentences[i][j+1]
		for k in range(j + 2, right):
			center_id = wordhash[sentences[i][j]]
			context_id = wordhash[sentences[i][k]]
			cooccur[center_id][context_id] = cooccur[center_id].get(context_id, 0) + 1
				
model = glove.Glove(cooccur, d = 10, alpha = 0.75, x_max = 100.0)

for epoch in range(100):
    err = model.train(workers = 9, batch_size = 10)
    #print("epoch %d, error %.3f" % (epoch, err))
    
vector = model.W

for i in range(len(vector)):
	print vector[i]

