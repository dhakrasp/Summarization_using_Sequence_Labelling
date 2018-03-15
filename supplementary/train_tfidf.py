# @Author: Pranav Dhakras
# @Date: 2018-02-24 23:38:57
# @Last Modified by: Pranav Dhakras
# @Last Modified time: 2018-02-24 23:38:57

import os
import sys
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
import nltk


def make_write_safe(text):
    return text.encode('latin-1', 'ignore').decode('utf-8', 'ignore')


def scale_value(val, old_min, old_max, new_min, new_max):
    old_range = old_max - old_min
    new_range = new_max - new_min
    return new_min + new_range * (val - old_min) / old_range


data_file = sys.argv[1]
dump_file = sys.argv[2]

with open(data_file, mode='rb') as file:
    docs = pickle.load(file)

sents = []
for doc in docs:
    # Pick abstractive sentences
    for sent in doc[1]:
        sent = sent.replace('*', '')
        sents.append(make_write_safe(sent + '\n'))

stopwords = nltk.corpus.stopwords.words('english') + list(punctuation)
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
tfidf_sparse_matrix = tfidf_vectorizer.fit_transform(sents)
idx2word = {i: w for i, w in enumerate(tfidf_vectorizer.get_feature_names())}
word2idx = {w: i for w, i in idx2word.items()}

mean_scores = np.asarray(tfidf_sparse_matrix.mean(axis=0))
mean_scores_dict = {idx2word[i]: value for i, value in enumerate(mean_scores[0])}

vals = list(mean_scores_dict.values())
min_val = min(vals)
max_val = max(vals)
for w, v in mean_scores_dict.items():
    new_value = scale_value(v, min_val, max_val, 0.01, 1)
    mean_scores_dict[w] = new_value
print(len(mean_scores_dict))
topk = sorted(mean_scores_dict.items(), key=lambda x: x[1], reverse=True)[:1000]
# for w, v in topk:
#     print(w, '\t', v)
with open(dump_file, mode='w') as file:
    json.dump(mean_scores_dict, file, indent=4)
