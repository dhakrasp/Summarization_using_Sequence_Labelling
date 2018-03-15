import pickle
import sys
import numpy as np
import json
from nltk import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint


def get_vector(word, vocab, glove_array):
    idx = vocab['<unk>']
    if word in vocab:
        idx = vocab[word]
    elif word.lower() in vocab:
        idx = vocab[word.lower()]
    return glove_array[idx]


def make_write_safe(text):
    text = text.replace('\u2013', '-')
    text = text.replace('\u2014', '')
    text = text.replace('\u22f1', '')
    # return text
    return text.encode('utf-8', 'ignore').decode('latin-1', 'ignore')


filename = sys.argv[1]
glove_file = sys.argv[2]
with open(glove_file.replace('.npy', '.json'), mode='r') as file:
    input_word2idx = json.load(file)

with open(filename, mode='rb') as file:
    data = pickle.load(file)

array = np.lib.format.open_memmap(glove_file, dtype='float32')

new_data = []
for idx, doc in enumerate(data, 1):
    new_doc = []
    doc_sent_vecs = []
    summ_sent_vecs = []
    sent_labels = []
    for sent, label in doc[0]:
        sent_labels.append(0)
        word_vecs = np.array([get_vector(word, input_word2idx, array)
                              for word in word_tokenize(sent)])
        sent_vec = np.mean(word_vecs, axis=0)
        doc_sent_vecs.append(sent_vec)
    for sent in doc[1]:
        word_vecs = np.array([get_vector(word.replace('*', ''), input_word2idx, array)
                              for word in word_tokenize(sent)])
        sent_vec = np.mean(word_vecs, axis=0)
        summ_sent_vecs.append(sent_vec)
    doc_matrix = np.array(doc_sent_vecs)
    summ_matrix = np.array(summ_sent_vecs)
    corr_matrix = cosine_similarity(summ_matrix, doc_matrix)
    for i in range(corr_matrix.shape[0]):
        selected = np.argsort(corr_matrix[i])[-2:]
        # print(selected, corr_matrix[i][selected])
        for j in selected:
            sent_labels[j] = 1
    # print(make_write_safe('\n'.join(doc[1]).replace('*', '')))
    # print('\n')
    for i in range(len(sent_labels)):
        new_sent = list(doc[0][i])
        new_sent[1] = sent_labels[i]
        doc[0][i] = new_sent
    # pprint(doc)
    # print('------------------------')
    # print('\n\n')
    # break
    if idx % 1000 == 0:
        print(idx)
with open(filename.replace('.pkl', '_new.pkl'), mode='wb') as file:
    pickle.dump(data, file)
    
