import os
# import torch
# from torch import nn
# from torch.autograd import Variable
# from random import randint
import pickle
# from gensim.models import KeyedVectors, Word2Vec


def dump_documents_and_extractive_sentence_labels_and_summaries(dir_name, dump_file_name):
    '''
    Args:
    something
    '''
    counter = 0
    documents = []
    for root, dirs, files in os.walk(dir_name):
        for fname in files:
            counter += 1
            print(os.path.join(root, fname), counter)
            try:
                with open(os.path.join(root, fname), 'r', encoding='utf-8') as f:
                    text = f.readlines()
            except UnicodeDecodeError:
                print('Error while reading file --------->\t', os.path.join(root, fname))
                continue
            i = 2
            j = 1
            doc = []
            summary_sentences = []
            while i < len(text):
                if text[i] == '\n':
                    j += 1
                    i += 1
                    if j > 2:
                        break
                    continue
                if j == 1:
                    words = text[i].replace('@', '').split()
                    sentence = " ".join(words[:-1])
                    sentence.replace('\t', ';')
                    label = int(words[-1])
                    label = int(label == 1)
                    doc.append((sentence, label))
                elif j == 2:
                    words = text[i].replace('@', '').split()
                    sentence = " ".join(words)
                    sentence.replace('\t', ';')
                    summary_sentences.append(sentence)
                else:
                    break
                i += 1

            summary = ' . '.join(summary_sentences)
            documents.append((doc, summary))
    print(counter)
    with open(dump_file_name, mode='wb') as file:
        pickle.dump(documents, file)


dump_documents_and_extractive_sentence_labels_and_summaries('/media/pranav/Data/Downloads/Datasets/Summarization/neuralsum/train', '../data/training.pkl')
