import itertools
import math
import pickle
from collections import namedtuple
from pprint import pprint

import numpy as np
import torch
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from pyrouge import Rouge155
from seq2seq.util.checkpoint import Checkpoint
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from torch import nn
from torch.autograd import Variable

import random
import math
import logging
from time import asctime
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

stop_symbols = set(stopwords.words('english') + list(punctuation))
use_cuda = torch.cuda.is_available()
EncoderConfig = namedtuple('EncoderConfig', ['input_size',
                                             'hidden_size',
                                             'bidirectional',
                                             'batch_first',
                                             'num_layers',
                                             'variable_lengths',
                                             'dropout'])
DecoderConfig = namedtuple('DecoderConfig', ['vocab_size',
                                             'hidden_size',
                                             'max_len',
                                             'sos_id',
                                             'eos_id',
                                             'bidirectional',
                                             'num_layers',
                                             'use_attention',
                                             'dropout_p',
                                             'input_dropout_p'])
LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'


def sort_tensor_by_lengths(unsorted_tensor, lengths):
    """[Sorts tensors in descending order by their lengths]

    [description]

    Arguments:
        unsorted_tensor {[A torch tensor]} -- [description]
        lengths {[list of integers]} -- [description]
    """
    _, indices = torch.sort(torch.LongTensor(lengths), descending=True)
    sorted_tensor = unsorted_tensor.index_select(0, indices)
    if unsorted_tensor.is_cuda:
        sorted_tensor.cuda()
    return sorted_tensor, indices


def unsort_tensor(sorted_tensor, indices):
    _, reverse_indices = torch.sort(indices, descending=False)
    unsorted_tensor = sorted_tensor.index_select(0, reverse_indices)
    if sorted_tensor.is_cuda:
        unsorted_tensor.cuda()
    return unsorted_tensor, reverse_indices


def clip_grad_norm(optimizer, max_grad_norm=5):
    params = itertools.chain.from_iterable(
        [group['params'] for group in optimizer.param_groups])
    torch.nn.utils.clip_grad_norm(params, max_grad_norm)


def to_categorical(x, num_classes, max_len=None):
    assert max(x) < num_classes
    if max_len is not None:
        ret = torch.Tensor(max_len, num_classes).zero_()
    else:
        ret = torch.Tensor(len(x), num_classes).zero_()
    for i, val in enumerate(x):
        ret[i, val] = 1
    return ret


def load_checkpoint(checkpoint_name, expt_dir):
    if checkpoint_name is not None:
        print("loading checkpoint from {}".format(os.path.join(
            expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, checkpoint_name)))
        checkpoint_path = os.path.join(
            expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, checkpoint_name)
    else:
        checkpoint_path = Checkpoint.get_latest_checkpoint(expt_dir)
    return Checkpoint.load(checkpoint_path)


# def get_vector_and_weight(word, vocab, glove_array, tfidf_weights_dict):
#     idx = vocab['<unk>']
#     if word in vocab:
#         idx = vocab[word]
#     elif word.lower() in vocab:
#         idx = vocab[word.lower()]
#     if tfidf_weights_dict and word in tfidf_weights_dict:
#         weight = tfidf_weights_dict[word]
#     else:
#         weight = 0.01
#     return glove_array[idx], weight


# def sents_to_input(sents, vocab, glove_array, max_len, include_length=False, tfidf_weights_dict=None):
#     sent_vecs = []
#     lengths = np.array([len(sent) for sent in sents], dtype='float32')
#     rel_lengths = lengths / np.sum(lengths)
#     for i, sentence in enumerate(sents):
#         weights = []
#         word_vecs = []
#         for word in word_tokenize(sentence):
#             if word in stop_symbols:
#                 continue
#             vec, wt = get_vector_and_weight(
#                 word, vocab, glove_array, tfidf_weights_dict)
#             word_vecs.append(vec * wt)
#             weights.append(wt)
#         sent_vec = np.mean(np.array(word_vecs, dtype='float32'),
#                            axis=0) / float(sum(weights))
#         if include_length:
#             sent_vec = np.concatenate((sent_vec, rel_lengths[i:i + 1]))
#         sent_vecs.append(torch.from_numpy(sent_vec))
#     length = len(sent_vecs)
#     for l in range(max_len - len(sent_vecs)):
#         sent_vecs.append(torch.zeros_like(sent_vecs[0]))
#     out = torch.stack(sent_vecs)
#     return out, length
def get_vector(word, vocab, glove_array):
    idx = vocab['<unk>']
    if word in vocab:
        idx = vocab[word]
    elif word.lower() in vocab:
        idx = vocab[word.lower()]

    return glove_array[idx]


def sents_to_input(sents, vocab, glove_array, max_len, include_length=False, tfidf_weights_dict=None):
    sent_vecs = []
    lengths = np.array([len(sent) for sent in sents], dtype='float32')
    rel_lengths = lengths / np.sum(lengths)
    for i, sentence in enumerate(sents):
        word_vecs = []
        for word in word_tokenize(sentence):
            vec = get_vector(word, vocab, glove_array)
            if word in stop_symbols:
                vec = vec * 0
            word_vecs.append(vec)
        sent_vec = np.mean(np.array(word_vecs, dtype='float32'), axis=0)
        if include_length:
            sent_vec = np.concatenate((sent_vec, rel_lengths[i:i + 1]))
        sent_vecs.append(torch.from_numpy(sent_vec))
    length = len(sent_vecs)
    for l in range(max_len - len(sent_vecs)):
        sent_vecs.append(torch.zeros_like(sent_vecs[0]))
    out = torch.stack(sent_vecs)
    return out, length


def batch_generator(inputs, lengths, outputs, batch_size):
    num_batchs = math.ceil(inputs.size(0) // batch_size)
    for i in range(num_batchs):
        start = i * batch_size
        end = (i + 1) * batch_size
        yield inputs[start:end], lengths[start:end], outputs[start:end]


def create_train_batch_from_docs(docs, input_word2idx, output_word2idx, glove_array, max_len, include_length=False, tfidf_weights_dict=None, sort=True):
    inputs = []
    lengths = []
    targets = []
    for doc in docs:
        sents = []
        sent_labels = ['<sos>']
        for sent, label in doc[0][:max_len]:
            sents.append(sent)
            # Convert to string. Also, convert 2 to '0'
            sent_labels.append('0' if label == 2 else str(label))
        sent_labels.append('<eos>')
        for l in range(max_len - len(sent_labels) + 2):
            sent_labels.append('NA')
        doc_input, doc_length = sents_to_input(
            sents, input_word2idx, glove_array, max_len, include_length, tfidf_weights_dict)
        sent_label_indices = [output_word2idx[label] for label in sent_labels]
        lengths.append(doc_length)
        inputs.append(doc_input)
        targets.append(sent_label_indices)
    inputs = torch.stack(inputs)
    targets = torch.LongTensor(targets)
    if sort:
        inputs, indices = sort_tensor_by_lengths(inputs, lengths)
        targets, _ = sort_tensor_by_lengths(targets, lengths)
        lengths = sorted(lengths, reverse=True)
    if use_cuda:
        inputs = inputs.cuda()
        targets = targets.cuda()
    return Variable(inputs), lengths, Variable(targets), indices


def create_predict_batch_from_docs(docs, input_word2idx, output_word2idx, glove_array, max_len, include_length=False, tfidf_weights_dict=None, sort=True):
    inputs = []
    lengths = []
    targets = []
    docs_to_return = docs
    for doc in docs:
        sents = []
        sent_labels = ['<sos>']
        for sent, label in doc[0][:max_len]:
            sents.append(sent)
            # Convert to string. Also, convert 2 to '1'
            sent_labels.append('1' if label == 2 else str(label))
        sent_labels.append('<eos>')
        for l in range(max_len - len(sent_labels) + 2):
            sent_labels.append('NA')
        doc_input, doc_length = sents_to_input(
            sents, input_word2idx, glove_array, max_len, include_length, tfidf_weights_dict)
        sent_label_indices = [output_word2idx[label] for label in sent_labels]
        lengths.append(doc_length)
        inputs.append(doc_input)
        targets.append(sent_label_indices)
    inputs = torch.stack(inputs)
    targets = torch.LongTensor(targets)
    if sort:
        inputs, indices = sort_tensor_by_lengths(inputs, lengths)
        targets, _ = sort_tensor_by_lengths(targets, lengths)
        lengths = sorted(lengths, reverse=True)
        docs_to_return = [docs_to_return[i] for i in indices]
    if use_cuda:
        inputs = inputs.cuda()
        targets = targets.cuda()
    return Variable(inputs), lengths, Variable(targets), indices, docs_to_return


def data_generator(data_path, input_word2idx, output_word2idx, glove_array, max_len, batch_size=16, include_length=False, tfidf_weights_dict=None, return_docs=False, max_docs=None):
    with open(data_path, mode='rb') as file:
        docs = pickle.load(file)
    if max_docs is not None:
        # docs = random.sample(docs, max_docs)
        docs = docs[:max_docs]
    # else:
    #     random.shuffle(docs)
    num_batchs = math.ceil(len(docs) / batch_size)
    for i in range(num_batchs):
        start = i * batch_size
        end = (i + 1) * batch_size
        if not return_docs:
            yield create_train_batch_from_docs(docs[start:end], input_word2idx,
                                               output_word2idx, glove_array, max_len, include_length, tfidf_weights_dict)
        else:
            yield create_predict_batch_from_docs(docs[start:end], input_word2idx,
                                                 output_word2idx, glove_array, max_len, include_length, tfidf_weights_dict)


def train_on_mini_batch(model, optimizer, loss_func, inputs, lengths, targets):
    model.zero_grad()
    model_outputs, _, ret_dict = model.forward(
        inputs, lengths, targets, teacher_forcing_ratio=0)
    loss = 0
    for idx in range(model_outputs.size()[0]):
        loss += loss_func(model_outputs[idx], targets[idx, 1:])
    loss.backward()
    clip_grad_norm(optimizer, max_grad_norm=5)
    optimizer.step()
    return loss.data[0]


def train_epochs_generator(train_generator_dict, model, optimizer, loss_func, epochs, start_epoch, experiment_dir, print_every=100, save_every=None, valid_generator_dict=None):
    best_epoch_loss = 1e8
    if start_epoch < 0:
        start_epoch = 0
    start_epoch += 1
    for epoch in range(start_epoch, start_epoch + epochs):
        train_generator = data_generator(train_generator_dict['path'],
                                         train_generator_dict['input_word2idx'],
                                         train_generator_dict['output_word2idx'],
                                         train_generator_dict['glove_array'],
                                         train_generator_dict['max_len'],
                                         train_generator_dict['batch_size'],
                                         train_generator_dict['include_length'],
                                         train_generator_dict['tfidf_weights_dict'])
        if valid_generator_dict is not None:
            valid_generator = data_generator(valid_generator_dict['path'],
                                             valid_generator_dict['input_word2idx'],
                                             valid_generator_dict['output_word2idx'],
                                             valid_generator_dict['glove_array'],
                                             valid_generator_dict['max_len'],
                                             valid_generator_dict['batch_size'],
                                             train_generator_dict['include_length'],
                                             train_generator_dict['tfidf_weights_dict'])
        step = 0
        epoch_loss = 0
        model.train()
        for inputs, lengths, targets, indices in train_generator:
            step += 1
            epoch_loss += train_on_mini_batch(model,
                                              optimizer, loss_func, inputs, lengths, targets)
            if step % print_every == 0:
                logging.info('Step : {} Avg Step Loss : {}'.format(
                    step, epoch_loss / step))
                # break
            if save_every is not None and step % save_every == 0:
                checkpoint = Checkpoint(model, optimizer, epoch, step,
                                        None, None)
                checkpoint.save(experiment_dir)
                logging.info(
                    'Saved step chekpoint ...epoch:{} \tstep:{}'.format(epoch, step))
        checkpoint = Checkpoint(model, optimizer, epoch, step, None, None)
        checkpoint.save(experiment_dir)
        logging.info(
            'Saved epoch chekpoint ...epoch:{} \tstep:{}'.format(epoch, step))
        val_acc = '--NA--'
        if valid_generator:
            val_acc = eval_generator(valid_generator, model)
            logging.info('\n-->--> Epoch: {} \tTraining Loss: {} \tValidation Accuracy: {}'.format(
                epoch, epoch_loss, val_acc))
        else:
            logging.info(
                '\n-->--> Epoch: {} \tTraining Loss: {}'.format(epoch, epoch_loss))


def eval_generator(generator, model):
    model.eval()
    y_pred = []
    y_true = []
    count = 0
    loss = 0
    for inputs, lengths, targets, indices in generator:
        count += 1
        model_outputs, _, ret_dict = model.forward(
            inputs, lengths, targets)
        predictions = torch.stack(
            ret_dict['sequence']).squeeze(-1).transpose(0, 1)
        for i in range(targets.size()[0]):
            p = predictions[i].squeeze(-1).data.tolist()
            t = targets[i].squeeze(-1).data[1:].tolist()
            if len(p) == len(t):
                y_pred += p
                y_true += t
    if len(y_true) > 1:
        logging.info('\n\n' + classification_report(y_true, y_pred))
        try:
            logging.info(confusion_matrix(y_true, y_pred))
        except TypeError:
            logging.info('Could not print confusion matrix!')
        acc = accuracy_score(y_true, y_pred)
    else:
        acc = -1
    return acc


def get_extractive_abstractive_gold_and_predicted_summary(predictions, docs, output_idx2word, byte_limit=None):
    assert predictions.size()[0] == len(docs)
    data = []
    position_weights = [0.94, 0.915, 0.88, 0.868, 0.827, 0.805, 0.794, 0.783, 0.78, 0.774, 0.763, 0.756, 0.74, 0.728, 0.724, 0.718, 0.712, 0.7, 0.698, 0.702, 0.694, 0.685, 0.683, 0.682, 0.676, 0.674, 0.673, 0.667, 0.653, 0.65, 0.645, 0.645, 0.643, 0.635, 0.623, 0.598, 0.599, 0.591, 0.598, 0.592, 0.585, 0.576, 0.561, 0.563, 0.563, 0.572, 0.573, 0.574, 0.562, 0.524]
    for idx, doc in enumerate(docs):
        abs_gold_summary = ' .\n'.join([sent for sent in doc[1]])
        ext_gold_summary = ' .\n'.join(
            [sent for sent, label in doc[0]])
        cut_off = min(len(doc[0]), predictions.size()[1])
        pred_sents = []
        pred_scores = []
        selected_sents = []
        sent_nums = []
        for j, tup in enumerate(doc[0][:cut_off]):
            sent = tup[0]
            pred_label = predictions[idx][j].data[0]
            score = scores[idx][j].data[0]
            pred_label = output_idx2word[pred_label]
            if pred_label == '1':
                pred_sents.append(sent)
                pred_scores.append(score)
                sent_nums.append(j + 1)

        if len(pred_sents) == 0:
            # selected_sents = list(map(lambda x: x[0], doc[0][:cut_off]))[:3]
            selected_sents = [''] * 3
            sent_nums = [4] * 3
        else:
            lengths = torch.Tensor([len(sent) for sent in pred_sents])
            lengths /= torch.sum(lengths) + 1
            # tensor_scores = torch.Tensor(pred_scores)
            # tensor_scores /= lengths
            # for s in sorted_scores.tolist():
            #     print('{:.3}'.format(s), end=' ')
            # print('\n')

            tensor_scores = torch.exp(torch.Tensor(pred_scores))
            scores_list = tensor_scores.tolist()
            sorted_scores, indices = torch.sort(tensor_scores, descending=True)
            # Cutting off at 3 sentences at max
            selected_sents = [pred_sents[i] for i in indices.tolist()[:3]]



        try:
            pred_summary = '\n'.join(selected_sents[:3])
        except TypeError:
            pprint(selected_sents)
            print('\n\n\n')
            exit()

        if byte_limit is not None:
            pred_summary = pred_summary[:byte_limit]
            abs_gold_summary = abs_gold_summary[:byte_limit].replace('*', '')
            ext_gold_summary = ext_gold_summary[:byte_limit]
        data.append((abs_gold_summary, ext_gold_summary, pred_summary))
    return data


def ignore_stopwords(text, stopwords):
    ret = [word for word in text.split() if word not in stopwords]
    return ' '.join(ret)


def predict_generator(generator, model, output_idx2word, byte_limit=None):
    """[summary]

    Arguments:
        generator {[type]} -- [description]
        model {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    model.eval()
    count = 0
    for inputs, lengths, targets, indices, docs in generator:
        count += 1
        model_outputs, _, ret_dict = model.forward(
            inputs, lengths, None, teacher_forcing_ratio=0)
        predictions = torch.stack(
            ret_dict['sequence']).squeeze(-1).transpose(0, 1)
        scores = torch.stack(
            ret_dict['score']).squeeze(-1).transpose(0, 1)
        data = get_extractive_abstractive_gold_and_predicted_summary(
            predictions, scores, docs, output_idx2word, byte_limit)
        yield data


def dump_in_ner_format(pickle_file='../../Downloads/Datasets/Summarization/neuralsum/data/data.pkl', text_file='text_file.txt'):
    with open(pickle_file, mode='rb') as file:
        obj = pickle.load(file)
    with open(text_file, mode='w') as file:
        for doc in obj:
            for sent, label in doc:
                file.write(sent + '\t' + str(label) + '\n')
            file.write('\n')


def make_write_safe(text):
    text = text.replace('\u2013', '-')
    text = text.replace('\u2014', '')
    text = text.replace('\u22f1', '')
    # return text
    return text.encode('utf-8', 'ignore').decode('latin-1', 'ignore')


def write_to_file(text, filename, end='\n'):
    with open(filename, mode='w') as file:
        file.write(make_write_safe(text) + end)


def evaluate_rouge(hypothesis_dir, gold_dir):
    r = Rouge155()
    r.system_dir = hypothesis_dir
    r.model_dir = gold_dir
    r.system_filename_pattern = '(\d+)_hypothesis.txt'
    r.model_filename_pattern = '#ID#_reference.txt'
    output = r.convert_and_evaluate()
    return output, r.output_to_dict(output)


def save_count_graph(x, y, filename):
    plt.clf()
    plt.cla()
    plt.close()
    # plt.plot(x, y)
    plt.bar(x, y)
    max_num = max(x)
    plt.xticks(np.arange(1, max_num + 1))
    plt.xlabel('Sentence postion')
    plt.ylabel('Count')
    plt.title('Selection strategy: top 3 by score')
    plt.savefig(filename)


def save_score_graph(x, y, filename):
    plt.clf()
    plt.cla()
    plt.close()
    plt.plot(x, y)
    max_num = max(x)
    plt.xticks(np.arange(1, max_num + 1, 5))
    plt.xlabel('Sentence postion')
    plt.ylabel('Avg. confidence score')
    # plt.title('Variation of confidence score according to sentence postion')
    plt.savefig(filename)
