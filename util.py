import itertools
import math
import pickle
from collections import namedtuple
from pprint import pprint

import numpy as np
import torch
from nltk import sent_tokenize, word_tokenize
from pyrouge import Rouge155
from seq2seq.util.checkpoint import Checkpoint
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from torch import nn
from torch.autograd import Variable

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
        logging.info("loading checkpoint from {}".format(os.path.join(
            expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, checkpoint_name)))
        checkpoint_path = os.path.join(
            expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, checkpoint_name)
    else:
        checkpoint_path = Checkpoint.get_latest_checkpoint(expt_dir)
    return Checkpoint.load(checkpoint_path)


def sents_to_input(sents, vocab, glove_array, max_len):
    sent_vecs = []
    for i, sentence in enumerate(sents):
        word_vecs = [get_vector(word, vocab, glove_array)
                     for word in word_tokenize(sentence)]
        sent_vec = np.mean(np.array(word_vecs, dtype='float32'), axis=0)
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


def get_vector(word, vocab, glove_array):
    idx = vocab['<unk>']
    if word in vocab:
        idx = vocab[word]
    elif word.lower() in vocab:
        idx = vocab[word.lower()]
    return glove_array[idx]


def create_train_batch_from_docs(docs, input_word2idx, output_word2idx, glove_array, max_len, sort=True):
    inputs = []
    lengths = []
    targets = []
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
            sents, input_word2idx, glove_array, max_len)
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


def create_predict_batch_from_docs(docs, input_word2idx, output_word2idx, glove_array, max_len, sort=True):
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
            sents, input_word2idx, glove_array, max_len)
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


def data_generator(data_path, input_word2idx, output_word2idx, glove_array, max_len, batch_size=16, return_docs=False):
    with open(data_path, mode='rb') as file:
        docs = pickle.load(file)

    num_batchs = math.ceil(len(docs) // batch_size)
    for i in range(num_batchs):
        start = i * batch_size
        end = (i + 1) * batch_size
        if not return_docs:
            yield create_train_batch_from_docs(docs[start:end], input_word2idx,
                                               output_word2idx, glove_array, max_len)
        else:
            yield create_predict_batch_from_docs(docs[start:end], input_word2idx,
                                                 output_word2idx, glove_array, max_len)


def train_on_mini_batch(model, optimizer, loss_func, inputs, lengths, targets):
    model.zero_grad()
    # print(lengths[:10])
    # print(targets[0])
    model_outputs, _, ret_dict = model.forward(inputs, lengths, targets)
    loss = 0
    for idx in range(model_outputs.size()[0]):
        loss += loss_func(model_outputs[idx], targets[idx, 1:])
    # print(model_outputs.size())
    # print(targets[:, -1].size())
    # print([x.data[0][0] for x in ret_dict['sequence']])
    # predictions = torch.stack(ret_dict['sequence']).squeeze(-1).transpose(0,1)
    # print(predictions.size())
    # print(targets.data.tolist())
    # loss = loss_func(model_outputs, targets[:, 1:])
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
                                         train_generator_dict['batch_size'])
        if valid_generator_dict is not None:
            valid_generator = data_generator(valid_generator_dict['path'],
                                             valid_generator_dict['input_word2idx'],
                                             valid_generator_dict['output_word2idx'],
                                             valid_generator_dict['glove_array'],
                                             valid_generator_dict['max_len'],
                                             valid_generator_dict['batch_size'])
        step = 0
        epoch_loss = 0
        model.train()
        for inputs, lengths, targets, indices in train_generator:
            step += 1
            epoch_loss += train_on_mini_batch(model,
                                              optimizer, loss_func, inputs, lengths, targets)
            if step % print_every == 0:
                print('Step : {} Avg Step Loss : {}'.format(
                    step, epoch_loss / step))
                # break
            if save_every is not None and step % save_every == 0:
                checkpoint = Checkpoint(model, optimizer, epoch, step,
                                        None, None)
                checkpoint.save(experiment_dir)
                print('Saved step chekpoint ...epoch:{} \tstep:{}'.format(epoch, step))
        if best_epoch_loss > epoch_loss:
            best_epoch_loss = epoch_loss
            checkpoint = Checkpoint(model, optimizer, epoch, step, None, None)
            checkpoint.save(experiment_dir)
            print('Saved epoch chekpoint ...epoch:{} \tstep:{}'.format(epoch, step))
        val_acc = '--NA--'
        if valid_generator:
            val_acc = eval_generator(valid_generator, model)
            print('\n-->--> Epoch: {} \tTraining Loss: {} \tValidation Accuracy: {}'.format(
                epoch, epoch_loss, val_acc))
        else:
            print('\n-->--> Epoch: {} \tTraining Loss: {}'.format(epoch, epoch_loss))


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
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        acc = accuracy_score(y_true, y_pred)
    else:
        acc = -1
    return acc


def get_extractive_abstractive_gold_and_predicted_summary(predictions, docs, output_idx2word, byte_limit=None):
    assert predictions.size()[0] == len(docs)
    data = []
    for idx, doc in enumerate(docs):
        abs_gold_summary = ' .\n'.join([sent for sent in doc[1]])
        ext_gold_summary = ' .\n'.join(
            [sent for sent, label in doc[0] if label])
        cut_off = min(len(doc), predictions.size()[1])
        pred_summary = ''
        for j, tup in enumerate(doc[0][:cut_off]):
            sent = tup[0]
            pred_label = predictions[idx][j].data[0]
            pred_label = output_idx2word[pred_label]
            if pred_label == '1':
                pred_summary += sent + ' .\n'
        if byte_limit is not None:
            pred_summary = pred_summary[:byte_limit]
        data.append((abs_gold_summary, ext_gold_summary, pred_summary))
    return data


def get_abstractive_gold_and_predicted_summary(predictions, docs, output_idx2word):
    assert predictions.size()[0] == len(docs)
    data = []
    for idx, doc in enumerate(docs):
        gold_summary = ' . '.join([sent for sent in doc[1]])
        # print('')
        # print('')
        # print('')
        # print(gold_summary.encode('latin-1', 'ignore').decode('utf-8', 'ignore'))
        # print('')
        # print('')
        # print('')

        # exit()
        cut_off = min(len(doc), predictions.size()[1])
        pred_summary = ''
        for j, tup in enumerate(doc[0][:cut_off]):
            sent = tup[0]
            pred_label = predictions[idx][j].data[0]
            pred_label = output_idx2word[pred_label]
            if pred_label == '1':
                pred_summary += sent + ' . '
        data.append((gold_summary, pred_summary))
    return data


def predict_generator(generator, model, output_idx2word, word_limit=None):
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
            inputs, lengths, targets)
        predictions = torch.stack(
            ret_dict['sequence']).squeeze(-1).transpose(0, 1)
        data = get_extractive_abstractive_gold_and_predicted_summary(
            predictions, docs, output_idx2word, word_limit)
        # if not abstractive:
        # else:
        #     data = get_abstractive_gold_and_predicted_summary(
        #         predictions, docs, output_idx2word)
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
    return text.encode('latin-1', 'ignore').decode('utf-8', 'ignore')


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
