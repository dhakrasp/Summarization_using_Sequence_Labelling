from __future__ import print_function

import os
import pickle
import argparse
from collections import OrderedDict

import numpy as np
import torch
import json
from nltk import sent_tokenize, word_tokenize
from torch import nn
from torch.autograd import Variable
from torch import optim

from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq, MySeq2Seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

from util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', action='store', dest='train_path',
                        help='Path to train data')
    parser.add_argument('--dev_path', action='store', dest='dev_path',
                        help='Path to dev data')
    parser.add_argument('--test_path', action='store', dest='test_path',
                        help='Path to test data')
    parser.add_argument('--glove_path', action='store', dest='glove_path',
                        help='Path to glove npy file')
    parser.add_argument('--tfidf_path', action='store', dest='tfidf_path', default='',
                        help='Path to tfidf json file')
    parser.add_argument('--num_layers', action='store', dest='num_layers', default=2,
                        type=int, help='Number of hidden layers')
    parser.add_argument('--include_length', action='store_true', dest='include_length', default=False,
                        help='Include length as a feature')
    parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./expt',
                        help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                        help='The name of the checkpoint to load, usually an encoded time string')
    parser.add_argument('--resume', action='store_true', dest='resume',
                        default=True,
                        help='Indicates if training has to be resumed from the latest checkpoint')
    parser.add_argument('--mode', action='store', dest='mode',
                        default='test',
                        help='Indicates the mode (train/test)')
    parser.add_argument('--log-level', dest='log_level',
                        default='info',
                        help='Logging level.')
    opt = parser.parse_args()
    log_dir = '{}/logs/'.format(opt.expt_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = '{}/logs/logger_{}'.format(opt.expt_dir, asctime().replace(' ', '_'))
    logging.basicConfig(filename=log_file, format=LOG_FORMAT, level=logging.INFO)
    logging.info(opt)
    logging.info('Use cuda: {}'.format(use_cuda))

    with open(opt.glove_path.replace('.npy', '.json'), mode='r') as file:
        input_word2idx = json.load(file)
    glove_array = np.lib.format.open_memmap(opt.glove_path, dtype='float32')
    if opt.tfidf_path != '':
        with open(opt.tfidf_path, mode='r') as file:
            tfidf_weights_dict = json.load(file)
    else:
        tfidf_weights_dict = None
    if opt.include_length:
        input_dim = glove_array.shape[1] + 1
    else:
        input_dim = glove_array.shape[1]

    hidden_dim = input_dim
    max_len = 50
    batch_size = 128
    output_word2idx = {'NA': 0, '<sos>': 1, '0': 2, '1': 3, '<eos>': 4}
    output_idx2word = {v: k for k, v in output_word2idx.items()}
    if opt.mode.strip() == 'train':
        if opt.resume:
            checkpoint = load_checkpoint(opt.load_checkpoint, opt.expt_dir)
            model = checkpoint.model
            # optimizer = checkpoint.optimizer
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            start_epoch = checkpoint.epoch
            print('Model loaded... {}'.format(checkpoint.path))
        else:
            start_epoch = 0
            enc_config = EncoderConfig(input_size=input_dim, hidden_size=hidden_dim,
                                       bidirectional=True, batch_first=True, num_layers=opt.num_layers, variable_lengths=False, dropout=0.5)
            dec_config = DecoderConfig(vocab_size=len(output_word2idx),
                                       hidden_size=hidden_dim * 2,
                                       max_len=max_len + 1,
                                       bidirectional=True,
                                       num_layers=opt.num_layers,
                                       use_attention=True,
                                       dropout_p=0.5,
                                       input_dropout_p=0.5,
                                       sos_id=output_word2idx['<sos>'],
                                       eos_id=output_word2idx['<eos>'])
            model = MySeq2Seq.MySeq2Seq(enc_config, dec_config)
            for param in model.parameters():
                param.data.uniform_(-0.05, 0.05)
            if use_cuda:
                model.cuda()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_func = torch.nn.NLLLoss()

        train_gen_dict = {'path': opt.train_path,
                          'input_word2idx': input_word2idx,
                          'output_word2idx': output_word2idx,
                          'glove_array': glove_array,
                          'max_len': max_len,
                          'batch_size': batch_size,
                          'include_length': opt.include_length,
                          'tfidf_weights_dict': tfidf_weights_dict}
        valid_gen_dict = {'path': opt.dev_path,
                          'input_word2idx': input_word2idx,
                          'output_word2idx': output_word2idx,
                          'glove_array': glove_array,
                          'max_len': max_len,
                          'batch_size': batch_size,
                          'include_length': opt.include_length,
                          'tfidf_weights_dict': tfidf_weights_dict}
        train_epochs_generator(train_gen_dict,
                               model,
                               optimizer,
                               loss_func,
                               epochs=20,
                               start_epoch=start_epoch,
                               experiment_dir=opt.expt_dir,
                               print_every=2000,
                               valid_generator_dict=valid_gen_dict,
                               save_every=None)

    if opt.mode.strip() == 'test':
        checkpoint = load_checkpoint(opt.load_checkpoint, opt.expt_dir)
        model = checkpoint.model
        optimizer = checkpoint.optimizer
        logging.info('Model loaded...{}'.format(checkpoint.path))
        byte_limit = None
        output_dir = '{}/outputs/byte_limit_{}'.format(
            opt.expt_dir, byte_limit)
        hypothesis_output_dir = '{}/outputs/byte_limit_{}/hypothesis'.format(
            opt.expt_dir, byte_limit)
        abs_reference_output_dir = '{}/outputs/byte_limit_{}/abs_reference'.format(
            opt.expt_dir, byte_limit)
        baseline_hypothesis_output_dir = '{}/outputs/byte_limit_{}/baseline_hypothesis'.format(
            opt.expt_dir, byte_limit)

        test_generator = data_generator(opt.test_path,
                                        input_word2idx,
                                        output_word2idx,
                                        glove_array,
                                        max_len,
                                        batch_size,
                                        opt.include_length,
                                        tfidf_weights_dict,
                                        return_docs=True,
                                        max_docs=None)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(hypothesis_output_dir):
            os.makedirs(hypothesis_output_dir)
        if not os.path.exists(abs_reference_output_dir):
            os.makedirs(abs_reference_output_dir)
        if not os.path.exists(baseline_hypothesis_output_dir):
            os.makedirs(baseline_hypothesis_output_dir)
        count = 0
        for batch_output in predict_generator(test_generator, model, output_idx2word, byte_limit):
            for abs_gold, baseline, pred in batch_output:
                count += 1
                pred_file = '{}/{:0>6}_hypothesis.txt'.format(
                    hypothesis_output_dir, count)
                abs_gold_file = '{}/{:0>6}_reference.txt'.format(
                    abs_reference_output_dir, count)
                baseline_file = '{}/{:0>6}_hypothesis.txt'.format(
                    baseline_hypothesis_output_dir, count)
                write_to_file(pred, pred_file)
                write_to_file(abs_gold, abs_gold_file)
                write_to_file(baseline, baseline_file)

        print('Predicted {} summaries.'.format(count))
        output, output_dict = evaluate_rouge(
            hypothesis_output_dir, abs_reference_output_dir)
        print('\nEvaluation against abstractive reference ...\n')
        try:
            print('\n\n' + output + '\n\n')
        except TypeError:
            logging.info('Could not print rouge scores !!!')
        # Write to a results file
        results_file = open(output_dir + '/results.txt', 'w')
        print('\n\nEvaluation against abstractive reference ...\n', file=results_file)
        print(output, file=results_file)
        results_file.close()
