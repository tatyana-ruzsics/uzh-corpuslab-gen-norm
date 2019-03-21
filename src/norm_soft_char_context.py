#!/usr/bin/env python
# -*- coding: utf-8 -*
"""Trains encoder-decoder model with soft attention.

Usage:
  norm_soft_char_context.py train [--dynet-seed SEED] [--dynet-mem MEM] [--input_format=INPUT_FORMAT]  [--lowercase=LOW] [--pos_split_space] [--conll_format]
    [--char_input=CHAR_INPUT] [--word_input=WORD_INPUT] [--feat_input=FEAT_INPUT] [--hidden=HIDDEN] [--hidden_context=HIDDEN_CONTEXT] [--layers=LAYERS] [--char_vocab_path=VOCAB_PATH_CHAR] [--feat_vocab_path=VOCAB_PATH_FEAT] [--word_vocab_path=VOCAB_PATH_WORD] [--feat_vocab_path_in=VOCAB_PATH_FEAT_IN]
    [--dropout=DROPOUT] [--epochs=EPOCHS] [--patience=PATIENCE] [--optimization=OPTIMIZATION] [--aux_pos_task] [--aux_weight=AUX_WEIGHT] [--pos_feature]
    MODEL_FOLDER --train_path=TRAIN_FILE --dev_path=DEV_FILE
  norm_soft_char_context.py test [--dynet-mem MEM] [--beam=BEAM] [--pred_path=PRED_FILE] [--input_format=INPUT_FORMAT] [--pos_split_space] [--conll_format]
    MODEL_FOLDER --test_path=TEST_FILE [--lowercase=LOW]
  norm_soft_char_context.py ensemble_test [--dynet-mem MEM] [--beam=BEAM] [--pred_path=PRED_FILE] [--input_format=INPUT_FORMAT] [--pos_split_space] [--conll_format]
    ED_MODEL_FOLDER MODEL_FOLDER --test_path=TEST_FILE [--lowercase=LOW]
    

Arguments:
MODEL_FOLDER  save/read model folder where also eval results are written to, possibly relative to RESULTS_FOLDER
ED_MODEL_FOLDER  ED model(s) folder, possibly relative to RESULTS_FOLDER, coma-separated

Options:
  -h --help                     show this help message and exit
  --dynet-seed SEED             DyNET seed
  --dynet-mem MEM               allocates MEM bytes for DyNET [default: 500]
  --char_input=CHAR_INPUT       charachters input vector dimensions [default: 100]
  --word_input=WORD_INPUT        word feature input vector dimension [default: 128]
  --feat_input=FEAT_INPUT       feature input vector dimension [default: 50]
  --hidden=HIDDEN               hidden layer dimensions for encoder/decoder LSTM [default: 200]
  --hidden_context=HIDDEN_CONTEXT  hidden layer dimensions for context LSTM [default: 100]
  --layers=LAYERS               amount of layers in LSTMs  [default: 1]
  --dropout=DROPOUT             amount of dropout in LSTMs [default: 0]
  --epochs=EPOCHS               number of training epochs   [default: 30]
  --patience=PATIENCE           patience for early stopping [default: 10]
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA [default: SGD]
  --train_path=TRAIN_FILE       train set path, possibly relative to DATA_FOLDER, only for training
  --dev_path=DEV_FILE           dev set path, possibly relative to DATA_FOLDER, only for training
  --test_path=TEST_FILE         test set path, possibly relative to DATA_FOLDER, only for evaluation
  --char_vocab_path=VOCAB_PATH_CHAR  char vocab path, possibly relative to RESULTS_FOLDER [default: char_vocab.txt]
  --feat_vocab_path=VOCAB_PATH_FEAT  feat vocab path, possibly relative to RESULTS_FOLDER [default: feat_vocab.txt]
  --feat_vocab_path_in=VOCAB_PATH_FEAT_IN  feat vocab path, possibly relative to RESULTS_FOLDER [default: feat_vocab_in.txt]
  --word_vocab_path=VOCAB_PATH_WORD  word vocab path, possibly relative to RESULTS_FOLDER [default: word_vocab.txt]
  --beam=BEAM                   beam width [default: 1]
  --pred_path=PRED_FILE         name for predictions file in the test mode [default: best.test]
  --input_format=INPUT_FORMAT   coma-separated list of input, output, features columns [default: 0,1,2]
  --lowercase=LOW               use lowercased data [default: True]
  --pos_split_space             use space to split POS tag features, the default is '+'
  --aux_pos_task                use auxilary task to predict POS tag
  --aux_weight=AUX_WEIGH        weight of auxilary loss [default: 0.2]
  --pos_feature                 use POS tag as a feature [default: False]
  --conll_format                use conll format
"""

from __future__ import division
from docopt import docopt
import os
import sys
import codecs
import random
import progressbar
import time
from collections import Counter, defaultdict

import time
import copy
import dynet as dy
import numpy as np
import os

from common import BEGIN_CHAR,STOP_CHAR,UNK_CHAR, SRC_FOLDER,RESULTS_FOLDER,DATA_FOLDER,check_path, write_pred_file, write_param_file, write_eval_file
from vocab_builder import build_vocabulary, Vocab
from norm_soft import SoftDataSet

MAX_PRED_SEQ_LEN = 50 # option
OPTIMIZERS = {'ADAM'    : lambda m: dy.AdamTrainer(m, alpha=0.0001, #common
                                                   beta_1=0.9, beta_2=0.999, eps=1e-8),
    'SGD'     : dy.SimpleSGDTrainer,
        'ADADELTA': dy.AdadeltaTrainer}


### IO handling and evaluation

def load_data_pos_ext(filename, input_format, lowercase=False, split_by_space=False, conll_format=False):
    """ Load data from file
        
        filename (str):   file containing input/output data, structure (tab-separated):
        input    output  POS
        return tuple (output, input) where each element is a list
        where each element in the list is one example
        """
    
    print 'loading data from file:', filename
    
    input_col, output_col, feat_col = input_format
    
    with codecs.open(filename, encoding='utf8') as f:
        sent = []
        inputs, outputs, features, features_in = [], [], [], []
        for i,line in enumerate(f):
            if not conll_format:
                if not (len(line.strip())==0):
                    try:
                        splt = line.strip().split('\t')
    #                    print splt
                        inputs.append(splt[input_col].lower() if lowercase else splt[input_col])
                        outputs.append(splt[output_col].lower() if lowercase else splt[output_col])
                        # pos tags to predict - can be concatenated
                        features.append(splt[feat_col])
                        # pos tags as an input to the normalization
                        features_in.append(splt[feat_col].split() if split_by_space else splt[feat_col].split('+'))

                    except:
                        print u"bad line: {}, {}".format(i,line)
                else:
                    if i!=0:
                        if inputs!=[]:
                            tup = (inputs, outputs, features, features_in)
                            sent.append(tup)
                            inputs, outputs, features, features_in = [], [], [], []
#                    else:
#                        print i,line
            else:
                if not (len(line.strip())==0):
                    # collect tokens in one sentence
                    if not line.startswith('#'):
                        splt = line.strip().split('\t')
                        if not(splt[feat_col]=='PUNCT' or any(c in '@+._/1234567890' for c in splt[output_col]+splt[input_col])):
                            try:
                                splt = line.strip().split('\t')
                                #                    print splt
                                inputs.append(splt[input_col].lower() if lowercase else splt[input_col])
                                outputs.append(splt[output_col].lower() if lowercase else splt[output_col])
                                # pos tags to predict - can be concatenated
                                features.append(splt[feat_col])
                                # pos tags as an input to the normalization
                                features_in.append(splt[feat_col].split() if split_by_space else splt[feat_col].split('+'))
                            
                            except:
                                print u"bad line: {}, {}".format(i,line)
                else:
                    if i!=0:
                        # collect sentence
                        if inputs!=[]:
                            tup = (inputs, outputs, features, features_in)
                            sent.append(tup)
                            inputs, outputs, features, features_in = [], [], [], []
                    
        if inputs!=[]:
            tup = (inputs, outputs, features, features_in)
            sent.append(tup)
    print 'found', len(sent), 'examples'
    print u'example of data: {}'.format(sent[:4] )
    return sent

def log_to_file(log_file_name, e, avg_train_loss, train_accuracy, dev_accuracy):
    # if first write, add headers
    if e == 0:
        log_to_file(log_file_name, 'epoch', 'avg_train_loss', 'train_accuracy', 'dev_accuracy')
    
    with open(log_file_name, "a") as logfile:
        logfile.write("{}\t{}\t{}\t{}\n".format(e, avg_train_loss, train_accuracy, dev_accuracy))

class SoftDataSetCont(object):
    def __init__(self, sents):
        self.inputs = []
        self.outputs = []
        self.features = []
        self.features_in = []
        for s in sents:
            inputs,outputs,features,features_in = s
            self.inputs.extend(inputs)
            self.outputs.extend(outputs)
            self.features.extend(features)
            self.features_in.extend(features_in)
        print self.inputs[:2]
        self.dataset = sents
        self.length = len(self.dataset)
            
    def iter(self, indices=None, shuffle=False):
#        zipped = zip(*self.dataset)
        zipped = self.dataset
        if indices or shuffle:
            if not indices:
                indices = range(self.length)
            elif isinstance(indices, int):
                indices = range(indices)
            else:
                assert isinstance(indices, (list, tuple))
            if shuffle:
                random.shuffle(indices)
            zipped = [zipped[i] for i in indices]
        return zipped
    
    @classmethod
    def from_file(cls, path, input_format, *args, **kwargs):
        # returns a `SoftDataSet` with fields: inputs, outputs
        sents = load_data_pos_ext(path, input_format, *args, **kwargs)
        return cls(sents)



class SoftAttention(object):
    def __init__(self, pc, model_hyperparams, best_model_path=None):
        
        self.hyperparams = model_hyperparams
        
        print 'Loading char vocabulary from {}:'.format(self.hyperparams['CHAR_VOCAB_PATH'])
        self.char_vocab = Vocab.from_file(self.hyperparams['CHAR_VOCAB_PATH'])
        print 'Loading feature vocabulary from {}:'.format(self.hyperparams['FEAT_VOCAB_PATH'])
        self.feat_vocab = Vocab.from_file(self.hyperparams['FEAT_VOCAB_PATH'])
#        print 'Loading word vocabulary from {}:'.format(self.hyperparams['WORD_VOCAB_PATH'])
#        self.word_vocab = Vocab.from_file(self.hyperparams['WORD_VOCAB_PATH'])
        print 'Loading feature vocabulary from {}:'.format(self.hyperparams['FEAT_VOCAB_PATH_IN'])
        self.feat_vocab_in = Vocab.from_file(self.hyperparams['FEAT_VOCAB_PATH_IN'])
        
        self.BEGIN   = self.char_vocab.w2i[BEGIN_CHAR]
        self.STOP   = self.char_vocab.w2i[STOP_CHAR]
        self.UNK       = self.char_vocab.w2i[UNK_CHAR]
#        self.UNK_WORD  = self.word_vocab.w2i[UNK_CHAR]
        self.hyperparams['VOCAB_SIZE_CHAR'] = self.char_vocab.size()
#        self.hyperparams['VOCAB_SIZE_WORD'] = self.word_vocab.size()
        self.hyperparams['VOCAB_SIZE_FEAT'] = self.feat_vocab.size()
        self.hyperparams['VOCAB_SIZE_FEAT_IN'] = self.feat_vocab_in.size()
        
        print 'Model Hypoparameters:'
        for k, v in self.hyperparams.items():
            print '{:20} = {}'.format(k, v)
        print
        
        print self.hyperparams['AUX_POS_TASK']
        
        self.build_model(pc, best_model_path)
        
    def build_model(self, pc, best_model_path):
        
        if best_model_path:
            print 'Loading model from: {}'.format(best_model_path)
            if not self.hyperparams['AUX_POS_TASK']:
#                self.fbuffRNN, self.bbuffRNN, self.fbuffRNN_cont, self.bbuffRNN_cont, self.CHAR_VOCAB_LOOKUP, self.WORD_VOCAB_LOOKUP, self.decoder, self.R, self.bias, self.W_c, self.W__a, self.U__a,  self.v__a = dy.load(best_model_path, pc)
                if not self.hyperparams['POS_FEATURE']:
                    self.fbuffRNN, self.bbuffRNN, self.fbuffRNN_cont, self.bbuffRNN_cont, self.CHAR_VOCAB_LOOKUP, self.decoder, self.R, self.bias, self.W_c, self.W__a, self.U__a,  self.v__a = dy.load(best_model_path, pc)
                else:
                    self.fbuffRNN, self.bbuffRNN, self.fbuffRNN_cont, self.bbuffRNN_cont, self.CHAR_VOCAB_LOOKUP, self.FEAT_VOCAB_LOOKUP, self.decoder, self.R, self.bias, self.W_c, self.W__a, self.U__a,  self.v__a = dy.load(best_model_path, pc)
            else:
#                self.fbuffRNN, self.bbuffRNN, self.fbuffRNN_cont, self.bbuffRNN_cont, self.CHAR_VOCAB_LOOKUP, self.WORD_VOCAB_LOOKUP, self.decoder, self.R, self.bias, self.W_c, self.W__a, self.U__a,  self.v__a, self.R_pos, self.bias_pos, self.W_c_pos = dy.load(best_model_path, pc)
                if not self.hyperparams['POS_FEATURE']:
                    self.fbuffRNN, self.bbuffRNN, self.fbuffRNN_cont, self.bbuffRNN_cont, self.CHAR_VOCAB_LOOKUP, self.decoder, self.R, self.bias, self.W_c, self.W__a, self.U__a,  self.v__a, self.R_pos, self.bias_pos, self.W_c_pos = dy.load(best_model_path, pc)
                else:
                    self.fbuffRNN, self.bbuffRNN, self.fbuffRNN_cont, self.bbuffRNN_cont, self.CHAR_VOCAB_LOOKUP, self.FEAT_VOCAB_LOOKUP, self.decoder, self.R, self.bias, self.W_c, self.W__a, self.U__a,  self.v__a, self.R_pos, self.bias_pos, self.W_c_pos = dy.load(best_model_path, pc)
        else:
            # BiLSTM for input
            self.fbuffRNN  = dy.CoupledLSTMBuilder(self.hyperparams['LAYERS'], self.hyperparams['INPUT_DIM_CHAR'], self.hyperparams['HIDDEN_DIM'], pc)
            self.bbuffRNN  = dy.CoupledLSTMBuilder(self.hyperparams['LAYERS'], self.hyperparams['INPUT_DIM_CHAR'], self.hyperparams['HIDDEN_DIM'], pc)
            
            # BiLSTM for context
#            self.fbuffRNN_cont  = dy.CoupledLSTMBuilder(self.hyperparams['LAYERS'], 4*self.hyperparams['HIDDEN_DIM']+self.hyperparams['INPUT_DIM_WORD'], self.hyperparams['HIDDEN_DIM_CONTEXT'], pc)
#            self.bbuffRNN_cont  = dy.CoupledLSTMBuilder(self.hyperparams['LAYERS'], 4*self.hyperparams['HIDDEN_DIM']+self.hyperparams['INPUT_DIM_WORD'], self.hyperparams['HIDDEN_DIM_CONTEXT'], pc)
            self.fbuffRNN_cont  = dy.CoupledLSTMBuilder(self.hyperparams['LAYERS'], 2*self.hyperparams['HIDDEN_DIM'], self.hyperparams['HIDDEN_DIM_CONTEXT'], pc)
            self.bbuffRNN_cont  = dy.CoupledLSTMBuilder(self.hyperparams['LAYERS'], 2*self.hyperparams['HIDDEN_DIM'], self.hyperparams['HIDDEN_DIM_CONTEXT'], pc)
            
            # embedding lookups for vocabulary (chars)
            self.CHAR_VOCAB_LOOKUP  = pc.add_lookup_parameters((self.hyperparams['VOCAB_SIZE_CHAR'], self.hyperparams['INPUT_DIM_CHAR']))
            
            # embedding lookups for vocabulary (words)
#            self.WORD_VOCAB_LOOKUP  = pc.add_lookup_parameters((self.hyperparams['VOCAB_SIZE_WORD'], self.hyperparams['INPUT_DIM_WORD']))
            
            if self.hyperparams['POS_FEATURE']:
                self.FEAT_VOCAB_LOOKUP = pc.add_lookup_parameters((self.hyperparams['VOCAB_SIZE_FEAT_IN'], self.hyperparams['INPUT_DIM_FEAT']))

            # attention MLPs - Loung-style with extra v_a from Bahdanau
            # concatenation layer for h_input (2*hidden_dim), h_output (hidden_dim)
            self.W__a = pc.add_parameters((self.hyperparams['HIDDEN_DIM'], self.hyperparams['HIDDEN_DIM']))
    
            # concatenation layer for h (hidden dim), c (2 * hidden_dim)
            self.U__a = pc.add_parameters((self.hyperparams['HIDDEN_DIM'], 2 * self.hyperparams['HIDDEN_DIM']))
        
            # concatenation layer for h_input (2*hidden_dim), h_output (hidden_dim)
            self.v__a = pc.add_parameters((1, self.hyperparams['HIDDEN_DIM']))
            
            
            # decoder LSTM
            self.decoder = dy.CoupledLSTMBuilder(self.hyperparams['LAYERS'], self.hyperparams['INPUT_DIM_CHAR'], self.hyperparams['HIDDEN_DIM'], pc)

            # softmax parameters
            if not self.hyperparams['POS_FEATURE']:
                # concatenation layer
                self.W_c = pc.add_parameters((3 * self.hyperparams['HIDDEN_DIM'] + 2 * self.hyperparams['HIDDEN_DIM_CONTEXT'], 3 * self.hyperparams['HIDDEN_DIM'] + 2 * self.hyperparams['HIDDEN_DIM_CONTEXT']))
                # softmax layer
                self.R = pc.add_parameters((self.hyperparams['VOCAB_SIZE_CHAR'], 3 * self.hyperparams['HIDDEN_DIM'] + 2 * self.hyperparams['HIDDEN_DIM_CONTEXT']))
            else:
                # concatenation layer
                self.W_c = pc.add_parameters((3 * self.hyperparams['HIDDEN_DIM'] + 2 * self.hyperparams['HIDDEN_DIM_CONTEXT'] + self.hyperparams['INPUT_DIM_FEAT'], 3 * self.hyperparams['HIDDEN_DIM'] + 2 * self.hyperparams['HIDDEN_DIM_CONTEXT'] + self.hyperparams['INPUT_DIM_FEAT']))
                # softmax layer
                self.R = pc.add_parameters((self.hyperparams['VOCAB_SIZE_CHAR'], 3 * self.hyperparams['HIDDEN_DIM'] + 2 * self.hyperparams['HIDDEN_DIM_CONTEXT'] + self.hyperparams['INPUT_DIM_FEAT']))
            self.bias = pc.add_parameters(self.hyperparams['VOCAB_SIZE_CHAR'])
            
            
            if self.hyperparams['AUX_POS_TASK']:
                # concatenation layer - auxiliry pos tagging task
                self.W_c_pos = pc.add_parameters(( 2 * self.hyperparams['HIDDEN_DIM_CONTEXT'],  2 * self.hyperparams['HIDDEN_DIM_CONTEXT']))
                # softmax layer - auxiliry pos tagging task
                self.R_pos = pc.add_parameters((self.hyperparams['VOCAB_SIZE_FEAT'], 2 * self.hyperparams['HIDDEN_DIM_CONTEXT']))
                self.bias_pos = pc.add_parameters(self.hyperparams['VOCAB_SIZE_FEAT'])
                
                
        
        print 'Model dimensions:'
        print ' * VOCABULARY CHAR EMBEDDING LAYER: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['VOCAB_SIZE_CHAR'], self.hyperparams['INPUT_DIM_CHAR'])
#        print ' * VOCABULARY WORD EMBEDDING LAYER: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['VOCAB_SIZE_WORD'], self.hyperparams['INPUT_DIM_WORD'])
        if self.hyperparams['POS_FEATURE']:
            print ' * FEATURES VOCABULARY EMBEDDING LAYER: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['VOCAB_SIZE_FEAT_IN'], self.hyperparams['INPUT_DIM_FEAT'])
        print
        print ' * CONTEXT ENCODER biLSTM: IN-DIM: {}, OUT-DIM: {}'.format(4*self.hyperparams['HIDDEN_DIM'], self.hyperparams['HIDDEN_DIM_CONTEXT'])
        print ' * DECODER LSTM: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['INPUT_DIM_CHAR'], self.hyperparams['HIDDEN_DIM'])
        print ' * ENCODER LSTM: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['INPUT_DIM_CHAR'], self.hyperparams['HIDDEN_DIM'])
        print ' All LSTMs have {} layer(s)'.format(self.hyperparams['LAYERS'])
        print
        if not self.hyperparams['POS_FEATURE']:
            print ' * SOFTMAX: IN-DIM: {}, OUT-DIM: {}'.format(3 * self.hyperparams['HIDDEN_DIM'] + 2 *self.hyperparams['HIDDEN_DIM_CONTEXT'], self.hyperparams['VOCAB_SIZE_CHAR'])
        else:
            print ' * SOFTMAX: IN-DIM: {}, OUT-DIM: {}'.format(3 * self.hyperparams['HIDDEN_DIM'] + 2 *self.hyperparams['HIDDEN_DIM_CONTEXT'] + self.hyperparams['INPUT_DIM_FEAT'], self.hyperparams['VOCAB_SIZE_CHAR'])
        if self.hyperparams['AUX_POS_TASK']:
            print ' * POS SOFTMAX: IN-DIM: {}, OUT-DIM: {}'.format(2 * self.hyperparams['HIDDEN_DIM_CONTEXT'], self.hyperparams['VOCAB_SIZE_FEAT'])
        print

    def save_model(self, best_model_path):
        if not self.hyperparams['AUX_POS_TASK']:
#            dy.save(best_model_path, [self.fbuffRNN, self.bbuffRNN, self.fbuffRNN_cont, self.bbuffRNN_cont, self.CHAR_VOCAB_LOOKUP, self.WORD_VOCAB_LOOKUP, self.decoder, self.R, self.bias, self.W_c, self.W__a, self.U__a,  self.v__a])
            if not self.hyperparams['POS_FEATURE']:
                dy.save(best_model_path, [self.fbuffRNN, self.bbuffRNN, self.fbuffRNN_cont, self.bbuffRNN_cont, self.CHAR_VOCAB_LOOKUP, self.decoder, self.R, self.bias, self.W_c, self.W__a, self.U__a,  self.v__a])
            else:
                dy.save(best_model_path, [self.fbuffRNN, self.bbuffRNN, self.fbuffRNN_cont, self.bbuffRNN_cont, self.CHAR_VOCAB_LOOKUP, self.FEAT_VOCAB_LOOKUP, self.decoder, self.R, self.bias, self.W_c, self.W__a, self.U__a,  self.v__a])
        else:
#            dy.save(best_model_path, [self.fbuffRNN, self.bbuffRNN, self.fbuffRNN_cont, self.bbuffRNN_cont, self.CHAR_VOCAB_LOOKUP, self.WORD_VOCAB_LOOKUP, self.decoder, self.R, self.bias, self.W_c, self.W__a, self.U__a,  self.v__a, self.R_pos, self.bias_pos, self.W_c_pos])
            if not self.hyperparams['POS_FEATURE']:
                dy.save(best_model_path, [self.fbuffRNN, self.bbuffRNN, self.fbuffRNN_cont, self.bbuffRNN_cont, self.CHAR_VOCAB_LOOKUP, self.decoder, self.R, self.bias, self.W_c, self.W__a, self.U__a,  self.v__a, self.R_pos, self.bias_pos, self.W_c_pos])
            else:
                dy.save(best_model_path, [self.fbuffRNN, self.bbuffRNN, self.fbuffRNN_cont, self.bbuffRNN_cont, self.CHAR_VOCAB_LOOKUP, self.FEAT_VOCAB_LOOKUP, self.decoder, self.R, self.bias, self.W_c, self.W__a, self.U__a,  self.v__a, self.R_pos, self.bias_pos, self.W_c_pos])

    def bilstm_transduce(self, encoder_frnn, encoder_rrnn, input_char_vecs):
        # returns the encoding for each element of the input sequence as a concat of the fwd and bwd LSTM-s
        
        # BiLSTM forward pass
        s_0 = encoder_frnn.initial_state()
        s = s_0
        frnn_outputs = []
        for c in input_char_vecs:
            s = s.add_input(c)
            frnn_outputs.append(s.output())
        
        # BiLSTM backward pass
        s_0 = encoder_rrnn.initial_state()
        s = s_0
        rrnn_outputs = []
        for c in reversed(input_char_vecs):
            s = s.add_input(c)
            rrnn_outputs.append(s.output())
        
        # BiLTSM outputs
        blstm_outputs = []
        for i in xrange(len(input_char_vecs)):
            blstm_outputs.append(dy.concatenate([frnn_outputs[i], rrnn_outputs[len(input_char_vecs) - i - 1]]))
        
        return blstm_outputs
    
    def bilstm_encode(self, encoder_frnn, encoder_rrnn, input_char_vecs):
        # returns the encoding for the input sequence as a concat of the last fwd and last bwd LSTM-s vectors
        
        # BiLSTM forward pass
        s_0 = encoder_frnn.initial_state()
        s = s_0
        for c in input_char_vecs:
            s = s.add_input(c)
        fwd = s.output()
            
        # BiLSTM backward pass
        s_0 = encoder_rrnn.initial_state()
        s = s_0
        for c in reversed(input_char_vecs):
            s = s.add_input(c)
        bwd = s.output()

        # BiLTSM output
        blstm_output = dy.concatenate([fwd,bwd])

        return blstm_output

    def param_init(self, inputs): #initialize parameters for current cg with the current input
    
        R = dy.parameter(self.R)   # from parameters to expressions
        bias = dy.parameter(self.bias)
        W_c = dy.parameter(self.W_c)
        W__a = dy.parameter(self.W__a)
        U__a = dy.parameter(self.U__a)
        v__a = dy.parameter(self.v__a)
        
        self.cg_params = (R, bias, W_c, W__a, U__a, v__a) # params for current cg and input
        
        self.biencoder = []
        input_emb_context = []
        for _input in inputs:
            
            # biLSTM encoder of input string
            input = [BEGIN_CHAR] + [c for c in _input] + [STOP_CHAR]
            
            input_emb = []
            for char_ in input:
                char_id = self.char_vocab.w2i.get(char_, self.UNK)
                char_embedding = self.CHAR_VOCAB_LOOKUP[char_id]
                input_emb.append(char_embedding)
            word_char_encoding = self.bilstm_transduce(self.fbuffRNN, self.bbuffRNN, input_emb)
            self.biencoder.append(word_char_encoding)
    
#            word_id = self.word_vocab.w2i.get(_input, self.UNK_WORD)
#            word_embedding = self.WORD_VOCAB_LOOKUP[word_id]
#            input_emb_context.append(dy.concatenate([word_embedding, word_char_encoding[0], word_char_encoding[-1]]))
            input_emb_context.append(self.bilstm_encode(self.fbuffRNN, self.bbuffRNN, input_emb))
 
        self.bicontext = self.bilstm_transduce(self.fbuffRNN_cont, self.bbuffRNN_cont, input_emb_context)
        
        if self.hyperparams['AUX_POS_TASK']:
            R_pos = dy.parameter(self.R_pos)
            bias_pos = dy.parameter(self.bias_pos)
            W_c_pos = dy.parameter(self.W_c_pos)
            self.cg_params_pos = (R_pos, bias_pos, W_c_pos)
    def param_init_feat(self, pos_features):
    # embedding of features for the current sentence at train time
        self.feat_emb = []
        for position in range(len(pos_features)):
            feat_emb = []
            for f in pos_features[position]:
                f_id = self.feat_vocab_in.w2i.get(f, self.UNK)
                feat_emb.append(self.FEAT_VOCAB_LOOKUP[f_id])
            self.feat_emb.append(dy.average(feat_emb))
                
    def param_init_feat_testtime(self, inputs):
    # embedding of features for the current sentence at test time
        self.feat_emb = []
        for position in range(len(inputs)):
            log_probs =  dy.log(self.predict_pos(position))
            pred_pos_id = np.argmax(log_probs.npvalue())
            pred_pos = self.feat_vocab.i2w.get(pred_pos_id, self.UNK)
            pred_pos_tags = pred_pos.split() if self.hyperparams['POS_SPLIT_SPACE'] else pred_pos.split('+')
            feat_emb = []
            for f in pred_pos_tags:
                f_id = self.feat_vocab_in.w2i.get(f, self.UNK)
                feat_emb.append(self.FEAT_VOCAB_LOOKUP[f_id])
            self.feat_emb.append(dy.average(feat_emb))

    def reset_decoder(self):
        self.s = self.decoder.initial_state()
        self.s = self.s.add_input(self.CHAR_VOCAB_LOOKUP[self.BEGIN])

    def predict_next(self, position, scores=False):
        (R, bias, W_c, W__a, U__a, v__a) = self.cg_params

        # soft attention vector
        att_scores = [v__a * dy.tanh(W__a * self.s.output() + U__a * h_input) for h_input in self.biencoder[position]]
        alphas = dy.softmax(dy.concatenate(att_scores))
        c = dy.esum([h_input * dy.pick(alphas, j) for j, h_input in enumerate(self.biencoder[position])])
            
        # softmax over vocabulary
        if not self.hyperparams['POS_FEATURE']:
            h_output = dy.tanh(W_c * dy.concatenate([self.s.output(), c, self.bicontext[position]]))
        else:
            h_output = dy.tanh(W_c * dy.concatenate([self.s.output(), c, self.bicontext[position], self.feat_emb[position]]))

        if not scores:
            return dy.softmax(R * h_output + bias)
        else:
            return R * h_output + bias

    def predict_next_(self, state, position, scores=False, *args, **kwargs):
        (R, bias, W_c, W__a, U__a, v__a) = self.cg_params
        
        # soft attention vector
        att_scores = [v__a * dy.tanh(W__a * state.output() + U__a * h_input) for h_input in self.biencoder[position]]
        alphas = dy.softmax(dy.concatenate(att_scores))
        c = dy.esum([h_input * dy.pick(alphas, j) for j, h_input in enumerate(self.biencoder[position])])
        
        # softmax over vocabulary
        if not self.hyperparams['POS_FEATURE']:
            h_output = dy.tanh(W_c * dy.concatenate([state.output(), c, self.bicontext[position]]))
        else:
            h_output = dy.tanh(W_c * dy.concatenate([state.output(), c, self.bicontext[position], self.feat_emb[position]]))
        if not scores:
#            print 'probs:'
            return dy.softmax(R * h_output + bias)
        else:
#            print 'scores:'
            return R * h_output + bias

    def predict_pos(self, pos):
        (R_pos, bias_pos, W_c_pos) = self.cg_params_pos
        h_output_pos = dy.tanh(W_c_pos * self.bicontext[pos])
        return dy.softmax(R_pos * h_output_pos + bias_pos)

    def consume_next(self, pred_id):
        self.s = self.s.add_input(self.CHAR_VOCAB_LOOKUP[pred_id])
    
    def consume_next_(self, state, pred_id):
        new_state = state.add_input(self.CHAR_VOCAB_LOOKUP[pred_id])
        return new_state

    def train_aux(self, inputs, _true_outputs, pos_features, pos_features_in, aux_weight):
        self.param_init(inputs)
        if self.hyperparams['POS_FEATURE']:
            self.param_init_feat(pos_features_in)
        sent_losses = 0
        #        print 'new sent'
        for position,(input,_true_output, pos_feature) in enumerate(zip(inputs, _true_outputs, pos_features)):
            self.reset_decoder()
            true_output = [self.char_vocab.w2i[a] for a in _true_output]
            true_output += [self.STOP]
            
            losses = []
            for pred_id in true_output:
                probs = self.predict_next(position)
                losses.append(-dy.log(dy.pick(probs, pred_id)))
                self.consume_next(pred_id)
            total_word_loss = dy.average(losses)
            
            pos_feature_id = self.feat_vocab.w2i[pos_feature]
            probs_pos = self.predict_pos(position)
            pos_loss = -dy.log(dy.pick(probs_pos, pos_feature_id))
                               
            sent_losses = sent_losses + total_word_loss + dy.scalarInput(aux_weight) * pos_loss
        #            import pdb; pdb.set_trace()
        #            print input, sent_losses.value()
        #        total_sent_loss = dy.esum(sent_losses) #if len(sent_losses)>1 else sent_losses[0]
        return sent_losses
            
    def train_unsup(self, inputs, _true_outputs, pos_features, pos_features_in, *args):
        self.param_init(inputs)
        if self.hyperparams['POS_FEATURE']:
            self.param_init_feat(pos_features_in)
        sent_losses = 0
#        print 'new sent'
        for position,(input,_true_output) in enumerate(zip(inputs, _true_outputs)):
            self.reset_decoder()
            true_output = [self.char_vocab.w2i[a] for a in _true_output]
            true_output += [self.STOP]
            losses = []
            for pred_id in true_output:
                probs = self.predict_next(position)
                losses.append(-dy.log(dy.pick(probs, pred_id)))
                self.consume_next(pred_id)
            total_word_loss = dy.average(losses)
            sent_losses = sent_losses + total_word_loss
#            import pdb; pdb.set_trace()
#            print input, sent_losses.value()
#        total_sent_loss = dy.esum(sent_losses) #if len(sent_losses)>1 else sent_losses[0]
        return sent_losses

#    def predict_greedy(self, input, features):
#        self.param_init(input, features)
#        output = []
#        while not len(output) == MAX_PRED_SEQ_LEN:
#            probs = self.predict_next()
#            pred_id = np.argmax(probs.npvalue())
#            if pred_id == self.STOP:
#                break
#            else:
#                pred_char = self.vocab.i2w.get(pred_id,UNK_CHAR)
#                output.append(pred_char)
#                self.consume_next(pred_id)
#        output = u''.join(output)
#        return output

    @staticmethod
    def _smallest(matrix, k, only_first_row=False):
        """Find k smallest elements of a matrix.
            Parameters
            ----------
            matrix : :class:`np.ndarray`
            The matrix.
            k : int
            The number of smallest elements required.
            Returns
            -------
            Tuple of ((row numbers, column numbers), values).
            """
        #flatten = matrix.flatten()
        if only_first_row:
            flatten = matrix[:1, :].flatten()
        else:
            flatten = matrix.flatten()
        args = np.argpartition(flatten, k)[:k]
        args = args[np.argsort(flatten[args])]
        return np.unravel_index(args, matrix.shape), flatten[args]

    def predict(self, input, position, beam_size, ignore_first_eol=False, predict_pos=False):
        """Performs beam search.
            If the beam search was not compiled, it also compiles it.
            Parameters
            ----------
            max_length : int
            Maximum sequence length, the search stops when it is reached.
            ignore_first_eol : bool, optional
            When ``True``, the end if sequence symbol generated at the
            first iteration are ignored. This useful when the sequence
            generator was trained on data with identical symbols for
            sequence start and sequence end.
            as_arrays : bool, optional
            If ``True``, the internal representation of search results
            is returned, that is a (matrix of outputs, mask,
            costs of all generated outputs) tuple.
            Returns
            -------
            outputs : list of lists of ints
            A list of the `beam_size` best sequences found in the order
            of decreasing likelihood.
            costs : list of floats
            A list of the costs for the `outputs`, where cost is the
            negative log-likelihood.
            """
        
        self.reset_decoder()
#        self.param_init(input, features)
        states = [self.s] * beam_size
        # This array will store all generated outputs, including those from
        # previous step and those from already finished sequences.
        all_outputs = np.full(shape=(1,beam_size),fill_value=self.BEGIN,dtype = int)
        all_masks = np.ones_like(all_outputs, dtype=float) # whether predicted symbol is self.STOP
        all_costs = np.zeros_like(all_outputs, dtype=float) # the cumulative cost of predictions
        
        for i in range(MAX_PRED_SEQ_LEN):
            if all_masks[-1].sum() == 0:
                break
        
            # We carefully hack values of the `logprobs` array to ensure
            # that all finished sequences are continued with `eos_symbol`.
            logprobs = np.array([-dy.log_softmax(self.predict_next_(s, position, scores=True)).npvalue() for s in states])
#            print logprobs
#            print all_masks[-1, :, None]
            next_costs = (all_costs[-1, :, None] + logprobs * all_masks[-1, :, None]) #take last row of cumul prev costs and turn into beam_size X 1 matrix, take logprobs distributions for unfinished hypos only and add it (elem-wise) with the array of prev costs; result: beam_size x vocab_len matrix of next costs
            (finished,) = np.where(all_masks[-1] == 0) # finished hypos have all their cost on the self.STOP symbol
            next_costs[finished, :self.STOP] = np.inf
            next_costs[finished, self.STOP + 1:] = np.inf
            
            # indexes - the hypos from prev step to keep, outputs - the next step prediction, chosen cost - cost of predicted symbol
            (indexes, outputs), chosen_costs = self._smallest(next_costs, beam_size, only_first_row=i == 0)
#            print outputs
            # Rearrange everything
            new_states = (states[ind] for ind in indexes)
            all_outputs = all_outputs[:, indexes]
            all_masks = all_masks[:, indexes]
            all_costs = all_costs[:, indexes]
            
            # Record chosen output and compute new states
            states = [self.consume_next_(s,pred_id) for s,pred_id in zip(new_states, outputs)]
            all_outputs = np.vstack([all_outputs, outputs[None, :]])
            all_costs = np.vstack([all_costs, chosen_costs[None, :]])
            mask = outputs != self.STOP
            if ignore_first_eol: #and i == 0:
                mask[:] = 1
            all_masks = np.vstack([all_masks, mask[None, :]])

        all_outputs = all_outputs[1:] # skipping first row of self.BEGIN
        all_masks = all_masks[1:-1] #? all_masks[:-1] # skipping first row of self.BEGIN and the last row of self.STOP
        all_costs = all_costs[1:] - all_costs[:-1] #turn cumulative cost ito cost of each step #?actually the last row would suffice for us?
        result = all_outputs, all_masks, all_costs
        if not predict_pos:
             return self.result_to_lists(self.char_vocab,result)
        else:
            log_probs =  dy.log(self.predict_pos(position))
            pred_pos_id = np.argmax(log_probs.npvalue())
            pred_pos_tag = self.feat_vocab.i2w.get(pred_pos_id, UNK_CHAR)
            return pred_pos_tag, self.result_to_lists(self.char_vocab,result)
    
    @staticmethod
    def result_to_lists(vocab, result):
        outputs, masks, costs = [array.T for array in result]
        outputs = [list(output[:int(mask.sum())]) for output, mask in zip(outputs, masks)]
        words = [u''.join([vocab.i2w.get(pred_id,UNK_CHAR) for pred_id in output]) for output in outputs]
        costs = list(costs.T.sum(axis=0))
        results = zip(costs, words)
        results.sort(key=lambda h: h[0])
        return results

    def evaluate(self, data, beam, predict_pos=False):
        # data is a list of tuples (an instance of SoftDataSet with iter method applied)
        correct = 0.
        correct_tag = 0
        final_results = []
        data_len = 0
        for i,sent in enumerate(data):
            dy.renew_cg()
            
            inputs,outputs,features,features_in = sent
            self.param_init(inputs)
            if self.hyperparams['POS_FEATURE']:
                if self.hyperparams['AUX_POS_TASK']:
                    self.param_init_feat_testtime(inputs)
                else:
                    self.param_init_feat(features_in)
            
            for position,(input,output,feature) in enumerate(zip(inputs,outputs,features)):
                data_len += 1
                if not predict_pos:
                    predictions = self.predict(input, position, beam)
                else:
                    pos_tag,predictions = self.predict(input, position, beam, predict_pos = True)
                    if pos_tag == feature:
                        correct_tag +=1
                prediction = predictions[0][1]
            #            print i, input, predictions
                if prediction == output:
                    correct += 1
#                    print u'{}, input: {}, pred: {}, true: {}'.format(i, input, prediction, output)
#                    print predictions
#                else:
#                    print u'{}, input: {}, pred: {}, true: {}'.format(i, input, prediction, output)
#                    print predictions
                final_results.append((input,prediction))  # pred expected as list
        accuracy = correct / data_len
        if not predict_pos:
            return accuracy, final_results
        else:
            accuracy_tag = correct_tag / data_len
            return accuracy_tag, accuracy, final_results


def evaluate_ensemble(nmt_models, data, beam, predict_pos=False):
    # data is a list of tuples (an instance of SoftDataSet with iter method applied)
    correct = 0.
    correct_tag = 0
    final_results = []
    data_len = 0
    for i,sent in enumerate(data):
        dy.renew_cg()
        
        inputs,outputs,features,features_in = sent
        for m in nmt_models:
            m.param_init(inputs)
        if nmt_models[0].hyperparams['POS_FEATURE']:
            for m in nmt_models:
                if nmt_models[0].hyperparams['AUX_POS_TASK']:
                    m.param_init_feat_testtime(inputs)
                else:
                    m.param_init_feat(features_in)
    
        for position,(input,output,feature) in enumerate(zip(inputs,outputs,features)):
            data_len += 1
            if not predict_pos:
                predictions = predict_ensemble(nmt_models, input, position, beam)
            else:
                pos_tag,predictions = predict_ensemble(nmt_models, input, position, beam, predict_pos = True)
                if pos_tag == feature:
                    correct_tag +=1
            prediction = predictions[0][1]
        #            print i, input, predictions
            if prediction == output:
                    correct += 1
#        else:
#            print u'{}, input: {}, pred: {}, true: {}'.format(i, input, prediction, output)
#            print predictions
            final_results.append((input,prediction))  # pred expected as list
    accuracy = correct / data_len
    if not predict_pos:
        return accuracy, final_results
    else:
        accuracy_tag = correct_tag / data_len
        return accuracy_tag, accuracy, final_results


def predict_ensemble(nmt_models, input, position, beam_size, ignore_first_eol=False, predict_pos=False):
    """Performs beam search for ensemble of models.
    If the beam search was not compiled, it also compiles it.
    Parameters
    ----------
    max_length : int
    Maximum sequence length, the search stops when it is reached.
    ignore_first_eol : bool, optional
    When ``True``, the end if sequence symbol generated at the
    first iteration are ignored. This useful when the sequence
    generator was trained on data with identical symbols for
    sequence start and sequence end.
    as_arrays : bool, optional
    If ``True``, the internal representation of search results
    is returned, that is a (matrix of outputs, mask,
    costs of all generated outputs) tuple.
    Returns
    -------
    outputs : list of lists of ints
    A list of the `beam_size` best sequences found in the order
    of decreasing likelihood.
    costs : list of floats
    A list of the costs for the `outputs`, where cost is the
    negative log-likelihood.
    """
    nmt_vocab = nmt_models[0].char_vocab # same vocab file for all nmt_models!!
    BEGIN   = nmt_vocab.w2i[BEGIN_CHAR]
    STOP   = nmt_vocab.w2i[STOP_CHAR]
    
    for m in nmt_models:
        m.reset_decoder()
#        m.param_init(input, features)
    states = [[m.s] * beam_size for m in nmt_models] # ensemble x beam matrix of states
    # This array will store all generated outputs, including those from
    # previous step and those from already finished sequences.
    all_outputs = np.full(shape=(1,beam_size),fill_value=BEGIN,dtype = int)
    all_masks = np.ones_like(all_outputs, dtype=float) # whether predicted symbol is self.STOP
    all_costs = np.zeros_like(all_outputs, dtype=float) # the cumulative cost of predictions

    for i in range(MAX_PRED_SEQ_LEN):
        if all_masks[-1].sum() == 0:
            break

        # We carefully hack values of the `logprobs` array to ensure
        # that all finished sequences are continued with `eos_symbol`.
        logprobs_lst = []
        for j,m in enumerate(nmt_models):
            logprobs_m = np.array([-dy.log_softmax(m.predict_next_(s, position, scores=True)).npvalue() for s in states[j]]) # beam_size x vocab_len matrix
        #            print logprobs
        #            print all_masks[-1, :, None]
            next_costs = (all_costs[-1, :, None] + logprobs_m * all_masks[-1, :, None]) #take last row of cumul prev costs and turn into beam_size X 1 matrix, take logprobs distributions for unfinished hypos only and add it (elem-wise) with the array of prev costs; result: beam_size x vocab_len matrix of next costs
            (finished,) = np.where(all_masks[-1] == 0) # finished hypos have all their cost on the self.STOP symbol
            next_costs[finished, :STOP] = np.inf
            next_costs[finished, STOP + 1:] = np.inf
#            print next_costs
            (indexes, outputs), chosen_costs = SoftAttention._smallest(next_costs, beam_size, only_first_row=i == 0)
#            print j
#            print ','.join(nmt_vocab.i2w.get(pred_id,UNK_CHAR) for pred_id in outputs)
#            print chosen_costs
#            print indexes
#            print logprobs_m[indexes]
            logprobs_lst.append(logprobs_m)

#        logprobs_lst = np.array([[-dy.log_softmax(m.predict_next_(s, scores=True)).npvalue() for s in m_states] for m,m_states in zip(nmt_models,states)])
#        print logprobs_lst
#        print np.array(logprobs_lst).shape
        logprobs = np.sum(logprobs_lst, axis=0)
#        print logprobs.shape
#        print all_costs[-1, :, None]
#        print logprobs
#        print all_masks[-1, :, None]
        next_costs = (all_costs[-1, :, None] + logprobs * all_masks[-1, :, None]) #take last row of cumul prev costs and turn into beam_size X 1 matrix, take logprobs distributions for unfinished hypos only and add it (elem-wise) with the array of prev costs; result: beam_size x vocab_len matrix of next costs
        (finished,) = np.where(all_masks[-1] == 0) # finished hypos have all their cost on the self.STOP symbol
        next_costs[finished, :STOP] = np.inf
        next_costs[finished, STOP + 1:] = np.inf

        # indexes - the hypos from prev step to keep, outputs - the next step prediction, chosen cost - cost of predicted symbol
        (indexes, outputs), chosen_costs = SoftAttention._smallest(next_costs, beam_size, only_first_row=i == 0)
#        print 'ensemble:'
#        print ','.join(nmt_vocab.i2w.get(pred_id,UNK_CHAR) for pred_id in outputs)
#        print chosen_costs
#        print indexes
        # Rearrange everything
        new_states=[]
        for j,m in enumerate(nmt_models):
            new_states.append([states[j][ind] for ind in indexes])

#        new_states = ((states_m[ind] for ind in indexes) for states_m in states)
        all_outputs = all_outputs[:, indexes]
        all_masks = all_masks[:, indexes]
        all_costs = all_costs[:, indexes]

        # Record chosen output and compute new states
        states = [[m.consume_next_(s,pred_id) for s,pred_id in zip(m_new_states, outputs)] for m,m_new_states in zip(nmt_models, new_states)]
        all_outputs = np.vstack([all_outputs, outputs[None, :]])
        all_costs = np.vstack([all_costs, chosen_costs[None, :]])
        mask = outputs != STOP
#        if ignore_first_eol: # and i == 0:
#            mask[:] = 1
        all_masks = np.vstack([all_masks, mask[None, :]])

    all_outputs = all_outputs[1:] # skipping first row of self.BEGIN
    all_masks = all_masks[1:-1] #? all_masks[:-1] # skipping first row of self.BEGIN and the last row of self.STOP
    all_costs = all_costs[1:] - all_costs[:-1] #turn cumulative cost ito cost of each step #?actually the last row would suffice for us?
    result = all_outputs, all_masks, all_costs
    if not predict_pos:
        return result_to_lists(nmt_vocab, result)
    else:
        logprobs_pos = []
        for j,m in enumerate(nmt_models):
            logprobs_pos.append(np.array([-dy.log(m.predict_pos(position)).npvalue()]))
        logprobs = np.sum(logprobs_lst, axis=0)
        pred_pos_id = np.argmax(logprobs.npvalue())
        pred_pos_tag = self.feat_vocab.i2w.get(pred_pos_id, UNK_CHAR)
        return pred_pos_tag, self.result_to_lists(self.char_vocab,result)

def result_to_lists(nmt_vocab, result):
    outputs, masks, costs = [array.T for array in result]
    outputs = [list(output[:int(mask.sum())]) for output, mask in zip(outputs, masks)]
    words = [u''.join([nmt_vocab.i2w.get(pred_id,UNK_CHAR) for pred_id in output]) for output in outputs]
    costs = list(costs.T.sum(axis=0))
    results = zip(costs, words)
    results.sort(key=lambda h: h[0])
    return results

def load_ed_models(model_paths, pc):
    ed_models= []
    ed_model_params = []
    for i,path in enumerate(model_paths):
        print '...Loading nmt model {}'.format(i)
        ed_model_folder =  check_path(path, 'ED_MODEL_FOLDER_{}'.format(i), is_data_path=False)
        best_model_path  = ed_model_folder + '/bestmodel.txt'
        hypoparams_file_reader = codecs.open(ed_model_folder + '/best.dev', 'r', 'utf-8')
        hyperparams_dict = dict([line.strip().split(' = ') for line in hypoparams_file_reader.readlines()])
        model_hyperparams = {'INPUT_DIM_CHAR': int(hyperparams_dict['INPUT_DIM_CHAR']),
            'INPUT_DIM_FEAT': int(hyperparams_dict['INPUT_DIM_FEAT']),
            'INPUT_DIM_WORD': int(hyperparams_dict['INPUT_DIM_WORD']),
            'HIDDEN_DIM': int(hyperparams_dict['HIDDEN_DIM']),
            'HIDDEN_DIM_CONTEXT': int(hyperparams_dict['HIDDEN_DIM_CONTEXT']),
            'LAYERS': int(hyperparams_dict['LAYERS']),
            #                            'WORD_VOCAB_PATH': hyperparams_dict['WORD_VOCAB_PATH'],
            'CHAR_VOCAB_PATH': hyperparams_dict['CHAR_VOCAB_PATH'],
                'FEAT_VOCAB_PATH': hyperparams_dict['FEAT_VOCAB_PATH'],
                'FEAT_VOCAB_PATH_IN' : hyperparams_dict['FEAT_VOCAB_PATH_IN'],
                'AUX_POS_TASK': True if hyperparams_dict['AUX_POS_TASK']=="True" else False,
                'POS_FEATURE': True if hyperparams_dict['POS_FEATURE']=="True" else False,
                'POS_SPLIT_SPACE': True if hyperparams_dict['POS_SPLIT_SPACE']=="True" else False}
        # a fix for vocab path when transferring files b/n vm
        model_hyperparams['CHAR_VOCAB_PATH'] = check_path(path + '/char_vocab.txt', 'vocab_path', is_data_path=False)
        #            model_hyperparams['WORD_VOCAB_PATH'] = check_path(path + '/word_vocab.txt', 'vocab_path', is_data_path=False)
        model_hyperparams['FEAT_VOCAB_PATH'] = check_path(path + '/feat_vocab.txt', 'vocab_path', is_data_path=False)
        model_hyperparams['FEAT_VOCAB_PATH_IN'] = check_path(path + '/feat_vocab_in.txt', 'vocab_path', is_data_path=False)
        
        ed_model_params.append(pc.add_subcollection('ed{}'.format(i)))
        ed_model =  SoftAttention(ed_model_params[i], model_hyperparams,best_model_path)
    
        ed_models.append(ed_model)
    return ed_models

if __name__ == "__main__":
    arguments = docopt(__doc__)
    print arguments
    
    np.random.seed(123)
    random.seed(123)
    
    model_folder = check_path(arguments['MODEL_FOLDER'], 'MODEL_FOLDER', is_data_path=False)
    
    if arguments['train']:
        
        print '=========TRAINING:========='
        
        assert (arguments['--train_path']!=None) & (arguments['--dev_path']!=None)
        
        # load data
        print 'Loading data...'
        data_set = SoftDataSetCont
        train_path = check_path(arguments['--train_path'], 'train_path')
        input_format = [int(col) for col in arguments['--input_format'].split(',')]
        train_data = data_set.from_file(train_path,input_format, arguments['--lowercase'], arguments['--pos_split_space'], arguments['--conll_format'])
        print 'Train data has {} examples'.format(train_data.length)
        dev_path = check_path(arguments['--dev_path'], 'dev_path')
        dev_data = data_set.from_file(dev_path,input_format, arguments['--lowercase'], arguments['--pos_split_space'], arguments['--conll_format'])
        print 'Dev data has {} examples'.format(dev_data.length)
    
        print 'Checking if any special symbols in data...'
        for data, name in [(train_data, 'train'), (dev_data, 'dev')]:
            data = set(data.inputs + data.outputs)
            for c in [BEGIN_CHAR, STOP_CHAR, UNK_CHAR]:
                assert c not in data
            print '{} data does not contain special symbols'.format(name)
        print
        
        if os.path.exists(arguments['--char_vocab_path']):
            char_vocab_path = arguments['--char_vocab_path'] # absolute path  to existing vocab file
        else:
            tmp = os.path.join(RESULTS_FOLDER, arguments['--char_vocab_path'])
            if os.path.exists(tmp): # relative path to existing vocab file
                char_vocab_path = tmp
            else:
                char_vocab_path = os.path.join(model_folder,arguments['--char_vocab_path']) # no vocab - use default name
                print 'Building char vocabulary..'
                data = set(train_data.inputs + train_data.outputs)
                build_vocabulary(data, char_vocab_path)
                    
        if os.path.exists(arguments['--feat_vocab_path']):
                feat_vocab_path = arguments['--feat_vocab_path'] # absolute path  to existing vocab file
        else:
            tmp = os.path.join(RESULTS_FOLDER, arguments['--feat_vocab_path'])
            if os.path.exists(tmp): # relative path to existing vocab file
                feat_vocab_path = tmp
            else:
                feat_vocab_path = os.path.join(model_folder,arguments['--feat_vocab_path']) # no vocab - use default name
                print 'Building features vocabulary..'
                data = train_data.features
                build_vocabulary(data, feat_vocab_path, over_words=True)
    
        if os.path.exists(arguments['--feat_vocab_path_in']):
            feat_vocab_path_in = arguments['--feat_vocab_path_in'] # absolute path  to existing vocab file
        else:
            tmp = os.path.join(RESULTS_FOLDER, arguments['--feat_vocab_path_in'])
            if os.path.exists(tmp): # relative path to existing vocab file
                feat_vocab_path_in = tmp
            else:
                feat_vocab_path_in = os.path.join(model_folder,arguments['--feat_vocab_path_in']) # no vocab - use default name
                print 'Building features vocabulary..'
                # flatten the structure
                data = [feature for sublist in train_data.features_in for feature in sublist]
                build_vocabulary(data, feat_vocab_path_in, over_words=True)

#        if os.path.exists(arguments['--word_vocab_path']):
#                word_vocab_path = arguments['--word_vocab_path'] # absolute path  to existing vocab file
#        else:
#            tmp = os.path.join(RESULTS_FOLDER, arguments['--word_vocab_path'])
#            if os.path.exists(tmp): # relative path to existing vocab file
#                word_vocab_path = tmp
#            else:
#                word_vocab_path = os.path.join(model_folder,arguments['--word_vocab_path']) # no vocab - use default name
#                print 'Building word vocabulary..'
#                data = set(train_data.inputs)
#                build_vocabulary(data, word_vocab_path, vocab_trunk=0.01, over_words=True)


        # Paths for checks and results
        log_file_name   = model_folder + '/log.txt'
        best_model_path  = model_folder + '/bestmodel.txt'
        output_file_path = model_folder + '/best.dev'

        # Model hypoparameters
        model_hyperparams = {'INPUT_DIM_CHAR': int(arguments['--char_input']),
                            'INPUT_DIM_FEAT': int(arguments['--feat_input']),
                            'INPUT_DIM_WORD': int(arguments['--word_input']),
                            'HIDDEN_DIM': int(arguments['--hidden']),
                            'HIDDEN_DIM_CONTEXT': int(arguments['--hidden_context']),
                            'LAYERS': int(arguments['--layers']),
                            #'WORD_VOCAB_PATH': word_vocab_path,
                            'CHAR_VOCAB_PATH': char_vocab_path,
                            'FEAT_VOCAB_PATH': feat_vocab_path,
                            'FEAT_VOCAB_PATH_IN' : feat_vocab_path_in,
                            'AUX_POS_TASK': arguments['--aux_pos_task'],
                            'POS_FEATURE': arguments['--pos_feature'],
                            'POS_SPLIT_SPACE' : arguments['--pos_split_space']}

        print 'Building model...'
        pc = dy.ParameterCollection()
        ti = SoftAttention(pc, model_hyperparams)

        if arguments ['--aux_pos_task']:
            training_mode = ti.train_aux
        else:
            training_mode = ti.train_unsup

        # Training hypoparameters
        train_hyperparams = {'MAX_PRED_SEQ_LEN': MAX_PRED_SEQ_LEN,
                            'OPTIMIZATION': arguments['--optimization'],
                            'EPOCHS': int(arguments['--epochs']),
                            'PATIENCE': int(arguments['--patience']),
                            'DROPOUT': float(arguments['--dropout']),
                            'BEAM_WIDTH': 1,
                            'TRAIN_PATH': train_path,
                            'DEV_PATH': dev_path,
                            'AUX_WEIGHT': float(arguments['--aux_weight'])}

        print 'Train Hypoparameters:'
        for k, v in train_hyperparams.items():
            print '{:20} = {}'.format(k, v)
        print
        
        trainer = OPTIMIZERS[train_hyperparams['OPTIMIZATION']]
        trainer = trainer(pc)

        best_dev_accuracy = -1.
        sanity_set_size = 100 # for speed - check prediction accuracy on train set
        patience = 0

        # progress bar init
        widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
        train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=train_hyperparams['EPOCHS']).start()
        
        for epoch in xrange(train_hyperparams['EPOCHS']):
            print 'Start training...'
            then = time.time()

            # compute loss for each sample and update
            train_loss = 0.  # total train loss
            avg_train_loss = 0.  # avg training loss
            
            train_len = 0
            for i, sent in enumerate(train_data.iter(shuffle=True)):
                inputs, outputs, features, features_in = sent
                train_len += len(inputs)
                # new graph for each sentence
                dy.renew_cg()
                loss = training_mode(inputs, outputs, features, features_in, train_hyperparams['AUX_WEIGHT'])
#                if loss is not None:
                train_loss += loss.scalar_value()
                loss.backward()
                trainer.update()

            avg_train_loss = train_loss / train_len

            print '\t...finished in {:.3f} sec'.format(time.time() - then)

            # get train accuracy
            print 'evaluating on train...'
            dy.renew_cg() # new graph for all the examples
            then = time.time()
            if not arguments ['--aux_pos_task']:
                train_accuracy, _ = ti.evaluate(train_data.iter(indices=sanity_set_size), int(arguments['--beam']))
            else:
                train_acc_tag, train_accuracy, _ = ti.evaluate(train_data.iter(indices=sanity_set_size), int(arguments['--beam']), predict_pos=True )
            print '\t...finished in {:.3f} sec'.format(time.time() - then)
            
            # get dev accuracy
            print 'evaluating on dev...'
            then = time.time()
            dy.renew_cg() # new graph for all the examples
            if not arguments ['--aux_pos_task']:
                dev_accuracy, _ = ti.evaluate(dev_data.iter(), int(arguments['--beam']))
            else:
                dev_acc_tag, dev_accuracy, _ = ti.evaluate(dev_data.iter(), int(arguments['--beam']), predict_pos=True)
            print '\t...finished in {:.3f} sec'.format(time.time() - then)

            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                # save best model
                ti.save_model(best_model_path)
                print 'saved new best model to {}'.format(best_model_path)
                patience = 0
            else:
                patience += 1

            # found "perfect" model
            if dev_accuracy == 1:
                train_progress_bar.finish()
                break

            if not arguments ['--aux_pos_task']:
                print ('epoch: {0} train loss: {1:.4f} dev accuracy: {2:.4f} '
                   'train accuracy: {3:.4f} best dev accuracy: {4:.4f} patience = {5}').format(epoch, avg_train_loss, dev_accuracy, train_accuracy, best_dev_accuracy, patience)
                log_to_file(log_file_name, epoch, avg_train_loss, train_accuracy, dev_accuracy)
            else:
                print ('epoch: {0} train loss: {1:.4f} dev accuracy: {2:.4f} '
                       'train accuracy: {3:.4f} best dev accuracy: {4:.4f} train accuracy tag: {5:.4f} dev accuracy tag: {6:.4f} patience = {7}').format(epoch, avg_train_loss, dev_accuracy, train_accuracy, best_dev_accuracy, train_acc_tag, dev_acc_tag, patience)
                log_to_file(log_file_name, epoch, avg_train_loss, train_accuracy, dev_accuracy)
                log_to_file(log_file_name, epoch, avg_train_loss, train_acc_tag, dev_acc_tag)



            if patience == train_hyperparams['PATIENCE']:
                print 'out of patience after {} epochs'.format(epoch)
                train_progress_bar.finish()
                break
            # finished epoch
            train_progress_bar.update(epoch)
                
        print 'finished training.'
        
        ti = SoftAttention(pc, model_hyperparams, best_model_path)
        dev_accuracy, dev_results = ti.evaluate(dev_data.iter(), int(arguments['--beam']))
        print 'Best dev accuracy: {}'.format(dev_accuracy)
        write_param_file(output_file_path, dict(model_hyperparams.items()+train_hyperparams.items()))
        write_pred_file(output_file_path, dev_results)
        write_eval_file(output_file_path, best_dev_accuracy, dev_path)

    elif arguments['test']:
        print '=========EVALUATION ONLY:========='
        # requires test path, model path of pretrained path and results path where to write the results to

        best_model_path  = model_folder + '/bestmodel.txt'
        output_file_path = os.path.join(model_folder,arguments['--pred_path'])
        hypoparams_file = model_folder + '/best.dev'
        
        hypoparams_file_reader = codecs.open(hypoparams_file, 'r', 'utf-8')
        hyperparams_dict = dict([line.strip().split(' = ') for line in hypoparams_file_reader.readlines()])
        model_hyperparams = {'INPUT_DIM_CHAR': int(hyperparams_dict['INPUT_DIM_CHAR']),
                            'INPUT_DIM_FEAT': int(hyperparams_dict['INPUT_DIM_FEAT']),
                            'INPUT_DIM_WORD': int(hyperparams_dict['INPUT_DIM_WORD']),
                            'HIDDEN_DIM': int(hyperparams_dict['HIDDEN_DIM']),
                            'HIDDEN_DIM_CONTEXT': int(hyperparams_dict['HIDDEN_DIM_CONTEXT']),
                            'LAYERS': int(hyperparams_dict['LAYERS']),
                            #'WORD_VOCAB_PATH': hyperparams_dict['WORD_VOCAB_PATH'],
                            'CHAR_VOCAB_PATH': hyperparams_dict['CHAR_VOCAB_PATH'],
                            'FEAT_VOCAB_PATH': hyperparams_dict['FEAT_VOCAB_PATH'],
                            'FEAT_VOCAB_PATH_IN' : hyperparams_dict['FEAT_VOCAB_PATH_IN'],
                            'AUX_POS_TASK': True if hyperparams_dict['AUX_POS_TASK']=="True" else False,
                            'POS_FEATURE': True if hyperparams_dict['POS_FEATURE']=="True" else False,
                            'POS_SPLIT_SPACE': True if hyperparams_dict['POS_SPLIT_SPACE']=="True" else False}
        # a fix for vocab path when transferring files b/n vm
        model_hyperparams['CHAR_VOCAB_PATH'] = check_path(model_folder + '/char_vocab.txt', 'vocab_path', is_data_path=False)
#        model_hyperparams['WORD_VOCAB_PATH'] = check_path(model_folder + '/word_vocab.txt', 'vocab_path', is_data_path=False)
        model_hyperparams['FEAT_VOCAB_PATH'] = check_path(model_folder + '/feat_vocab.txt', 'vocab_path', is_data_path=False)
        model_hyperparams['FEAT_VOCAB_PATH_IN'] = check_path(model_folder + '/feat_vocab_in.txt', 'vocab_path', is_data_path=False)
        
        pc = dy.ParameterCollection()
        ti = SoftAttention(pc, model_hyperparams, best_model_path)

        assert arguments['--test_path']!=None
        print 'Loading data...'
        test_path = check_path(arguments['--test_path'], '--test_path')
        data_set = SoftDataSetCont
        input_format = [int(col) for col in arguments['--input_format'].split(',')]
        test_data = data_set.from_file(test_path,input_format, arguments['--lowercase'], ti.hyperparams['POS_SPLIT_SPACE'], arguments['--conll_format'])
        print 'Test data has {} examples'.format(test_data.length)
        
        print 'Checking if any special symbols in data...'
        data = set(test_data.inputs + test_data.outputs)
        for c in [BEGIN_CHAR, STOP_CHAR, UNK_CHAR]:
            assert c not in data
        print 'Test data does not contain special symbols'

        print 'Evaluating on test..'

        t = time.clock()
        accuracy, test_results = ti.evaluate(test_data.iter(), int(arguments['--beam']))
        print 'Time: {}'.format(time.clock()-t)
        print 'accuracy: {}'.format(accuracy)
        write_pred_file(output_file_path, test_results)
        write_eval_file(output_file_path, accuracy, test_path)

    elif arguments['ensemble_test']:
        print '=========EVALUATION ONLY:========='
        # requires test path, model path of pretrained path and results path where to write the results to

        pc = dy.ParameterCollection()
        
        ed_models = load_ed_models(arguments['ED_MODEL_FOLDER'].split(','), pc)
        ensemble_number = len(ed_models)
        output_file_path = os.path.join(model_folder,arguments['--pred_path'])

        assert arguments['--test_path']!=None
        print 'Loading data...'
        test_path = check_path(arguments['--test_path'], '--test_path')
        data_set = SoftDataSetCont
        input_format = [int(col) for col in arguments['--input_format'].split(',')]
        test_data = data_set.from_file(test_path,input_format, arguments['--lowercase'], ed_models[0].hyperparams['POS_SPLIT_SPACE'], arguments['--conll_format'])
        print 'Test data has {} examples'.format(test_data.length)
        
        print 'Checking if any special symbols in data...'
        data = set(test_data.inputs + test_data.outputs)
        for c in [BEGIN_CHAR, STOP_CHAR, UNK_CHAR]:
            assert c not in data
        print 'Test data does not contain special symbols'

        print 'Evaluating on test..'
        t = time.clock()
        accuracy, test_results = evaluate_ensemble(ed_models, test_data.iter(), int(arguments['--beam']))
        print 'Time: {}'.format(time.clock()-t)
        print 'accuracy: {}'.format(accuracy)
        write_pred_file(output_file_path, test_results)
        write_eval_file(output_file_path, accuracy, test_path)

