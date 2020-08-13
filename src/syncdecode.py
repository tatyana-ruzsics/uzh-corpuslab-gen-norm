#!/usr/bin/env python
# -*- coding: utf-8 -*
"""Synchronized decoding for combining soft attention models trained over chars with language model over segments(i.e. words/morphemes).

Usage:
  syncdecode.py [--dynet-mem MEM] [--beam=BEAM] [--pred_path=PRED_FILE]
  ED_MODEL_FOLDER LM_MODEL_FOLDER MODEL_FOLDER LM_WEIGHTS --test_path=TEST_FILE [--segformat]
  
Arguments:
  ED_MODEL_FOLDER  ED model(s) folder, possibly relative to RESULTS_FOLDER, coma-separated
  LD_MODEL_FOLDER  LM model(s) folder, possibly relative to RESULTS_FOLDER, coma-separated
  MODEL_FOLDER     results folder, possibly relative to RESULTS_FOLDER
  LM_WEIGHTS       weights for LM predictors, coma-separated

Options:
  -h --help                     show this help message and exit
  --dynet-mem MEM               allocates MEM bytes for DyNET [default: 500]
  --dev_path=DEV_FILE           dev set path, possibly relative to DATA_FOLDER, only for training
  --beam=BEAM                   beam width [default: 1]
  --test_path=TEST_FILE         test set path, possibly relative to DATA_FOLDER, only for evaluation
  --pred_path=PRED_FILE         name for predictions file in the test mode [default: 'best.test']
  --segformat                   format of the segmentation input file (3 cols)
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

import math
import re

import dynet as dy
import numpy as np
import os
from itertools import izip
import copy

from common import BEGIN_CHAR,STOP_CHAR,UNK_CHAR, BOUNDARY_CHAR, SRC_FOLDER,RESULTS_FOLDER,DATA_FOLDER,check_path, write_pred_file, write_param_file, write_eval_file
from vocab_builder import build_vocabulary, Vocab
from norm_soft import log_to_file, SoftDataSet, SoftAttention
from rnnlm import RNNLanguageModel

MAX_PRED_SEQ_LEN = 50 # option

def _compute_scores(lm_models, s_lm, w_lm, segment, UNK, eow=False):
    """compute scores of model ensemble """
    lm_total_score=0
    if eow:
        for i,(m,s,w) in enumerate(zip(lm_models,s_lm,w_lm)):
            scores = m.predict_next_(s, scores=False)
            if m.hyperparams['OVER_SEGS'] == True:
                segment_id = m.vocab.w2i.get(segment, UNK)
                lm_eow_score = m.score3_(s, segment_id, scores=False, eof = True)
            else:
                lm_eow_score = dy.pick(scores,m.vocab.w2i[STOP_CHAR])
            lm_total_score += -np.log(lm_eow_score.value()) * w
            
    else:
        for i,(m,s,w) in enumerate(zip(lm_models,s_lm,w_lm)):
            scores = m.predict_next_(s, scores=False)
            if m.hyperparams['OVER_SEGS'] == True:
                segment_id = m.vocab.w2i.get(segment, UNK)
                lm_boundary_score = dy.pick(scores,segment_id)
            else:
                lm_boundary_score = dy.pick(scores,m.vocab.w2i[BOUNDARY_CHAR])
            lm_total_score += -np.log(lm_boundary_score.value()) * w
    return lm_total_score
            
    
def predict_syncbeam(input, nmt_models, lm_models, lm_weights, beam = 1):
    """predicts a string of characters performing synchronous beam-search."""
    dy.renew_cg()
    for nmt_model in nmt_models:
        nmt_model.param_init(input)
    for lm_model in lm_models:
        lm_model.param_init()
    nmt_vocab = nmt_models[0].vocab # same vocab file for all nmt_models
    BEGIN   = nmt_vocab.w2i[BEGIN_CHAR]
    STOP   = nmt_vocab.w2i[STOP_CHAR]
    UNK       = nmt_vocab.w2i[UNK_CHAR]
    BOUNDARY = nmt_vocab.w2i[BOUNDARY_CHAR]

    m_hypos = [([m.s for m in nmt_models],[m.s for m in lm_models] ,0., '', '')] # hypos to be expanded by morphemes
    m_complete_hypos = [] # hypos wihich end with STOP
    m_pred_length = 0 # number of morphemes
#        max_score = np.inf

    while m_pred_length <= MAX_PRED_SEQ_LEN and len(m_complete_hypos) < beam:# continue expansion while we don't have beam closed hypos #todo: MAX_PRED_SEQ_LEN should be changed to max morphemes number per word
        m_expansion = [] # beam * m_hypos expansion to be collected on the current iteration
        
        for m_hypo in m_hypos:
            hypos = [m_hypo] # hypos to be expanded by chars
            complete_hypos = [] # hypos wihich end with STOP or BOUNDARY
            pred_length = 0 # number of chars per morpheme
        
            while pred_length <= MAX_PRED_SEQ_LEN and len(hypos) > 0: # continue expansion while there is a hypo to expand
                expansion = [] # beam * m_hypos expansion to be collected on the current iteration
                for s_nmt, s_lm, log_p, word, segment in hypos:
                    log_probs = np.array([-dy.log_softmax(m.predict_next_(s, scores=True)).npvalue() for m,s in zip(nmt_models,s_nmt)])
#                    print log_probs
                    log_probs = np.sum(log_probs, axis=0)
#                    print log_probs
                    top = np.argsort(log_probs,axis=0)[:beam]
                    expansion.extend(( (s_nmt, s_lm, log_p + log_probs[pred_id], copy.copy(word), copy.copy(segment), pred_id) for pred_id in top ))
                hypos = []
                expansion.extend(complete_hypos)
                complete_hypos = []
                expansion.sort(key=lambda e: e[2])
#                print u'expansion: {}'.format([(w+nmt_vocab.i2w.get(pred_id,UNK_CHAR),log_p) for _,_,log_p,w,_,pred_id in expansion[:beam]])
                for e in expansion[:beam]:
                    s_nmt, s_lm, log_p, word, segment, pred_id = e
                    if pred_id == STOP:
                        lm_score= _compute_scores(lm_models, s_lm, lm_weights, segment, UNK, eow=True)
#                            if log_p < max_score:
#                                max_score = max(max_score,log_p)
                        complete_hypos.append((s_nmt, s_lm, log_p+lm_score, word, segment, pred_id))
                    elif pred_id == BOUNDARY:
                        lm_score= _compute_scores(lm_models, s_lm, lm_weights, segment, UNK, eow=False)
                        complete_hypos.append((s_nmt, s_lm, log_p+lm_score, word, segment, pred_id))
                    else:
                        pred_char = nmt_vocab.i2w.get(pred_id,UNK_CHAR)
                        word+=pred_char
                        segment+=pred_char
                        new_lm_states = []
                        for m,s in zip(lm_models,s_lm):
                            if m.hyperparams['OVER_SEGS'] != True:
                                new_lm_states.append(m.consume_next_(s,m.vocab.w2i.get(pred_char, UNK)))
                            else:
                                new_lm_states.append(s)
                        hypos.append(([m.consume_next_(s,pred_id) for m,s in zip(nmt_models,s_nmt)],new_lm_states, log_p, word, segment))
                pred_length += 1
            m_expansion.extend(complete_hypos)

        m_hypos = []
        m_expansion.extend(m_complete_hypos)
        m_complete_hypos = []
        m_expansion.sort(key=lambda e: e[2])
#        print u'm_expansion: {}'.format([(w+nmt_vocab.i2w.get(pred_id,UNK_CHAR),log_p) for _,_,log_p,w,_,pred_id in m_expansion[:beam]])
        for e in m_expansion[:beam]:
            s_nmt, s_lm, log_p, word, segment, pred_id = e
            if pred_id == STOP:
                m_complete_hypos.append(e)
            else: #BOUNDARY
                pred_char = nmt_vocab.i2w.get(pred_id,UNK_CHAR)
                word+=pred_char
                new_lm_states = []
                for m,s in zip(lm_models,s_lm):
                    if m.hyperparams['OVER_SEGS'] != True:
                        new_lm_states.append(m.consume_next_(s,m.vocab.w2i.get(nmt_vocab.i2w[pred_id], UNK)))
                    else:
                        segment_id = m.vocab.w2i.get(segment, UNK)
                        scores = m.predict_next_(s, scores=True)
                        new_lm_states.append(m.consume_next_(s,m.vocab.w2i.get(segment, UNK)))
                m_hypos.append(([m.consume_next_(s,pred_id) for m,s in zip(nmt_models,s_nmt)],new_lm_states, log_p, word,''))
        m_pred_length += 1
            
    if not m_complete_hypos:
        # nothing found
        m_complete_hypos = [(log_p, word) for s_nmt, s_lm, log_p, word, _ in m_hypos]
            
    m_complete_hypos.sort(key=lambda e: e[2])
    final_hypos = []
    for _,_, log_p, word,_,_ in m_complete_hypos[:beam]:
        final_hypos.append((log_p, word))
    return final_hypos

def evaluate_syncbeam(data, ed_models, lm_models, lm_weights, beam):
        # data is a list of tuples (an instance of SoftDataSet with iter method applied)
    correct = 0.
    final_results = []
    for i,(input,output) in enumerate(data):
        predictions = predict_syncbeam(input, ed_models, lm_models, lm_weights, beam)
        prediction = predictions[0][1]
        if prediction == output.lower():
            correct += 1
        else:
            print u'{}, input: {}, pred: {}, true: {}'.format(i, input, prediction, output)
            print predictions
        final_results.append((input,prediction))  # pred expected as list
    accuracy = correct / len(data)
    return accuracy, final_results

if __name__ == "__main__":
    arguments = docopt(__doc__)
    print arguments
    
    np.random.seed(123)
    random.seed(123)

    model_folder = check_path(arguments['MODEL_FOLDER'], 'MODEL_FOLDER', is_data_path=False)

    print '=========EVALUATION ONLY:========='
    # requires test path, model path of pretrained path and results path where to write the results to
    assert arguments['--test_path']!=None
    
    print 'Loading data...'
    test_path = check_path(arguments['--test_path'], '--test_path')
    data_set = SoftDataSet
    three_col_format = True if arguments['--segformat'] else False
    test_data = data_set.from_file(test_path,three_col_format)
    print 'Test data has {} examples'.format(test_data.length)
    
    print 'Checking if any special symbols in data...'
    data = set(test_data.inputs + test_data.outputs)
    for c in [BEGIN_CHAR, STOP_CHAR, UNK_CHAR]:
        assert c not in data
    print 'Test data does not contain special symbols'
    
    pc = dy.ParameterCollection()
    
    lm_weights = [float(w) for w in arguments['LM_WEIGHTS'].split(',')]

    ed_models= []
    ed_model_params = []
    ## loading the nmt models
    for i,path in enumerate(arguments['ED_MODEL_FOLDER'].split(',')):
        print '...Loading nmt model {}'.format(i)
        ed_model_folder =  check_path(path, 'ED_MODEL_FOLDER_{}'.format(i), is_data_path=False)
        best_model_path  = ed_model_folder + '/bestmodel.txt'
        hypoparams_file_reader = codecs.open(ed_model_folder + '/best.dev', 'r', 'utf-8')
        hyperparams_dict = dict([line.strip().split(' = ') for line in hypoparams_file_reader.readlines()])
        model_hyperparams = {'INPUT_DIM': int(hyperparams_dict['INPUT_DIM']),
            'HIDDEN_DIM': int(hyperparams_dict['HIDDEN_DIM']),
                #'FEAT_INPUT_DIM': int(hyperparams_dict['FEAT_INPUT_DIM']),
                'LAYERS': int(hyperparams_dict['LAYERS']),
                    'VOCAB_PATH': hyperparams_dict['VOCAB_PATH']}
        # vocab folder is taken from the first nmt folder
        vocab_path = check_path(path, 'ED_MODEL_FOLDER_{}'.format(0), is_data_path=False) + '/vocab.txt'
        model_hyperparams['VOCAB_PATH'] = vocab_path
        ed_model_params.append(pc.add_subcollection('ed{}'.format(i)))
        ed_model =  SoftAttention(ed_model_params[i], model_hyperparams,best_model_path)
        
        ed_models.append(ed_model)
    ensemble_number = len(ed_models)

    lm_models= []
    lm_model_params = []
    ## loading the language models
    for i,path in enumerate(arguments['LM_MODEL_FOLDER'].split(',')):
        print '...Loading lm model {}'.format(i)
        lm_model_folder =  check_path(path, 'LM_MODEL_FOLDER_{}'.format(i), is_data_path=False)
        best_model_path  = lm_model_folder + '/bestmodel.txt'
        hypoparams_file_reader = codecs.open(lm_model_folder + '/best.dev', 'r', 'utf-8')
        hyperparams_dict = dict([line.strip().split(' = ') for line in hypoparams_file_reader.readlines()])
        model_hyperparams = {'INPUT_DIM': int(hyperparams_dict['INPUT_DIM']),
            'HIDDEN_DIM': int(hyperparams_dict['HIDDEN_DIM']),
                #'FEAT_INPUT_DIM': int(hyperparams_dict['FEAT_INPUT_DIM']),
                'LAYERS': int(hyperparams_dict['LAYERS']),
                    'VOCAB_PATH': hyperparams_dict['VOCAB_PATH'],
                        'OVER_SEGS':  'OVER_SEGS' in hyperparams_dict}
        # vocab folder is taken from the first nmt folder
        vocab_path = check_path(path, 'ED_MODEL_FOLDER_{}'.format(0), is_data_path=False) + '/vocab.txt'
        model_hyperparams['VOCAB_PATH'] = vocab_path
        lm_model_params.append(pc.add_subcollection('lm{}'.format(i)))
        lm_model =  RNNLanguageModel(lm_model_params[i], model_hyperparams,best_model_path)
        
        lm_models.append(lm_model)
    lm_number  = len(lm_models)

    output_file_path = os.path.join(model_folder,arguments['--pred_path'])

    # save best dev model parameters and predictions
    print 'Evaluating on test..'
    t = time.clock()
    accuracy, test_results = evaluate_syncbeam(test_data.iter(), ed_models, lm_models, lm_weights, int(arguments['--beam']))
    print 'Time: {}'.format(time.clock()-t)
    print 'accuracy: {}'.format(accuracy)
    write_pred_file(output_file_path, test_results)
    write_eval_file(output_file_path, accuracy, test_path)
