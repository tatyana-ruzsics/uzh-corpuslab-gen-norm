#!/usr/bin/env python
# -*- coding: utf-8 -*
"""Trains LSTM language model.

Usage:
  rnnlm.py train [--dynet-seed SEED] [--dynet-mem MEM] [--segformat] [--dictformat]
  [--input=INPUT] [--hidden=HIDDEN] [--feat-input=FEAT] [--layers=LAYERS] [--segments] [--vocab_path=VOCAB_PATH] [--vocab_trunk=VOCAB_TRUNK]
  [--dropout=DROPOUT] [--epochs=EPOCHS] [--patience=PATIENCE] [--optimization=OPTIMIZATION]
  MODEL_FOLDER --train_path=TRAIN_FILE --dev_path=DEV_FILE
  rnnlm.py test [--dynet-mem MEM] [--segformat] [--dictformat]
  MODEL_FOLDER --test_path=TEST_FILE [--segments]
  
Arguments:
  MODEL_FOLDER  save/read model folder where also eval results are written to, possibly relative to RESULTS_FOLDER

Options:
  -h --help                     show this help message and exit
  --dynet-seed SEED             DyNET seed
  --dynet-mem MEM               allocates MEM bytes for DyNET [default: 500]
  --input=INPUT                 input vector dimensions [default: 100]
  --hidden=HIDDEN               hidden layer dimensions [default: 200]
  --feat-input=FEAT             feature input vector dimension [default: 20]
  --layers=LAYERS               amount of layers in LSTMs  [default: 1]
  --dropout=DROPOUT             amount of dropout in LSTMs [default: 0]
  --patience=PATIENCE           patience for early stopping [default: 10]
  --epochs=EPOCHS               number of training epochs   [default: 10]
  --optimization=OPTIMIZATION   chosen optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA [default: SGD]
  --train_path=TRAIN_FILE       train set path, possibly relative to DATA_FOLDER, only for training
  --dev_path=DEV_FILE           dev set path, possibly relative to DATA_FOLDER, only for training
  --test_path=TEST_FILE         test set path, possibly relative to DATA_FOLDER, only for evaluation
  --segments                    run LM over segments instead of chars
  --vocab_path=VOCAB_PATH       vocab path, possibly relative to RESULTS_FOLDER [default: vocab.txt]
  --segformat                   format of the segmentation input file (3 cols)
  --dictformat                  format of the dictionary (1 col)
  --vocab_trunk=VOCAB_TRUNK     precentage of vocabulary to be replaced with unk [default: 0]
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
import copy


import dynet as dy
import numpy as np
import os
from itertools import izip

from common import BEGIN_CHAR,STOP_CHAR,UNK_CHAR, BOUNDARY_CHAR, SRC_FOLDER,RESULTS_FOLDER,DATA_FOLDER,check_path, write_pred_file, write_param_file, write_eval_file
from vocab_builder import build_vocabulary, Vocab


# Model defaults
MAX_PRED_SEQ_LEN = 50
OPTIMIZERS = {'ADAM'    : lambda m: dy.AdamTrainer(m, lam=0.0, alpha=0.0001, #common
                                                   beta_1=0.9, beta_2=0.999, eps=1e-8),
    'SGD'     : dy.SimpleSGDTrainer,
        'ADADELTA': dy.AdadeltaTrainer}


### IO handling and evaluation

def log_to_file(log_file_name, e, train_perplexity, dev_perplexity):
    # if first write, add headers
    if e == 0:
        log_to_file(log_file_name, 'epoch', 'train_perplexity', 'dev_perplexity')
    
    with open(log_file_name, "a") as logfile:
        logfile.write("{}\t{}\t{}\n".format(e, train_perplexity, dev_perplexity))

def read(filename, col_format=2, over_segs=False):
    """
        Read a file where each line is of the form "word1 word2 ..."
        Yields lists of the lines from file
        """
    with codecs.open(filename, encoding='utf8') as fh:
        if col_format==2:
            for line in fh:
                splt = line.strip().split('\t')
                assert len(splt) == 2, 'bad line: ' + line.encode('utf8') + '\n'
                input, output = splt
                # language model is trained on the target side of the corpus
                if over_segs:
                    # Segments
                    yield output.split(BOUNDARY_CHAR)
                else:
                    # Chars
                    yield [c for c in output]
        elif col_format==3:
            for line in fh:
                splt = line.strip().split('\t')
                assert len(splt) == 3, 'bad line: ' + line.encode('utf8') + '\n'
                #language model is trained on the target side of the corpus
                if over_segs:
                    # Segments
                    yield splt[2].split(BOUNDARY_CHAR)
                else:
                    # Chars
                    yield [c for c in splt[2]]
        else:
            for line in fh:
                l = line.strip()
                #language model is trained on the target side of the corpus
                if over_segs:
                    # Segments
                    yield l.split(BOUNDARY_CHAR)
                else:
                    # Chars
                    yield [c for c in l]


class RNNLanguageModel(object):
    def __init__(self, pc, model_hyperparams, best_model_path=None):
        
        self.hyperparams = model_hyperparams
        
        print 'Loading vocabulary from {}:'.format(self.hyperparams['VOCAB_PATH'])
        self.vocab = Vocab.from_file(self.hyperparams['VOCAB_PATH'])
        self.BEGIN   = self.vocab.w2i[BEGIN_CHAR]
        self.STOP   = self.vocab.w2i[STOP_CHAR]
        self.UNK       = self.vocab.w2i[UNK_CHAR]
        self.hyperparams['VOCAB_SIZE'] = self.vocab.size()
        
        self.build_model(pc, best_model_path)
        
            
        print 'Model Hypoparameters:'
        for k, v in self.hyperparams.items():
            print '{:20} = {}'.format(k, v)
        print
        

    def build_model(self, pc, best_model_path):
        
        if best_model_path:
            print 'Loading model from: {}'.format(best_model_path)
            self.RNN, self.VOCAB_LOOKUP, self.R, self.bias = dy.load(best_model_path, pc)
        else:
            # LSTM
            self.RNN  = dy.CoupledLSTMBuilder(self.hyperparams['LAYERS'], self.hyperparams['INPUT_DIM'], self.hyperparams['HIDDEN_DIM'], pc)
            
            # embedding lookups for vocabulary
            self.VOCAB_LOOKUP  = pc.add_lookup_parameters((self.hyperparams['VOCAB_SIZE'], self.hyperparams['INPUT_DIM']))

            # softmax parameters
            self.R = pc.add_parameters((self.hyperparams['VOCAB_SIZE'], self.hyperparams['HIDDEN_DIM']))
            self.bias = pc.add_parameters(self.hyperparams['VOCAB_SIZE'])
        
        
        print 'Model dimensions:'
        print ' * VOCABULARY EMBEDDING LAYER: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['VOCAB_SIZE'], self.hyperparams['INPUT_DIM'])
        print
        print ' * LSTM: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['INPUT_DIM'], self.hyperparams['HIDDEN_DIM'])
        print ' LSTM has {} layer(s)'.format(self.hyperparams['LAYERS'])
        print
        print ' * SOFTMAX: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['HIDDEN_DIM'], self.hyperparams['VOCAB_SIZE'])
        print
    
    def save_model(self, best_model_path):
        dy.save(best_model_path, [self.RNN, self.VOCAB_LOOKUP, self.R, self.bias])

    def BuildLMGraph(self, input):
#        dy.renew_cg()
        R = dy.parameter(self.R)   # from parameters to expressions
        bias = dy.parameter(self.bias)
        s = self.RNN.initial_state()
        
        input = [BEGIN_CHAR] + input + [STOP_CHAR]
        inputs_id = [self.vocab.w2i.get(c, self.UNK) for c in input]
        inputs_emb = [self.VOCAB_LOOKUP[c_id] for c_id in inputs_id]

        inputs = inputs_emb[:-1]
        true_preds = inputs_id[1:]
        
        states = s.transduce(inputs)
#        print [s_t.npvalue()[:3] for s_t in states]
        prob_ts = (bias + (R * s_t) for s_t in states)
        losses = [dy.pickneglogsoftmax(prob_t,true_pred) for prob_t, true_pred in izip(prob_ts, true_preds)]
#        print [loss.npvalue()[:3] for loss in losses]

#        losses = []
#        s = s.add_input(self.VOCAB_LOOKUP[inputs_id[0]])
#        for wid in inputs_id[1:]:
#            scores = bias + (R * s.output())
#            loss = dy.pickneglogsoftmax(scores, wid)
#            losses.append(loss)
#            s = s.add_input(self.VOCAB_LOOKUP[wid])

        return dy.esum(losses)

    def evaluate(self, data):
    # dev_data: a list of inputs where input is a list of units (chars/words)
        total_loss = 0.
        units_count = 0
        for input in data:
            loss = self.BuildLMGraph(input)
            total_loss += loss.scalar_value()
            units_count += len(input) + 1
        avg_loss = total_loss/len(data)
        perplexity = np.exp(total_loss/units_count)
        return avg_loss, perplexity

    def param_init(self):
        R = dy.parameter(self.R)   # from parameters to expressions
        bias = dy.parameter(self.bias)
        self.cg_params = (R, bias)
        
        self.s = self.RNN.initial_state()
        self.s = self.s.add_input(self.VOCAB_LOOKUP[self.BEGIN])

    def predict_next(self, scores=False, states = False):
        (R, bias) = self.cg_params
        if states:
            return self.s.output()
        else:
            next_scores = bias + (R * self.s.output())
            if not scores:
                return dy.softmax(next_scores)
            else:
                #                print 'next scores: {}'.format(next_scores.npvalue())
                return next_scores

    def predict_next_(self, state, scores=False, states = False):
        (R, bias) = self.cg_params
        if states:
            return state.output()
        else:
            next_scores = bias + (R * self.s.output())
            if not scores:
                return dy.softmax(next_scores)
            else:
                return next_scores

    def consume_next(self, next_id):
        self.s = self.s.add_input(self.VOCAB_LOOKUP[next_id])

    def consume_next_(self, state, next_id):
        new_state = state.add_input(self.VOCAB_LOOKUP[next_id])
        return new_state

    def train(self, input):
        self.param_init()
        input = [BEGIN_CHAR] + input + [STOP_CHAR]
        inputs_id = [self.vocab.w2i.get(c, self.UNK) for c in input]
        losses = []
        for next_id in inputs_id[1:]:
#            print self.vocab.i2w[next_id]
#            print self.s.output().npvalue()[:3]
#            probs = self.predict_next()
#            print probs.npvalue()[:3]
#            losses.append(-dy.log(dy.pick(probs, next_id)))
            scores = self.predict_next(scores=True)
            losses.append(dy.pickneglogsoftmax(scores, next_id))
#            print -dy.log(dy.pick(probs, next_id)).npvalue()
#            print dy.pickneglogsoftmax(scores, next_id).npvalue()
            self.consume_next(next_id)
#            print self.s.output().npvalue()[:3]
        return dy.esum(losses)

    def score3(self, segm_id, scores = False, eof = False):
        (R, bias) = self.cg_params
        if not eof:
            scores_seg = bias + (R * self.s.output())
            if not scores:
                probs = dy.softmax(scores_seg)
                return probs[segm_id]
            else:
                return scores_seg[segm_id]
        else:
            s_temp = self.s
            #            print 'segm id: {}'.format(segm_id)
            #s_temp.add_input(self.VOCAB_LOOKUP[segm_id])
            #scores_eof = bias + (R * s_temp.output())
            s = s_temp.transduce([self.VOCAB_LOOKUP[segm_id]])
            scores_eof = bias + (R * s[-1])
            if not scores:
                probs_eof = dy.softmax(scores_eof)
                return probs_eof[self.STOP]
            else:
                #                print 'eof scores: {}'.format(scores_eof.npvalue())
                return scores_eof[self.STOP]

    def score3_(self, state, segm_id, scores = False, eof = False):
        (R, bias) = self.cg_params
        scores_seg = bias + (R * state.output())
        if not eof:
            if not scores:
                probs = dy.softmax(scores_seg)
                return probs[segm_id]
            else:
                return scores_seg[segm_id]
        else:
            s_temp = state
            #            s_temp.add_input(self.VOCAB_LOOKUP[segm_id])
            #            scores_eof = bias + (R * s_temp.output())
            s = s_temp.transduce([self.VOCAB_LOOKUP[segm_id]])
            scores_eof = bias + (R * s[-1])
            if not scores:
                probs_eof = dy.softmax(scores_eof)
                return probs_eof[self.STOP]
            else:
                return scores_eof[self.STOP]


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
        over_segs = arguments['--segments']
        train_path = check_path(arguments['--train_path'], 'train_path')
        if arguments['--segformat']:
            col_format=3
        elif  arguments['--dictformat']:
            col_format=1
        else:
            col_format=2
        train_data = list(read(train_path, col_format, over_segs))
        print 'Train data has {} examples'.format(len(train_data))
        dev_path = check_path(arguments['--dev_path'], 'dev_path')
        dev_data = list(read(dev_path, col_format, over_segs))
        print 'Dev data has {} examples'.format(len(dev_data))
        
        print 'Checking if any special symbols in data...'
        for data, name in [(train_data, 'train'), (dev_data, 'dev')]:
            data = set([c for w in data for c in w])
            for c in [BEGIN_CHAR, STOP_CHAR, UNK_CHAR]:
                assert c not in data
            print '{} data does not contain special symbols'.format(name)
        print
        
        vocab_path = os.path.join(model_folder,arguments['--vocab_path'])
        if not os.path.exists(vocab_path):
            print 'Building vocabulary..'
            build_vocabulary(train_data, vocab_path, vocab_trunk=float(arguments['--vocab_trunk']))

        log_file_name   = model_folder + '/log.txt'
        best_model_path  = model_folder + '/bestmodel.txt'
        output_file_path = model_folder + '/best.dev'
        
        # Model hypoparameters
        model_hyperparams = {'INPUT_DIM': int(arguments['--input']),
                            'HIDDEN_DIM': int(arguments['--hidden']),
                            #'FEAT_INPUT_DIM': int(arguments['--feat-input']),
                            'LAYERS': int(arguments['--layers']),
                            'VOCAB_PATH': vocab_path,
                            'OVER_SEGS': over_segs}
    
        print 'Building model...'
        pc = dy.ParameterCollection()
        lm = RNNLanguageModel(pc, model_hyperparams)

        # Training hypoparameters
        train_hyperparams = {'MAX_PRED_SEQ_LEN': MAX_PRED_SEQ_LEN,
                            'OPTIMIZATION': arguments['--optimization'],
                            'EPOCHS': int(arguments['--epochs']),
                            'PATIENCE': int(arguments['--patience']),
                            'DROPOUT': float(arguments['--dropout']),
                            'BEAM_WIDTH': 1,
                            'TRAIN_PATH': train_path,
                            'DEV_PATH': dev_path}
        print 'Train Hypoparameters:'
        for k, v in train_hyperparams.items():
            print '{:20} = {}'.format(k, v)
        print

        trainer = OPTIMIZERS[train_hyperparams['OPTIMIZATION']]
        trainer = trainer(pc)

        best_dev_perplexity = 999.
#        patience = 0

        # progress bar init
        widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
        train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=train_hyperparams['EPOCHS']).start()

        for epoch in xrange(train_hyperparams['EPOCHS']):
            print 'Start training...'
            then = time.time()

            # compute loss for each sample and update
#            random.shuffle(train_data)
            train_loss = 0.
            loss_processed = 0 # for intermidiate reporting
            train_units_processed = 0 # for intermidiate reporting
            train_units = 0
            
            for i, input in enumerate(train_data, 1):
                # comp graph for each training example
                dy.renew_cg()
#                loss = lm.BuildLMGraph(input)
                loss = lm.train(input)

#                if loss.scalar_value()!=loss_cg.scalar_value():
#                    print 'epoch, i, loss, loss_cg: {}, {}, {}, {}'.format(epoch, i, loss.scalar_value(),loss_cg.scalar_value())
                train_loss += loss.scalar_value()
                loss_processed += loss.scalar_value()
                loss.backward()
                trainer.update()
                train_units_processed += len(input) + 1
                train_units += len(input) + 1
                
                # intermediate report on perplexity
                if i % 20000 == 0:
                    trainer.status()
                    print 'processed: {}, loss: {:.3f}, perplexity: {:.3f}'.format(i, loss_processed/train_units_processed, np.exp(loss_processed/train_units_processed))
                    train_units_processed = 0
                    loss_processed = 0

            avg_train_loss = train_loss/len(train_data)
            train_perplexity = np.exp(train_loss/train_units)

            print '\t...finished in {:.3f} sec'.format(time.time() - then)

            # get dev accuracy
            print 'evaluating on dev...'
            then = time.time()
            dy.renew_cg() # new graph for all the examples
            avg_dev_loss, dev_perplexity = lm.evaluate(dev_data)

            print '\t...finished in {:.3f} sec'.format(time.time() - then)

            if dev_perplexity < best_dev_perplexity:
                best_dev_perplexity = dev_perplexity
                # save best model
                lm.save_model(best_model_path)
                print 'saved new best model to {}'.format(best_model_path)
#                patience = 0
#            else:
#                patience += 1

            print ('epoch: {0} avg train loss: {1:.4f} avg dev loss: {2:.4f} train perplexity: {3:.4f} '
                   'dev perplexity: {4:.4f} best dev perplexity: {5:.4f} '
                   ).format(epoch, avg_train_loss, avg_dev_loss, train_perplexity, dev_perplexity, best_dev_perplexity)

            log_to_file(log_file_name, epoch, train_perplexity, dev_perplexity)

#            if patience == max_patience:
#                print 'out of patience after {} epochs'.format(epoch)
#                train_progress_bar.finish()
#                break
            # finished epoch
            train_progress_bar.update(epoch)
    
        print 'finished training.'
        
        # save best dev model parameters
        write_param_file(output_file_path, dict(model_hyperparams.items()+train_hyperparams.items()))
        write_eval_file(output_file_path, best_dev_perplexity, dev_path, 'Perplexity')

    elif arguments['test']:
        print '=========EVALUATION ONLY:========='
        # requires test path, model path of pretrained path and results path where to write the results to
        assert arguments['--test_path']!=None
        
        print 'Loading data...'
        over_segs = arguments['--segments']
        test_path = check_path(arguments['--test_path'], '--test_path')
        if arguments['--segformat']:
            col_format=3
        elif  arguments['--dictformat']:
            col_format=1
        else:
            col_format=2
        test_data = list(read(test_path, col_format, over_segs))
        print 'Test data has {} examples'.format(len(test_data))
        
        print 'Checking if any special symbols in data...'
        data = set([c for w in test_data for c in w])
        for c in [BEGIN_CHAR, STOP_CHAR, UNK_CHAR]:
            assert c not in data
        print 'Test data does not contain special symbols'

        best_model_path  = model_folder + '/bestmodel.txt'
        output_file_path = model_folder + '/best.test'
        hypoparams_file = model_folder + '/best.dev'

        hypoparams_file_reader = codecs.open(hypoparams_file, 'r', 'utf-8')
        hyperparams_dict = dict([line.strip().split(' = ') for line in hypoparams_file_reader.readlines()])
        model_hyperparams = {'INPUT_DIM': int(hyperparams_dict['INPUT_DIM']),
                            'HIDDEN_DIM': int(hyperparams_dict['HIDDEN_DIM']),
                            #'FEAT_INPUT_DIM': int(hyperparams_dict['FEAT_INPUT_DIM']),
                            'LAYERS': int(hyperparams_dict['LAYERS']),
                            'VOCAB_PATH': hyperparams_dict['VOCAB_PATH']}
        
        pc = dy.ParameterCollection()
        lm = RNNLanguageModel(pc, model_hyperparams, best_model_path)

        print 'Evaluating ont test..'
        _, perplexity = lm.evaluate(test_data)
        print 'Perplexity: {}'.format(perplexity)

        write_eval_file(output_file_path, perplexity, test_path, 'Perplexity')
