'''
Scalable Bayesian Learning of Recurrent Neural Networks for Language Modeling 
https://arxiv.org/pdf/1611.08034.pdf
Developed by Zhe Gan, zg27@duke.edu, July, 12, 2016
'''

import time
import logging
#import cPickle

import numpy as np
import theano
import theano.tensor as tensor
from collections import OrderedDict

from datasets import *
from model.language_model import init_params, init_tparams, build_model
from model.optimizers import SGLD
from model.utils import get_minibatches_idx

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def calu_negll(f_cost, data_x, data_y, iterator):

    total_negll = 0.
    for _, valid_index in iterator:
        x = data_x[valid_index].T
        y = data_y[valid_index].T
        negll = f_cost(x, y) * x.shape[0] * x.shape[1]
        total_negll += negll

    return total_negll / data_x.shape[0] / data_x.shape[1]
    
def calu_pred_prob(f_pred_prob, data_x, data_y, iterator):

    pred_prob = np.array([])
    for _, valid_index in iterator:
        x = data_x[valid_index].T
        y = data_y[valid_index].T
        pred_prob = np.concatenate((pred_prob, f_pred_prob(x, y)))

    return pred_prob

""" Training the model. """

def train_model(train_x, train_y, valid_x, valid_y, test_x, test_y, n_words=10000, 
    n_x=300, n_h=1500, max_epochs=55, collect_epoch = 4, lrate=1, anneal_lr_epoch = 15, 
    anneal_lr_factor = 1.15, dropout_val = 0.65, batch_size=32, valid_batch_size=64, dispFreq=10, 
    validFreq=400, saveFreq=1000, saveto = 'ptb_result_large_sgld_with_dropout.npz'):
        
    """ n_words : vocabulary size
        n_x : word embedding dimension
        n_h : LSTM/GRU number of hidden units 
        n_z : latent embedding sapce for a sentence 
        patience : Number of epoch to wait before early stop if no progress
        max_epochs : The maximum number of epoch to run
        lrate : learning rate
        optimizer : methods to do optimization
        batch_size : batch size during training
        valid_batch_size : The batch size used for validation/test set
        use_dropout : whether use dropout or not
        dispFreq : Display to stdout the training progress every N updates
        validFreq : Compute the validation error after this number of update.
        test_size : If >0, we keep only this number of test example.
    """

    
    options = {}
    options['n_words'] = n_words
    options['n_x'] = n_x
    options['n_h'] = n_h
    options['max_epochs'] = max_epochs
    options['lrate'] = lrate
    options['batch_size'] = batch_size
    options['valid_batch_size'] = valid_batch_size
    options['dispFreq'] = dispFreq
    options['validFreq'] = validFreq
    options['saveFreq'] = saveFreq
   
    logger.info('Model options {}'.format(options))
    logger.info('Building model...')
    
    params = init_params(options)
    tparams = init_tparams(params)

    use_noise, x, y, f_pred_prob, cost = build_model(tparams,options)
    
    f_cost = theano.function([x, y], cost, name='f_cost')
    
    lr_theano = tensor.scalar(name='lr')
    ntrain_theano = tensor.scalar(name='ntrain')
    f_grad_shared, f_update = SGLD(tparams, cost, [x, y], ntrain_theano, lr_theano)

    logger.info('Training model...')

    kf_valid = get_minibatches_idx(valid_x.shape[0], valid_batch_size)
    kf_test = get_minibatches_idx(test_x.shape[0], valid_batch_size)
    
    estop = False  # early stop
    history_negll = []
    best_p = None
    best_valid_negll, best_test_negll = 0., 0.
    bad_counter = 0    
    uidx = 0  # the number of update done
    start_time = time.time()
    
    # statistics of data
    train_num_words = train_x.shape[0] * train_x.shape[1]
    valid_num_words = valid_x.shape[0] * valid_x.shape[1]  
    test_num_words = test_x.shape[0] * test_x.shape[1]  
    
    n_average = 0
    valid_probs = np.zeros((valid_num_words,))
    test_probs = np.zeros((test_num_words,)) 
        
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0
            kf = get_minibatches_idx(train_x.shape[0], batch_size, shuffle=True)
            
            if eidx >= anneal_lr_epoch:
                #annealing learning rate
                lrate = lrate/anneal_lr_factor

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(dropout_val)

                x = train_x[train_index].T
                y = train_y[train_index].T
                
                n_samples += x.shape[1]

                cost = f_grad_shared(x, y)
                f_update(lrate,train_num_words)

                if np.isnan(cost) or np.isinf(cost):
                    
                    logger.info('NaN detected')
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    
                    logger.info('Epoch {} Update {} Cost {}'.format(eidx, uidx, np.exp(cost)))
                    
                if np.mod(uidx, saveFreq) == 0:
                    
                    logger.info('Saving ...')
                
                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    np.savez(saveto, history_negll=history_negll, **params)
                    
                    logger.info('Done ...')

                if np.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    
                    if eidx < collect_epoch: 
                        valid_negll = calu_negll(f_cost, valid_x, valid_y, kf_valid)
                        test_negll = calu_negll(f_cost, test_x, test_y, kf_test)
                        history_negll.append([valid_negll, test_negll])
                    else:
                        valid_probs_curr = calu_pred_prob(f_pred_prob, valid_x, valid_y, kf_valid)
                        test_probs_curr = calu_pred_prob(f_pred_prob, test_x, test_y, kf_test)
                                           
                        valid_probs = (n_average * valid_probs + valid_probs_curr)/(n_average+1) 
                        test_probs = (n_average * test_probs + test_probs_curr)/(n_average+1) 
                        n_average += 1
                        
                        valid_negll = -np.log(valid_probs + 1e-6).sum() / valid_num_words
                        test_negll = -np.log(test_probs + 1e-6).sum() / test_num_words
                        history_negll.append([valid_negll, test_negll])
                        
                        logger.info('Saving {}th Sample...'.format(n_average))
                        
                        params = unzip(tparams)
                        np.savez('ptb_result_sgld_large_{}.npz'.format(n_average), valid_probs_curr=valid_probs_curr, test_probs_curr=test_probs_curr, **params)
                        logger.info('Done ...')
                    
                    if (uidx == 0 or
                        valid_negll <= np.array(history_negll)[:,0].min()):
                             
                        best_p = unzip(tparams)
                        
                        best_valid_negll = valid_negll
                        best_test_negll = test_negll
                        
                        bad_counter = 0
                        
                    logger.info('Valid {} Test {}'.format(np.exp(valid_negll),
                                 np.exp(test_negll)))

                    if (len(history_negll) > 10 and
                        valid_negll >= np.array(history_negll)[:-10,0].min()):
                            bad_counter += 1
                            if bad_counter > 10:
                                logger.info('Early Stop!')
                                estop = True
                                break

            logger.info('Seen {} samples'.format(n_samples))
            
            if estop:
                break

    except KeyboardInterrupt:
        
        logger.info('Training interupted')

    end_time = time.time()
    
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)
    
#    use_noise.set_value(0.)
#    kf_train_sorted = get_minibatches_idx(len(train), batch_size)
#    train_negll = calu_negll(f_cost, prepare_data, train, kf_train_sorted)
#    valid_negll = calu_negll(f_cost, prepare_data, valid, kf_valid)
#    test_negll = calu_negll(f_cost, prepare_data, test, kf_test)
    
    
    logger.info('Valid {} Test {}'.format(np.exp(best_valid_negll), np.exp(best_test_negll)))
    np.savez(saveto, history_negll=history_negll, **best_p)

    logger.info('The code run for {} epochs, with {} sec/epochs'.format(eidx + 1, 
                 (end_time - start_time) / (1. * (eidx + 1))))
    
    return best_valid_negll, best_test_negll

if __name__ == '__main__':
    
    # create logger with 'eval_ptb'
    # https://docs.python.org/2/howto/logging-cookbook.html
    logger = logging.getLogger('eval_ptb_large_sgld_with_dropout')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('eval_ptb_large_sgld_with_dropout.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    
    ptb_train_path = 'data/ptb/ptb.train.txt.gz'
    ptb_valid_path = 'data/ptb/ptb.valid.txt.gz'
    ptb_test_path = 'data/ptb/ptb.test.txt.gz'

    # data params
    time_steps = 20
    batch_size = 32
    
    # data
    train_set = Text(ptb_train_path, batch_size, time_steps)
    valid_set = Text(ptb_valid_path, batch_size, time_steps,
                 vocab_map=train_set.vocab_map, vocab_index=train_set.vocab_idx)
    test_set = Text(ptb_test_path, batch_size, time_steps,
                 vocab_map=train_set.vocab_map, vocab_index=train_set.vocab_idx)
    vocab_size = len(train_set.vocab_map)
    
    train_x, train_y = train_set.X, train_set.y
    valid_x, valid_y = valid_set.X, valid_set.y
    test_x, test_y = test_set.X, test_set.y
    
    [va_negll, te_negll] = train_model(train_x, train_y, valid_x, valid_y,
        test_x, test_y, n_words=vocab_size)
