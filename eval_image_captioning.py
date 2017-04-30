'''
Scalable Bayesian Learning of Recurrent Neural Networks for Language Modeling 
https://arxiv.org/pdf/1611.08034.pdf
Developed by Zhe Gan, zg27@duke.edu, July, 12, 2016
'''

import time
import logging
import cPickle

import numpy as np
import scipy.io
import theano
import theano.tensor as tensor

from model.img_cap import init_params, init_tparams, build_model
from model.optimizers import pSGLD
from model.utils import get_minibatches_idx, zipp, unzip

def prepare_data(seqs):
    
    # x: a list of sentences
    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples)).astype('int64')
    x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask

def calu_negll(f_cost, prepare_data, data, img_feats, iterator):

    totalcost = 0.
    totallen = 0.
    for _, valid_index in iterator:
        x = [data[0][t]for t in valid_index]
        x, mask = prepare_data(x)
        z = np.array([img_feats[:,data[1][t]]for t in valid_index])
                
        cost = f_cost(x, mask,z) * x.shape[1]
        length = np.sum(mask)
        totalcost += cost
        totallen += length
    return totalcost/totallen

def calu_pred_prob(f_pred_prob, prepare_data, data, img_feats, iterator):

    pred_prob = np.array([])
    for _, valid_index in iterator:
        x, mask = prepare_data([data[0][t] for t in valid_index])
        z = np.array([img_feats[:,data[1][t]]for t in valid_index])
        pred_prob = np.concatenate((pred_prob, f_pred_prob(x, mask,z)))

    return pred_prob


""" Training the model. """

def train_model(train, valid, test, img_feats, W, n_words=7414, n_x=300, n_h=512,
    max_epochs=20, lrate=0.001, batch_size=64, valid_batch_size=64, dropout_val=0.5,
    dispFreq=10, validFreq=500, saveFreq=1000, saveto = 'flickr30k_result_psgld_dropout.npz'):
        
    """ n_words : vocabulary size
        n_x : word embedding dimension
        n_h : LSTM/GRU number of hidden units 
        max_epochs : The maximum number of epoch to run
        lrate : learning rate
        batch_size : batch size during training
        valid_batch_size : The batch size used for validation/test set
        dropout_val : the probability of dropout
        dispFreq : Display to stdout the training progress every N updates
        validFreq : Compute the validation error after this number of update.
        saveFreq : save results after this number of update.
        saveto : where to save.
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
    
    options['n_z'] = img_feats.shape[0]
   
    logger.info('Model options {}'.format(options))
    logger.info('{} train examples'.format(len(train[0])))
    logger.info('{} valid examples'.format(len(valid[0])))
    logger.info('{} test examples'.format(len(test[0])))

    logger.info('Building model...')
    
    params = init_params(options,W)
    tparams = init_tparams(params)

    (use_noise, x, mask, z, f_pred_prob, cost) = build_model(tparams,options)
    
    f_cost = theano.function([x, mask, z], cost, name='f_cost')
    
    lr_theano = tensor.scalar(name='lr')
    ntrain_theano = tensor.scalar(name='ntrain')
    f_grad_shared, f_update = pSGLD(tparams, cost, [x, mask,z], ntrain_theano, lr_theano)

    logger.info('Training model...')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)
    
    estop = False  # early stop
    history_negll = []
    best_p = None
    best_valid_negll, best_test_negll = 0., 0.
    bad_counter = 0    
    uidx = 0  # the number of update done
    start_time = time.time()
    
    # statistics of data
    train_num_words, valid_num_words, test_num_words = 0, 0, 0
    for sent in train[0]:
        train_num_words = train_num_words + len(sent)
    for sent in valid[0]:
        valid_num_words = valid_num_words + len(sent)
    for sent in test[0]:
        test_num_words = test_num_words + len(sent)
    
    n_average = 0
    valid_probs = np.zeros((valid_num_words,))
    test_probs = np.zeros((test_num_words,)) 
    
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(dropout_val)

                x = [train[0][t]for t in train_index]
                z = np.array([img_feats[:,train[1][t]]for t in train_index])
                
                x, mask = prepare_data(x)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask,z)
                f_update(lrate,len(train[0]))

                if np.isnan(cost) or np.isinf(cost):
                    logger.info('NaN detected')
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    logger.info('Epoch {} Update {} Cost {}'.format(eidx, uidx, cost))
                    
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
                    
                    if eidx < 3: 
                        valid_negll = calu_negll(f_cost, prepare_data, valid, img_feats, kf_valid)
                        test_negll = calu_negll(f_cost, prepare_data, test, img_feats, kf_test)
                        history_negll.append([valid_negll, test_negll])
                    else:
                        valid_probs_curr = calu_pred_prob(f_pred_prob, prepare_data, valid, img_feats, kf_valid)
                        test_probs_curr = calu_pred_prob(f_pred_prob, prepare_data, test, img_feats, kf_test)
                        valid_probs = (n_average * valid_probs + valid_probs_curr)/(n_average+1) 
                        test_probs = (n_average * test_probs + test_probs_curr)/(n_average+1) 
                        n_average += 1
                        
                        valid_negll = -np.log(valid_probs + 1e-6).sum() / valid_num_words
                        test_negll = -np.log(test_probs + 1e-6).sum() / test_num_words
                        history_negll.append([valid_negll, test_negll])
                        
                        logger.info('Saving {}th Sample...'.format(n_average))
                        
                        params = unzip(tparams)
                        np.savez('flickr30k_result_psgld_{}.npz'.format(n_average), valid_probs_curr=valid_probs_curr, test_probs_curr=test_probs_curr, **params)
                        logger.info('Done ...')
                        
                    
                    if (uidx == 0 or
                        valid_negll <= np.array(history_negll)[:,0].min()):
                             
                        best_p = unzip(tparams)
                        
                        best_valid_negll = valid_negll
                        best_test_negll = test_negll
                        
                        bad_counter = 0
                        
                    logger.info('Perp: Valid {} Test {}'.format(np.exp(valid_negll), np.exp(test_negll)))

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
    
    logger.info('Perp: Valid {} Test {}'.format(np.exp(best_valid_negll), np.exp(best_test_negll)))
    np.savez(saveto, history_negll=history_negll, **best_p)

    
    logger.info('The code run for {} epochs, with {} sec/epochs'.format(eidx + 1, 
                 (end_time - start_time) / (1. * (eidx + 1))))
    
    return best_valid_negll, best_test_negll

if __name__ == '__main__':
    
    # create logger with 'eval_bookcorpus'
    # https://docs.python.org/2/howto/logging-cookbook.html
    logger = logging.getLogger('eval_flickr30k_psgld_dropout')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('eval_flickr30k_psgld_dropout.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    
    x = cPickle.load(open("./data/flickr30k/data.p","rb"))
    train, val, test = x[0], x[1], x[2]
    wordtoix, ixtoword = x[3], x[4]
    del x
    n_words = len(ixtoword)
    
    x = cPickle.load(open("./data/flickr30k/word2vec.p","rb"))
    W = x[0]
    del x
    
    data = scipy.io.loadmat('./data/flickr30k/resnet_feats.mat')
    img_feats = data['feats'].astype(theano.config.floatX)

    [val_negll, te_negll] = train_model(train, val, test, img_feats, W,
        n_words=n_words)
        
