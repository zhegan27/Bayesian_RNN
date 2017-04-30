
import numpy as np
import theano
import theano.tensor as tensor

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import dropout, numpy_floatX
from utils import uniform_weight, zero_bias

from lstm_layers_lm import param_init_decoder, decoder_layer

# Set the random number generators' seeds for consistency
SEED = 123  
np.random.seed(SEED)

""" init. parameters. """  
def init_params(options):
    
    n_words = options['n_words']
    n_x = options['n_x']  
    n_h = options['n_h']
    
    params = OrderedDict()
    params['Wemb'] = uniform_weight(n_words,n_x)
    params = param_init_decoder(options,params,prefix='decoder_h1')
    
    options['n_x'] = n_h
    params = param_init_decoder(options,params,prefix='decoder_h2')
    options['n_x'] = n_x
    
    params['Vhid'] = uniform_weight(n_h,n_words)
    params['bhid'] = zero_bias(n_words)                                    

    return params

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams
    
""" Building model... """

def build_model(tparams,options):
    
    trng = RandomStreams(SEED)
    
    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))
    
    # x: n_steps * n_samples
    x = tensor.matrix('x', dtype='int64')
    y = tensor.matrix('y', dtype='int64')
    
    n_steps = x.shape[0]
    n_samples = x.shape[1]
    
    n_x = tparams['Wemb'].shape[1]
    
    emb = tparams['Wemb'][x.flatten()].reshape([n_steps,n_samples,n_x])
    emb = dropout(emb, trng, use_noise)
    
    h_decoder = decoder_layer(tparams, emb, prefix='decoder_h1')
    h_decoder = dropout(h_decoder, trng, use_noise)
    
    h_decoder = decoder_layer(tparams, h_decoder, prefix='decoder_h2')
    h_decoder = dropout(h_decoder, trng, use_noise)
    
    # n_steps * n_samples * n_h                                    
    shape = h_decoder.shape
    h_decoder = h_decoder.reshape((shape[0]*shape[1], shape[2]))
    
    pred = tensor.dot(h_decoder, tparams['Vhid']) + tparams['bhid']
    pred = tensor.nnet.softmax(pred)
    
    y_vec = y.reshape((shape[0]*shape[1],))
    index = tensor.arange(shape[0]*shape[1])
    y_pred = pred[index, y_vec]
    
    f_pred_prob = theano.function([x, y], y_pred, name='f_pred_prob')
    cost = -tensor.log(y_pred + 1e-6).sum() / n_steps / n_samples                     

    return use_noise, x, y, f_pred_prob, cost
