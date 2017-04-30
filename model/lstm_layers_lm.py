
import numpy as np
import theano
import theano.tensor as tensor
from utils import _p, numpy_floatX
from utils import ortho_weight, uniform_weight, zero_bias

"""Using LSTM Recurrent Neural Network. """

def param_init_decoder(options, params, prefix='decoder_lstm'):
    
    n_x = options['n_x']
    n_h = options['n_h']
    
    W = np.concatenate([uniform_weight(n_x,n_h),
                        uniform_weight(n_x,n_h),
                        uniform_weight(n_x,n_h),
                        uniform_weight(n_x,n_h)], axis=1)
    params[_p(prefix, 'W')] = W
    
    U = np.concatenate([ortho_weight(n_h),
                        ortho_weight(n_h),
                        ortho_weight(n_h),
                        ortho_weight(n_h)], axis=1)
    params[_p(prefix, 'U')] = U
    
    params[_p(prefix,'b')] = zero_bias(4*n_h)
    params[_p(prefix, 'b')][n_h:2*n_h] = 3*np.ones((n_h,)).astype(theano.config.floatX)

    return params   
    

def decoder_layer(tparams, state_below, prefix='decoder_lstm'):
    
    """ state_below: size of n_steps * n_samples * n_x 
    """

    nsteps = state_below.shape[0]
    n_h = tparams[_p(prefix,'U')].shape[0]
    
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]
        
    def _step(x_, h_, c_, U):
        preact = tensor.dot(h_, U)
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, n_h))
        f = tensor.nnet.sigmoid(_slice(preact, 1, n_h))
        o = tensor.nnet.sigmoid(_slice(preact, 2, n_h))
        c = tensor.tanh(_slice(preact, 3, n_h))

        c = f * c_ + i * c

        h = o * tensor.tanh(c)

        return h, c
        
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
                          
    seqs = [state_below_]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info = [tensor.alloc(numpy_floatX(0.),
                                                           n_samples,n_h),
                                                tensor.alloc(numpy_floatX(0.),
                                                           n_samples,n_h)],
                                non_sequences = [tparams[_p(prefix, 'U')]],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                strict=True)
                                
    h_rval = rval[0] 
                            
    return h_rval
