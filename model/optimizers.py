import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def SGD(tparams, cost, inps, lr):
    """ default: lr=0.01 """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, 5):
        grads = [g*5/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)
    
    updates = []

    for p, g in zip(tparams.values(), gshared):        
        updated_p = p - lr * g
        updates.append((p, updated_p))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 

def RMSprop(tparams, cost, inps, lr, rho=0.9, epsilon=1e-6):
    """ default: lr=0.001 
        This is the implementation of the RMSprop algorithm used in
        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf.
    """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, 5):
        grads = [g*5/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)     
    
    updates = []

    for p, g in zip(tparams.values(), gshared):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        updates.append((acc, acc_new))
        
        updated_p = p - lr * (g / tensor.sqrt(acc_new + epsilon))
        updates.append((p, updated_p))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update

def SGLD(tparams, cost, inps, ntrain, lr):
    """ default: lr=0.01 """

    trng = RandomStreams(123)

    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, 5):
        grads = [g*5/norm for g in grads]

    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k)
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)

    updates = []

    for p, g in zip(tparams.values(), gshared):
        eps = trng.normal(p.get_value().shape, avg = 0.0, std = 1.0, 
                          dtype=theano.config.floatX)

	updated_p = p - lr * (g-p/ntrain) + tensor.sqrt(lr*2./ntrain) * eps
        updates.append((p, updated_p))

    f_update = theano.function([lr,ntrain], [], updates=updates)

    return f_grad_shared, f_update

      
def pSGLD(tparams, cost, inps, ntrain, lr, rho=0.9, epsilon=1e-6, clip_norm=5):
    """ default: lr=0.001 """
    
    trng = RandomStreams(123)
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)     
    
    updates = []
    
    for p, g in zip(tparams.values(), gshared):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        updates.append((acc, acc_new))
        
        G = tensor.sqrt(acc_new + epsilon)
        
        eps = trng.normal(p.get_value().shape, avg = 0.0, std = 1.0, 
                          dtype=theano.config.floatX)
        
        updated_p = p - lr * (g-p/ntrain) / G + tensor.sqrt(lr/G)*2./ntrain * eps 
        updates.append((p, updated_p))
    
    f_update = theano.function([lr,ntrain], [], updates=updates)
    
    return f_grad_shared, f_update