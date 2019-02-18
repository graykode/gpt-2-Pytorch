import torch
from torch.autograd import Variable

def shape_list(x):
    return [s if s is None else s for s in x.shape]

def conv1d( x, nf, w_init_stdev=0.02):
    *start, nx = shape_list(x)
    w = Variable(torch.randn(1, nx, nf), name='w')
    b = Variable(torch.zeros(1, nf), name='b')
    c = torch.reshape(torch.mm(torch.reshape(x, (-1, nx)), torch.reshape(w, (-1, nf)) + b), start + [nf])
    return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.
    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = torch.Tensor([i for i in range(nd)])[:,None]
    j = torch.Tensor([j for j in range(ns)])
    m = i >= j - ns + nd
    return m.to(dtype)

def softmax(x, axis=-1):
    if x.shape[axis] == 0:
        return x
    else:
        x = x - torch.max(x, dim=axis, keepdim=True)[0]
        ex = torch.exp(x)
        return ex / torch.max(ex, dim=axis, keepdim=True)[0]