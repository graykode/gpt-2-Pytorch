import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

import utils, attention

class GPT2(nn.Module):
    def __init__(self, n_vocab, n_ctx, n_embd, n_head, n_layer):
        super(GPT2, self).__init__()
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.attn = attention.attention(n_vocab, n_ctx, n_embd, n_head, n_layer)

    def gelu(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def expand_tile(self, value, size):
        """Add a new axis of given size."""
        ndims = len(value.shape)
        return value.unsqueeze(dim=0)[:,:-1].repeat([size] + [1] * ndims)

    def positions_for(self, tokens, past_length):
        batch_size = tokens.shape[0]
        nsteps = tokens.shape[1]
        return self.expand_tile(past_length + torch.LongTensor([i for i in range(0, nsteps + 1)]), batch_size)

    def norm(self, x, axis=-1, epsilon=1e-5):
        n_state = x.shape[-1]
        g = Variable(torch.ones([n_state]), name='g')
        b = Variable(torch.zeros([n_state]), name='b')
        u = torch.mean(x, dim=axis, keepdim=True)
        s = torch.mean(torch.pow((x-u), 2), dim=axis, keepdim=True)
        x = (x - u) * torch.rsqrt(s + epsilon)
        x = x * g + b
        return x

    def mlp(self, x,  n_state):
        nx = x.shape[-1]
        h = self.gelu(utils.conv1d(x, nf=n_state))
        h2 = utils.conv1d(h, nf=nx)
        return h2

    def block(self, x, past):
        nx = x.shape[-1]
        a, present = self.attn(x=self.norm(x), n_state=nx, past=past)
        x = x + a
        m = self.mlp(self.norm(x), nx * 4)
        x = x + m
        return x, present

    def forward(self, X, past):
        results = {}
        batch, sequence = utils.shape_list(X)

        wpe = Variable(torch.randn([self.n_ctx, self.n_embd]), name='wpe')
        wte = Variable(torch.randn([self.n_vocab, self.n_embd]), name='wte')

        past_length = 0 if past is None else past.shape[-2]
        h = wte[X] + wpe[self.positions_for(X, past_length)]

        # Transformer
        presents = []
        pasts = torch.unbind(past, dim=1) if past is not None else [None] * self.n_layer
        assert len(pasts) == self.n_layer

        for layer, past in enumerate(pasts):
            h, present = self.block(h, past=past)
            presents.append(present)
        results['present'] = torch.stack(presents, dim=1)
        h = self.norm(h)

        # Language model loss.  Do tokens <n predict token n?
        h_flat = torch.reshape(h, [batch * sequence, self.n_embd])
        logits = torch.matmul(h_flat, wte.t())
        logits = torch.reshape(logits, [batch, sequence, self.n_vocab])
        results['logits'] = logits
        return results


