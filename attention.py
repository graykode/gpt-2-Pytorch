import torch
import torch.nn as nn
import utils

class attention(nn.Module):
    def __init__(self, n_vocab, n_ctx, n_embd, n_head, n_layer):
        super(attention, self).__init__()
        self.n_head = n_head

    def split_states(self, x, n):
        """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
        *start, m = utils.shape_list(x)
        return torch.reshape(x, start + [n, m // n])

    def split_heads(self, x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return self.split_states(x, self.n_head).permute(0, 2, 1, 3)

    def mask_attn_weights(self, w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = utils.shape_list(w)
        b = utils.attention_mask(nd, ns, dtype=w.dtype)
        b = torch.reshape(b, [1, 1, nd, ns])
        w = w * b -  torch.Tensor([1e10]).to(w.dtype) * (1 - b)
        return w

    def multihead_attn(self, q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = torch.matmul(q, k.transpose(2,3))
        w = w * torch.rsqrt(torch.Tensor([v.shape[-1]]).to(w.dtype))
        w = self.mask_attn_weights(w)
        w = utils.softmax(w)
        a = torch.matmul(w, v)
        return a

    def merge_states(self, x):
        """Smash the last two dimensions of x into a single dimension."""
        *start, a, b = utils.shape_list(x)
        return torch.reshape(x, start + [a * b])

    def merge_heads(self, x):
        # Reverse of split_heads
        return self.merge_states(x.permute(0, 2, 1, 3))

    def forward(self, x, n_state, past):
        assert len(x.shape) == 3  # Should be [batch, sequence, features]
        assert n_state % self.n_head == 0
        if past is not None:
            assert len(past.shape) == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

        c = utils.conv1d(x, nf=n_state * 3)
        q, k, v = map(self.split_heads, c.split(c.shape[-1]//3, dim=2))
        present = torch.stack([k, v], dim=1)

        if past is not None:
            pk, pv = torch.unbind(past, dim=1)
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)
        a = self.multihead_attn(q, k, v)
        a = self.merge_heads(a)
        a = utils.conv1d(a, nf=n_state)

        return a, present