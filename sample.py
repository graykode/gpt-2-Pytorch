import model_def
import torch
import numpy as np
import torch.nn as nn
import utils

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = torch.top_k(input=logits, k=k)
        min_values = values[:, -1, np.newaxis]
        return torch.where(
            logits < min_values,
            torch.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return lambda: logits if torch.equal(k, 0) is True else lambda: _top_k()

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.LongTensor(batch_size, 1).fill_(start_token)

    def step(tokens, past=None):
        lm_output = model(X=tokens, past=past)
        logits = lm_output['logits'][:, :, :model.n_vocab]
        presents = lm_output['present']
        return {
            'logits': logits,
            'presents': presents,
        }

    def body(past, prev, output, length):
        for i in range(length):
            next_outputs = step(prev[:, np.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :] / torch.Tensor([temperature]).float()
            logits = top_k_logits(logits, k=top_k)
            logits = utils.softmax(logits) # avoid encountering probability entry < 0

            samples = torch.multinomial(logits, num_samples=1)
            past = torch.cat([past, next_outputs['presents']], dim=-2)
            output = torch.cat([output, samples], dim=1)

        return output

    context_output = step(tokens=context[:, :-1])
    tokens = body(context_output['presents'], context[:, -1], context, length)
    return tokens