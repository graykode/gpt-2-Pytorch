import os
import json
import torch
import numpy as np
import model_def, sample, encoder

def sample_model(model_name='117M', seed=None, nsamples=0, batch_size=1, length=None, temperature=1, top_k=0):
    np.random.seed(seed)
    enc = encoder.get_encoder(model_name)

    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        parameters = json.load(f)

    model = model_def.GPT2(n_vocab=parameters['n_vocab'],
                           n_ctx =parameters['n_ctx'],
                           n_embd=parameters['n_embd'],
                           n_head=parameters['n_head'],
                           n_layer=parameters['n_layer'])

    if length is None:
        length = parameters['n_ctx']
    elif length > parameters['n_ctx']:
        raise ValueError("Can't get samples longer than window size: %s" % parameters['n_ctx'])

    output = sample.sample_sequence(
        model=model, length=length,
        start_token=enc.encoder['<|endoftext|>'],
        batch_size=batch_size,
        temperature=temperature, top_k=top_k
    )[:, 1:]

    # @TODO replace tensorflow to pytorch model
    # saver = tf.train.Saver()
    # ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
    # saver.restore(sess, ckpt)

    # model.load_state_dict(torch.load(os.path.join('models', model_name)))
    # model.eval()

    generated = 0
    while nsamples == 0 or generated < nsamples:
        for i in range(batch_size):
            generated += batch_size
            text = enc.decode(output[i])
            print(text)
            print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)

if __name__ == '__main__':
    sample_model()