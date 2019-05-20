"""
Module to deal with loading text data and sampling from it.
"""
import glob
import numpy as np
import os
import tqdm


def load_dataset(enc, path, combine):
    paths = []

    # Simple file
    if os.path.isfile(path):
        paths.append(path)

    # Directory
    elif os.path.isdir(path):
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))

    # Assume glob
    else:
        paths = glob.glob(path)

    # filter paths
    paths = [p for p in paths if '.DS_Store' not in p]

    token_chunks = []
    raw_text = ''
    for path in tqdm.tqdm(paths):

        if path.endswith('.npz'):

            # Pre-encoded
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        else:

            # Plain text
            with open(path, mode='r', encoding='utf-8') as fp:
                raw_text += fp.read()

            if len(raw_text) >= combine:
                tokens = np.stack(enc.encode(raw_text))
                token_chunks.append(tokens)
                raw_text = ''
            else:
                raw_text += '<|endoftext|>'

    if raw_text:
        tokens = np.stack(enc.encode(raw_text))
        token_chunks.append(tokens)

    return token_chunks


def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.

    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks, seed=None):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])
        self.rs = np.random.RandomState(seed=seed)

    def sample(self, length):
        assert length < self.total_size // len(self.chunks), "Dataset files are too small to sample {} tokens at a time".format(length)

        while True:
            index = self.rs.randint(0, self.total_size - length - 1)

            # i = boundary that index is in
            i = binary_search(lambda j: self.boundaries[j] > index, 0, len(self.boundaries) - 1) - 1

            # sample length fits within the chunk at the starting index in that chunk
            if self.boundaries[i + 1] > index + length:
                # finding start of boundary from index
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk: within_chunk + length]
