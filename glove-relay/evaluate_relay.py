from argparse import ArgumentParser
from functools import partial
import _pickle as pickle

import numpy as np
import h5py


def make_id2word(vocab):
    return dict((id, word) for word, (id, _) in vocab.items())


# シンメトリーなパラメータの平均を計算する。
def merge_target_context(W, merge_fun=lambda m, c: np.mean([m, c], axis=0),
    normalize=True):

    vocab_size = int(len(W) / 2)
    for i, row in enumerate(W[:vocab_size]):
        merged = merge_fun(row, W[i + vocab_size])
        if normalize:
            merged /= np.linalg.norm(merged)
        W[i, :] = merged

# 対称な箇所を足して平均をとったパラメータの保存
    with h5py.File('glove_weights_relay.h5', 'w') as f:
        f.create_group('glove')
        f['glove'].create_dataset('merged_weights', data=W[:vocab_size])

    return W[:vocab_size]


def most_similar(W, vocab, id2word, word, n=15):

    assert len(W) == len(vocab)

    word_id = vocab[word][0]

    dists = np.dot(W, W[word_id])
# Numpy.argsort()は、昇順でソートしたあとインデックスを返す。
# 降順にするには、[::-1]でインデックスを参照する。
    top_ids = np.argsort(dists)[::-1][:n + 1]

    return [id2word[id] for id in top_ids if id != word_id][:n]


def parse_args():
    parser = ArgumentParser(
        description=('Evaluate a GloVe vector-space model on a word '
            'analogy test set'))

    parser.add_argument('vectors_path', type=partial(open, mode='rb'),
        help=('Path to serialized vectors file as '
            'produced by this GloVe implementation'))

    parser.add_argument('analogies_paths', type=partial(open, mode='r'),
        nargs='+',
        help=('Paths to analogy text files, where each '
            'line consists of  four words separated by '
            'spaces `a b c d`, expressing the analogy '
            'a:b :: c:d'))

    return parser.parse_args()
