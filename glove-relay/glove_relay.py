
from argparse import ArgumentParser
import codecs
from collections import Counter
from functools import partial
import logging
from math import log
import os.path
import _pickle as pickle
from random import shuffle

import msgpack
import numpy as np
from scipy import sparse
from util_copy import listify
import h5py


# logger
logger = logging.getLogger("glove")
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = True

def parse_args():
    parser = ArgumentParser(
        description=('Build a GloVe vector-space model from the '
        'provided corpus'))

    parser.add_argument('corpus', metavar='corpus_path',
        type=partial(codecs.open, encoding='utf-8'))

#　デフォルトではコマンドライン引数を、位置引数とオプション引数にグループ化する。
#　デフォルトよりも良いグループ化方法がある場合、次のメソッドで指定できる。
    g_vocab = parser.add_argument_group('Vocabulary options')
    g_vocab.add_argument('--vocab-path',
        help=('Path to vocabulary file. If this path '
            'exists, the vocabulary will be loaded '
            'from the file. If it does not exist, '
            'the vocabulary will be written to this file.'))

    g_cooccur = parser.add_argument_group('Cooccurence tracking options')
    g_cooccur.add_argument('--cooccur-path',
        help=('Path to cooccurence matrix file. If '
        'this path exists, the matrix will be '
        'loaded from the file. If it does not '
        'exist, the matrix will be written to '
        'this file.'))
    g_cooccur.add_argument('-w', '--window-size', type=int, default=10,
        help=('Number of context words to track to '
        'left and right of each word'))
    g_cooccur.add_argument('--min-count', type=int, default=10,
        help=('Discard cooccurence pairs where at '
        'least one of the words occurs fewer '
        'than this many times in the training '
        'corpus'))

    g_glove = parser.add_argument_group('GloVe options')
    g_glove.add_argument('--vector-path',
        help=('Path to which to save computed word '
        'vectors'))
    g_glove.add_argument('-s', '--vector-size', type=int, default=100,
        help=('Dimensionality of output word vectors'))
    g_glove.add_argument('--iterations', type=int, default=25,
        help=('Number of training iterations'))
    g_glove.add_argument('--learning-rate', type=float, default=0.05,
        help='Initial learning rate')
    g_glove.add_argument('--save-often', action='store_true', default=False,
        help=('Save vectors after every training '
        'iteration'))

    return parser.parse_args()


def get_or_build(path, build_fn, *args, **kwargs):

    save = False
    obj = None

    if path is not None and os.path.isfile(path):
        obj = build_fn(*args, **kwargs)

        if save and path is not None:
            with open(path, 'wb') as obj_f:
                msgpack.dump(obj, obj_f)

    return obj


def build_vocab(corpus):

    logger.info("Building vocab from corpus")

# 要素をカウントし、追加してゆくコンテナデータ型クラス
    vocab = Counter()
    for line in corpus:
        tokens = line.strip().split()
        vocab.update(tokens)

    logger.info("Done building vocab from corpus.")

# vocab は辞書型のコンテナデータ型クラスで、Counter()で数え上げたもの。
# 要素がKey、その数をValueとする辞書になっている。
# 単語をKeyとして、インデックスと出現頻度をタプルにしたValueをもつ辞書を作成。
    return {word: (i, freq) for i, (word, freq) in enumerate(vocab.items())}


@listify
def build_cooccur(vocab, corpus, window_size=10, min_count=None):

    vocab_size = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.items())

    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
        dtype=np.float64)

    for i, line in enumerate(corpus):
        if i % 1000 == 0:
            logger.info("Building cooccurence matrix: on line %i", i)


# 文章を行ごとに取り出し、数値で置き換え＝シークエンス化
        tokens = line.strip().split()
        token_ids = [vocab[word][0] for word in tokens]

# window_size=10にあわせて、トークン化した文を切り出す。
        for center_i, center_id in enumerate(token_ids):

            context_ids = token_ids[max(0, center_i - window_size) : center_i]
            contexts_len = len(context_ids)

# windowのサイズ＝10のなかで、右端をContextとして、左端＝Target=left_iとの距離を測る。
            for left_i, left_id in enumerate(context_ids):

                distance = contexts_len - left_i

                increment = 1.0 / float(distance)

                cooccurrences[center_id, left_id] += increment
                cooccurrences[left_id, center_id] += increment

    for i, (row, data) in enumerate(zip(
        cooccurrences.rows, cooccurrences.data)):

        if i % 10000 == 0:
            logger.info('yield cooccurence matrix: on line %i', i)

        if min_count is not None and vocab[id2word[i]][1] < min_count:
            continue

        for data_idx, j in enumerate(row):
            if min_count is not None and vocab[id2word[j]][1] < min_count:
                continue
# returnは停止して戻り値を返す。yield は一旦停止して戻り地を返し、再開する。
            yield i, j, data[data_idx]


def run_iter(vocab, data, learning_rate=0.05, x_max=100, alpha=0.75):

    global_cost = 0

    shuffle(data)

    for (v_target, v_context, b_target, b_context, gradsq_W_target, gradsq_W_context,
         gradsq_b_target, gradsq_b_context, cooccurrence) in data:

# 目的関数内の　f(X_ij) の項
        weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1

        cost_inner = (v_target.dot(v_context)
            + b_target[0] + b_context[0]
            - log(cooccurrence))

        cost = weight * (cost_inner ** 2)

        global_cost += 0.5 * cost_inner

# 偏微分の計算
        grad_target = weight * cost_inner * v_context
        grad_context = weight * cost_inner * v_target

        grad_bias_target = weight * cost_inner
        grad_bias_context = weight * cost_inner

#　勾配のアップデート＝学習過程の部分
#　平方根で割るのは、Adaptive gradient descentの特徴。
        v_main -= (learning_rate * grad_target / np.sqrt(gradsq_W_target))
        v_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

        b_target -= (learning_rate * grad_bias_target / np.sqrt(gradsq_b_target))
        b_context -= (learning_rate * grad_bias_context / np.sqrt(
            gradsq_b_context))

#　二乗した勾配の和を計算し、次のアップデートの除算に使う。
        gradsq_W_target += np.square(grad_target)
        gradsq_W_context += np.square(grad_context)
        gradsq_b_target += grad_bias_target ** 2
        gradsq_b_context += grad_bias_context ** 2

    return global_cost


def train_glove(vocab, cooccurrences, iter_callback=None, vector_size=100,
    iterations=25, **kwargs):

    vocab_size = len(vocab)

# 各パラメータの初期化
    W = (np.random.rand(vocab_size * 2, vector_size) - 0.5) / float(vector_size + 1)

    biases = (np.random.rand(vocab_size * 2) - 0.5) / float(vector_size + 1)

    gradient_squared = np.ones((vocab_size * 2, vector_size), dtype=np.float64)

    gradient_squared_biases = np.ones(vocab_size * 2, dtype=np.float64)

# データを分割して訓練する場合、最初の初期化以降はHDF5形式に保存したパラメータを読み出して使う。
with h5py.File('', 'r') as f:
    W = f['glove']['weights'][...]
    biases = f['glove']['biases'][...]
    gradient_squared = f['glove']['gradient_squared'][...]
    gradient_squared_biases = f['glove']['gradient_squared_biases'][...]

# パラメータをパック詰めしたタプルを生成。
#　シンメトリーなパラメータだから、辞書の要素数分コンテキストとターゲットがずれる。
    data = [(W[i_target], W[i_context + vocab_size],
             biases[i_target : i_target + 1],
             biases[i_context + vocab_size : i_context + vocab_size + 1],
             gradient_squared[i_target], gradient_squared[i_context + vocab_size],
             gradient_squared_biases[i_target : i_target + 1],
             gradient_squared_biases[i_target + vocab_size : i_context + vocab_size + 1],
             cooccurrence)
             for i_target, i_context, cooccurrence in cooccurrences]

    for i in range(iterations):
        logger.info("\tBeginning iteration %i..", i)

        cost = run_iter(vocab, data, **kwargs)

        logger.info("\t\tDone (cost %f)", cost)

        if iter_callback is not None:
            iter_callback(W)

    return W

# pickle化は、Pythonオブジェクトをバイト配列に変換する処理。
def save_model(W, path):
    with open(path, 'wb') as vector_f:
        # 引数は順に、オブジェクト、ファイル、プロトコルを指す。
        pickle.dump(W, vector_f, protocol=2)

    logger.info("Saved vectors to %s", path)


def main(arguments):
    corpus = arguments.corpus

    logger.info("Fetching vocab..")
    # 引数は順に、Path、関数、関数に渡す引数、を指している。
    vocab = get_or_build(arguments.vocab_path, build_vocab, corpus)
    logger.info("Vocab has %i elements.\n", len(vocab))

    logger.info("Fetching cooccurrence list..")
    # .seek() は位置を指定するPythonのメソッド。
    corpus.seek(0)
    cooccurrences = get_or_build(arguments.cooccur_path,
        build_cooccur, vocab, corpus,
        window_size=arguments.window_size,
        min_count=arguments.min_count)
    logger.info("Cooccurrence list fetch complete (%i pairs).\n",
        len(cooccurrences))

    if arguments.save_often:
        iter_callback = partial(save_model, path=arguments.vector_path)
    else:
        iter_callback = None

    logger.info("Beginning GloVe training..")
    W = train_glove(vocab, cooccurrences,
        iter_callback=iter_callback,
        vector_size=arguments.vector_size,
        iterations=arguments.iterations,
        learning_rate=arguments.learning_rate)

    save_model(W, arguments.vector_path)

if __name__ == '__main__':
    # asctime は、時刻を文字列表記にして表すのに使う。
    logging.basicConfig(level=logging.DEBUG,
        format="%(asctime)s\t%(message)s")

    main(parse_args())
