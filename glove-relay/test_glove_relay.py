import logging
import numpy as np

import evaluate_relay
import glove_relay
import _pickle as pickle


# logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

logger.info('Hello')

# データを分割した訓練をする前に、使用するテキストデータ全体の単語の辞書を作成。
# 数値＝トークンをValueにもつ単語辞書、KeyとValueを入れ替えた辞書を、dictionary形式に保存しpickle化。
# 訓練毎に、２つの辞書を読み込んで使用する。
with open('vocab_dict_relay.pickle', mode='rb') as f:
    vocab_dict = pickle.load(f)
    vocab = vocab_dict['vocab']
    id2word = vocab_dict['id2word']


# Wikiped日本語版の１００MB分のテキストデータ
with open("preprocessed.txt") as f:
    corpus = f.readlines()

test_corpus = []
for line in corpus:
    striped = line.lower().strip()
    test_corpus.append(striped)

logger.info('Test corpus 1-10: ', test_corpus[:10])
logger.info('読み込み完了')

glove_py3.logger.setLevel(logging.DEBUG)
vocab = glove_py3.build_vocab(test_corpus)

cooccur = glove_py3.build_cooccur(vocab, test_corpus, window_size=10)

id2word = evaluate_py3.make_id2word(vocab)

#vocab_dict = {}
#vocab_dict['vocab'] = vocab
#vocab_dict['id2word'] = id2word

#with open('vocab_dict.pickle', mode='wb') as f:
#    pickle.dump(vocab_dict, f)



W = glove_py3.train_glove(vocab, cooccur, vector_size=100, iterations=10)

W = evaluate_py3.merge_target_context(W)


def test_similarity():
    similar = evaluate_py3.most_similar(W, vocab, id2word, '言語')
    logging.debug(similar)

    #assert_equal('trees', similar[0])
    print('\nもっとも近い単語:', similar[0])
    print(similar[:])

test_similarity()
