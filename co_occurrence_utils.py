# -*- coding: utf-8 -*-
# this code revised from https://github.com/lovit/soynlp/blob/3497e958035e24f9204d91572252fae61791e6d7/soynlp/vectorizer/_word_context.py

import psutil
import sys, os


from scipy.sparse import csr_matrix
from collections import defaultdict
from tqdm import tqdm


def get_process_memory():
    """It returns the memory usage of current process"""

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def _print_status(message, i_sent, new_line=False):
    print('\r{} from {} sents, mem={} Gb'.format(
            message, i_sent, '%.3f' % get_process_memory()),
        flush=True, end='\n' if new_line else ''
    )

def _scanning_vocabulary(sents, min_tf, verbose):

    # counting word frequency, first
    word_counter = defaultdict(int)

    for i_sent, sent in enumerate(sents):

        if verbose and i_sent % 1000 == 0:
            _print_status('  - counting word frequency', i_sent)

        # words = tokenizer(sent)
        for word in sent:
            word_counter[word] += 1

    if verbose:
        _print_status('  - counting word frequency', i_sent, new_line=True)

    # filtering with min_tf
    vocab2idx = {word for word, count in word_counter.items() if count >= min_tf}
    vocab2idx = {word:idx for idx, word in enumerate(
        sorted(vocab2idx, key=lambda w:-word_counter[w]))}
    idx2vocab = [word for word, _ in sorted(vocab2idx.items(), key=lambda w:w[1])]
    del word_counter

    return vocab2idx, idx2vocab

def _word_context(sents, windows, dynamic_weight, verbose, vocab2idx):
    # scanning (word, context) pairs
    word2contexts = defaultdict(lambda: defaultdict(int))

    if dynamic_weight:
        weight = [(windows-i)/windows for i in range(windows)]
    else:
        weight = [1] * windows

    for i_sent, sent in enumerate(sents):

        if verbose and i_sent % 1000 == 0:
            _print_status('  - scanning (word, context) pairs', i_sent)

        words = sent
        if not words:
            continue

        n = len(words)

        for i, word in enumerate(words):
            if not (word in vocab2idx):
                continue

            # left_contexts
            for w in range(windows):
                j = i - (w + 1)
                if j < 0 or not (words[j] in vocab2idx):
                    continue
                word2contexts[word][words[j]] += weight[w]

            # right_contexts
            for w in range(windows):
                j = i + w + 1
                if j >= n or not (words[j] in vocab2idx):
                    continue
                word2contexts[word][words[j]] += weight[w]

    if verbose:
        _print_status('  - scanning (word, context) pairs', i_sent, new_line=True)

    return word2contexts


def sent_to_word_contexts_matrix(sents, windows=3, min_tf=10, dynamic_weight=False, verbose=True):

    """
    :param dynamic_weight : Use dynamic weight if True.
        co-occurrence weight = [1, (w-1)/w, (w-2)/w, ... 1/w]
    """

    if verbose:
        print('Create (word, contexts) matrix')

    vocab2idx, idx2vocab = _scanning_vocabulary(
        sents, min_tf, verbose)

    word2contexts = _word_context(
        sents, windows, dynamic_weight, verbose, vocab2idx)

    x = _encode_as_matrix(word2contexts, vocab2idx, verbose)

    if verbose:
        print('  - done')
    return x, idx2vocab


def _encode_as_matrix(word2contexts, vocab2idx, verbose):

    rows = []
    cols = []
    data = []
    for word, contexts in word2contexts.items():
        word_idx = vocab2idx[word]
        for context, cooccurrence in contexts.items():
            context_idx = vocab2idx[context]
            rows.append(word_idx)
            cols.append(context_idx)
            data.append(cooccurrence)
    x = csr_matrix((data, (rows, cols)))

    if verbose:
        print('  - (word, context) matrix was constructed. shape = {}{}'.format(
            x.shape, ' '*20))

    return x

if __name__ == '__main__':
    corpus = []

    with open('./output/test_encoded_corpus.txt', 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            corpus.append(line.strip().split(' '))

        f.close()

    x, idx2vocab = sent_to_word_contexts_matrix(
        corpus,
        windows=5,
        min_tf=5,
        dynamic_weight=True,
        verbose=True)

    print(x)