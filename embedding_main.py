# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

from gensim.models import Word2Vec, FastText
from tqdm import tqdm

import argparse
import logging
import multiprocessing


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model_constants = ['word2vec', 'fasttext']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tool for embedding')
    parser.add_argument('--corpus_path', type=str, required=True,
                        help='path of corpus')
    parser.add_argument('--output_path', type=str, required=True,
                        help='path of output')
    parser.add_argument('--size', type=int, required=True,
                        help='dimension size')
    parser.add_argument('--model', type=str, required=True,
                        help='model name')

    args = parser.parse_args()
    corpus_path = args.corpus_path
    output_path = args.output_path
    dim_size = args.size
    model_name = args.model
    worker = multiprocessing.cpu_count()

    if model_name not in model_constants:
        raise ValueError('invalid model name. check again.')

    corpus = []

    with open(args.corpus_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            corpus.append(line.strip().split(" "))
        f.close()

    if model_name == 'word2vec':
        model = Word2Vec(corpus, size=dim_size, workers=worker, sg=1, iter=10)
        model.save(output_path)
