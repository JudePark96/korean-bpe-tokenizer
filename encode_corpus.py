# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import sentencepiece as spm
from tqdm import tqdm

sp = spm.SentencePieceProcessor()
sp.Load('./token_model/{}.model'.format('kor-bpe-26000'))

corpus = []

with open('./output/pre_all.txt', 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc='reading corpus file...'):
        corpus.append(line.strip())

    f.close()

with open('./output/encoded_corpus.txt', 'w', encoding='utf-8') as f:
    for line in tqdm(corpus, desc='writing encoded corpus file...'):
        token = sp.EncodeAsPieces(line)
        f.write(' '.join(token)+'\n')
    f.close()