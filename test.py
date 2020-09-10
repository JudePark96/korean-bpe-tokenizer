# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load('{}.model'.format('kor-bpe-26000'))
token = sp.EncodeAsPieces('순수결정체는 무슨 개뿔.')
print(sp.EncodeAsIds('순수결정체는 무슨 개뿔.'))
print(token)
