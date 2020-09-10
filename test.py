# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load('./token_model/{}.model'.format('kor-bpe-26000'))
token = sp.EncodeAsPieces('순수결정체는 무슨 개뿔.')
print(sp.EncodeAsIds('순수결정체는 무슨 개뿔.'))
print(token)
print(' '.join(token))

print(sp.DecodePieces('▁순수 결정 체는 ▁무슨 ▁개 뿔 .'.split(' ')))
print(sp.DecodePieces('▁영화는 ▁1982 년에 ▁연재 된 ▁크리스 ▁클레 어 먼 트와 ▁프랭크 ▁밀 러의 ▁" w ol ver ine " 을 ▁원작으로 ▁하며 , ▁일본을 ▁무대로 ▁무 사와 ▁대립 하는 ▁주인공 ▁로 건의 ▁모습을 ▁그 린다 .'.split(' ')))