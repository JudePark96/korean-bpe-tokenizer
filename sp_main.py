# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import sentencepiece as spm


templates = '--input={} --model_prefix={} --model_type={} --vocab_size={} --user_defined_symbols={}'
user_defined_symbols = '[pad],[unk],[cls],[sep],[mask],<s>,</s>'
prefix = 'kor'
model_type = 'bpe'
vocab_size = 26000
cmd = templates.format('./output/pre_all.txt', prefix, model_type, vocab_size, user_defined_symbols)

spm.SentencePieceTrainer.Train(cmd)
