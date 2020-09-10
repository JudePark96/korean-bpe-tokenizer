# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import argparse


from tokenizers import (
    CharBPETokenizer,
    ByteLevelBPETokenizer,
    BertWordPieceTokenizer
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", type=str)
    parser.add_argument("--vocab_size", type=int, default=32000)
    args = parser.parse_args()

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=[args.corpus_file],
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    tokenizer.save('./', f'bpe-{str(args.vocab_size)}')


