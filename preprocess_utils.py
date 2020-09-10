# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import os, sys, mmap
import re


from typing import List, Any, Optional
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


"""
The utility class to preprocess the given corpus
"""


class PreprocessUtils(object):
    def __init__(self,
                 base: str=None) -> None:
        super(PreprocessUtils, self).__init__()

        if base is None:
            self.base = 'wiki_'

        self.corpus = []


    def process_files(self, files) -> None:
        fsize = Path(f'./data/{self.base + files}').stat().st_size

        tot = 0

        with open(f'./data/{self.base + files}', 'r') as fp:
            with tqdm(total=fsize, desc=files) as pbar:
                mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
                for line in iter(mm.readline, b""):
                    try:
                        # TODO => preprocess util
                        preprocessed = self.preprocess_text(line.decode('utf-8').strip())

                        if preprocessed is not None:
                            self.corpus.append(preprocessed.strip())

                        tot += len(line)
                        pbar.update(tot - pbar.n)
                    except Exception as e:
                        continue

                mm.close()

    def preprocess_text(self, text: str, min_text_len: int=16, lower: bool=True) -> Optional[str]:
        text = re.sub(r'[^ .,?!/@$%~％·∼()\x00-\x7F가-힣]+', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\{.*?\}', '', text)

        if len(text) < min_text_len:
            return None
        else:
            if lower:
                return text.lower()
            else:
                return text

    def write_file(self, file_name: str) -> None:
        with open(file_name, 'w', encoding='utf-8') as f:
            for line in tqdm(self.corpus, desc='write files ...'):
                f.write(f'{line}\n')
            f.close()


if __name__ == '__main__':
    files = ['1.dat', '2.dat', '3.dat', '4.dat', '5.dat', '6.dat']
    with ThreadPoolExecutor() as executor:
        p = PreprocessUtils()
        executor.map(p.process_files, files)

    p.write_file('./pre_all.txt')