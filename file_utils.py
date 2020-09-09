# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import os, sys, mmap


from typing import List
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


base = "wiki_"
files = ['1.dat', '2.dat', '3.dat', '4.dat', '5.dat', '6.dat']


def read_return_text_list(filename: str) -> List[str]:
    data = []

    with open(filename, 'r', encoding='ISO-8859-1') as f:
        for line in tqdm(f):
            data.append(line.strip())
        f.close()

    return data


def process_file(file):
    fsize = Path(f'./data/{base + file}').stat().st_size

    result = []
    tot = 0

    with open(f'./data/{base + file}', 'r') as fp:
        with tqdm(total=fsize, desc=file) as pbar:
            mm = mmap.mmap(fp.fileno(), 0,access=mmap.ACCESS_READ)
            for line in iter(mm.readline, b""):
                try:
                    # TODO => preprocess util
                    result.append(line.decode('utf-8').strip())
                except Exception as e:
                    print(e)
                tot += len(line)
                pbar.update(tot - pbar.n)
            mm.close()

    return result


if __name__ == '__main__':
    with ThreadPoolExecutor() as executor:
        result = executor.map(process_file, files)
        print(list(result)[0][0])