'''
download the word embedding and filter it based on our corpus
'''

import urllib.request
import os
import shutil
import zipfile
import re
import numpy as np


DIR_PATH = os.path.dirname(__file__)

GLOVE_WE_URL = 'http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip'

WE_FOLDER = os.path.join(DIR_PATH, 'word_embedding')
ZIP_FILE = os.path.join(WE_FOLDER, 'glove.42B.300d.zip')
WE_FILE = os.path.join(WE_FOLDER, 'glove.42B.300d.txt')
# WE_FILE = os.path.join(WE_FOLDER, 'glove.6B.300d.txt')

EMBEDDING_SIZE = 300

CORPUS_FILES = [
    os.path.join(DIR_PATH, file) for file in [
        'aol_fmt.txt'
    ]
]

OUTPUT_FILES = [
    os.path.join(WE_FOLDER, 'filtered.glove.42B.300d.txt')
]

def main():
    if not os.path.exists(WE_FOLDER):
        print('create folder', WE_FOLDER)
        os.mkdir(WE_FOLDER)

    if not os.path.exists(WE_FILE):
        print('download embedding from', GLOVE_WE_URL)
        # Download the file from `url` and save it locally under:
        with urllib.request.urlopen(GLOVE_WE_URL) as response, open(ZIP_FILE, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(WE_FOLDER)

        os.remove(ZIP_FILE)

    voc = {}
    for corpus_file in CORPUS_FILES:
        print('read corpus:', corpus_file)

        with open(corpus_file, 'r', encoding='utf8') as file:
            lines = file.read().strip().split('\n')
            # first line is header
            for line in lines:
                parts = line.split('\t')
                tokens = parts[1].split(' ')

                for token in tokens:
                    if token not in voc:
                        voc[token] = 0
                    voc[token] += 1

    print('size of the voc:', len(voc))

    voc = sorted(voc.items(), key=lambda i: i[1], reverse=True)

    voc_size = 10 ** 5
    voc = voc[:voc_size]
    print('trim voc to the size of:', len(voc))
    print('minimum freq in voc:', voc[-1][1])

    default_ebd = ' '.join(['0.0'] * EMBEDDING_SIZE)
    we = {t[0]: default_ebd for t in voc}

    # unknown
    we['<unk>'] = default_ebd

    with open(WE_FILE, 'r', encoding='utf8') as file:
        # emdedding file is too large, read line by line
        size = 0
        line = file.readline().rstrip('\n')
        i = 1
        while line:
            space_idx = line.find(' ')

            token = line[: space_idx]

            if token in we:
                size += 1
                we[token] = line[space_idx + 1:]

            if i % 50000 == 0:
                print('read word embedding lines:', i)

            line = file.readline().rstrip('\n')
            i += 1

        print('size of the mapped embedding:', size)

    lines = [k + ' ' + v for k, v in we.items()]

    # embedding is too big to commit
    splited_lines = np.array_split(lines, len(OUTPUT_FILES))
    for lines, output_file in zip(splited_lines, OUTPUT_FILES):
        with open(output_file, 'w', encoding='utf8') as file:
            file.write('\n'.join(lines))


if __name__ == '__main__':
    main()
