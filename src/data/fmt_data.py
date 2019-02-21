'''
format the aol data to consume by the model
'''

import os
import random
import math
from datetime import datetime
from nltk.tokenize import RegexpTokenizer

DIR_PATH = os.path.dirname(__file__)

tokenizer = RegexpTokenizer('[a-z0-9]+', discard_empty=True)

CORPUS_FILES = [
    os.path.join(DIR_PATH, 'aol', file) for file in [
        'user-ct-test-collection-01.txt',
        'user-ct-test-collection-02.txt',
        'user-ct-test-collection-03.txt',
        'user-ct-test-collection-04.txt',
        'user-ct-test-collection-05.txt',
        'user-ct-test-collection-06.txt',
        'user-ct-test-collection-07.txt',
        'user-ct-test-collection-08.txt',
        'user-ct-test-collection-09.txt',
        'user-ct-test-collection-10.txt'
    ]
]

class Session:
    def __init__(self, aol_sid):
        self.sentences = []
        self.sid = aol_sid
        self.l_time = None

    def size(self):
        return len(self.sentences)

    def add(self, s, lt):
        self.l_time = lt
        if not self.sentences or s != self.sentences[-1]:
            self.sentences.append(s)

    def to_str(self, idx):
        return '\n'.join([
            str(idx) + '\t' + self.sentences
        ])



def main():
    sessions = []

    for corpus_file in CORPUS_FILES:
        print('read corpus:', corpus_file)

        with open(corpus_file, 'r', encoding='utf8') as file:
            lines = file.read().strip().split('\n')

            session = None

            # first line is header
            for line in lines[1:]:
                parts = line.split('\t')

                sid = parts[0]

                if session is None:
                    session = Session(sid)
                elif sid != session.sid:
                    if session.size() > 1:
                        sessions.append(session)

                    session = Session(sid)

                sentence = parts[1]
                tokens = tokenizer.tokenize(sentence.lower())
                if not tokens:
                    continue

                time = datetime.strptime(parts[2], '%Y-%m-%d %H:%M:%S')

                if session.size() != 0 and (time - session.l_time).seconds > 1800:
                    if session.size() > 1:
                        sessions.append(session)
                    session = Session(sid)

                session.add(' '.join(tokens), time)

            if session.size() > 1:
                sessions.append(session)

    print('size of sessions:', len(sessions))

    test_size = math.floor(len(sessions) * 0.15)

    random.shuffle(sessions)
    dev_sessions = sessions[-test_size:]
    sessions = sessions[:-test_size]
    test_sessions = sessions[-test_size:]
    sessions = sessions[:-test_size]

    data_map = {
        'train': sessions,
        'dev': dev_sessions,
        'test': test_sessions
    }

    for name, data in data_map.items():
        print(f'size of {name} sessions:', len(data))

        lines = []
        for i, session in enumerate(data):
            lines += [str(i) + '\t' + s for s in session.sentences]

        print(f'size of {name} queries:', len(lines))

        with open(os.path.join(DIR_PATH, 'aol', 'aol_fmt_' + name + '.txt'), 'w', encoding='utf8') as file:
            file.write('\n'.join(lines))

if __name__ == '__main__':
    main()
