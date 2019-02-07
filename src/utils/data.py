# -*- coding: utf-8 -*-

import re
import unicodedata
import itertools
import torch
import config

# Lowercase and remove non-letter characters
def normalize_str(s):
    s = s.lower()
    # give a leading & ending spaces to punctuations
    s = re.sub(r'([.!?,\'])', r' \1 ', s)
    # purge unrecognized token with space
    s = re.sub(r'[^a-z.!?,\']+', r' ', s)
    # squeeze multiple spaces
    s = re.sub(r'([ ]+)', r' ', s)
    # remove extra leading & ending space
    return s.strip()


def filter_pair(pair):
    '''
    Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
    '''

    return len(pair[0].split(' ')) <= config.MAX_LENGTH and len(pair[1].split(' ')) <= config.MAX_LENGTH

# Using the functions defined above, return a populated voc object and pairs list
def load_pairs(datafile):
    print("Start preparing training data ...")

    print("Reading lines from %s..." % datafile)
    # Read the file and split into lines
    with open(datafile, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
        lines = lines[1:]

    # Split every line into pairs
    lines = [[s for s in l.split('\t')] for l in lines]

    # normalize
    for line in lines:
        line[1] = normalize_str(line[1])

    pairs = []
    for i in range(1, len(lines)):
        line = lines[i]
        prev_line = lines[i - 1]
        if line[0] == prev_line[0]:
            pairs.append([prev_line[1], line[1]])

    print(f'Read {len(pairs)} sentence pairs')
    pairs = [pair for pair in pairs if filter_pair(pair)]
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))

    return pairs


def trim_unk_data(pairs, voc):
    # Filter out pairs with trimmed words
    keep_pairs = []

    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]

        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if not voc.has(word):
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if not voc.has(word):
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

def indexes_from_sentence(sentence, voc):
    unk = voc.unk
    eos = voc.eos

    tokens = [word if voc.has(word) else unk for word in sentence.split(' ')]
    tokens.append(eos)

    return [voc.get_index(token) for token in tokens]


def zip_padding(batches, fillvalue):
    return list(itertools.zip_longest(*batches, fillvalue=fillvalue))

def binary_mask(seqs, fillvalue):
    mask = []
    for batch in seqs:
        mask_batch = [0 if index == fillvalue else 1 for index in batch]
        mask.append(mask_batch)
    return mask

# Returns padded input sequence tensor and lengths
def input_var(input_batch, padding):
    lengths = torch.tensor([len(indexes) for indexes in input_batch])
    pad_list = zip_padding(input_batch, padding)

    pad_var = torch.LongTensor(pad_list)
    return pad_var, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def output_var(indexes_batch, padding):
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    pad_list = zip_padding(indexes_batch, padding)
    mask = binary_mask(pad_list, padding)
    mask = torch.ByteTensor(mask)

    pad_var = torch.LongTensor(pad_list)
    return pad_var, mask, max_target_len

# Returns all items for a given batch of pairs
def batch_2_seq(pair_batch, padding):
    # sort by input length, no idea why
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)

    input_batch = [pair[0] for pair in pair_batch]
    output_batch = [pair[1] for pair in pair_batch]

    inp, lengths = input_var(input_batch, padding)
    output, mask, max_target_len = output_var(output_batch, padding)

    # Return speaker_variable tensor with shape=(batch_size)
    return inp, lengths, output, mask, max_target_len


def data_2_indexes(pair, voc):
    return [
        indexes_from_sentence(pair[0], voc),
        indexes_from_sentence(pair[1], voc),
    ]
