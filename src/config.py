# -*- coding: utf-8 -*-

# Corpus, path relateds to /src
CORPUS = ['data/aol/' + file for file in [
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
]]

# checkpoints relevant
SAVE_DIR = 'checkpoints'
MODEL_NAME = 'default'
SAVE_EVERY = 5000           # save the checkpoint every x iterations

# Iterations of training
N_ITER = 150 * 10 ** 3

# Configure models - chat relevant
BEAM_SEARCH_ON = True        # use Beam Search or Greedy Search
BEAM_WIDTH = 10              # size of beam
# Return the 'best' or 'random' result
BEAM_MODE = 'random'

# Configure models - training relevant
RNN_TYPE = 'LSTM'            # use LSTM or GRU as RNN
ATTN_TYPE = 'dot'           # type of the attention model: dot/general/concat
HIDDEN_SIZE = 300            # size of the word embedding & number of hidden units in GRU
WORD_EMBEDDING_SIZE = 300
MODEL_LAYERS = 2
MODEL_DROPOUT_RATE = 0.1
ENCODER_DROPOUT_RATE = 0.1   # dropout rate in bi-GRU encoder
DECODER_DROPOUT_RATE = 0.1   # dropout rate in GRU decoder
TEACHER_FORCING_RATIO = 1.0  # ratio for training decoder on ground truth or last output of decoder

# Configure training/optimization
BATCH_SIZE = 64            # size of the mini batch in training state
CLIP = 50.0                # gradient norm clip
LR = 0.0001                # encoder learning ratio
DECODER_LR = 5.0           # decoder learning ratio: LR*DECODER_LR
PRINT_EVERY = 500          # print the loss every x iterations
TF_RATE_DECAY_FACTOR = 15 * 10 ** 3      # k in the inverse sigmoid decay func of the teacher force rate k/(k+exp(i/k)), which is related to N_ITER

# Parameters for processing the dataset
MAX_LENGTH = 20  # Maximum sentence length to consider


# Word Embedding
WORD_EMBEDDING_FILES = [
    'data/word_embedding/filtered.glove.42B.300d.part1.txt',
    'data/word_embedding/filtered.glove.42B.300d.part2.txt'
]
