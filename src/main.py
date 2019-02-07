"""
SouthPark Chatbot
"""

import os
import argparse
import torch

import config
from utils import CheckpointManager, Vocabulary
from utils.data import trim_unk_data, load_pairs, data_2_indexes
from search_decoder import GreedySearchDecoder, BeamSearchDecoder
from models import Seq2Seq
from trainer import Trainer
from evaluate import evaluateInput

DIR_PATH = os.path.dirname(__file__)
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

SAVE_PATH = os.path.join(DIR_PATH, config.SAVE_DIR, config.MODEL_NAME)

def init_word_embedding(embedding_paths):
    print('Init word embedding from: ', ', '.join(embedding_paths))

    lines = []
    for embedding_path in embedding_paths:
        embedding_path = os.path.join(DIR_PATH, embedding_path)

        with open(embedding_path, encoding='utf-8') as file:
            lines += file.read().strip().split('\n')

    tokens_of_lines = [l.strip().split(' ') for l in lines]
    words = [l[0] for l in tokens_of_lines]
    weight = [[float(str_emb) for str_emb in l[1:]] for l in tokens_of_lines]

    voc = Vocabulary(words)
    print('Vocabulary size:', voc.size())

    # also init the embedding for special tokens
    while len(weight) < voc.size():
        embedding_len = len(weight[0])
        weight.append([0] * embedding_len)

    weight = torch.FloatTensor(weight)

    return voc, weight

def load_data(corpus, voc, trim=False):
    pairs = []
    for corpus_path in corpus:
        datafile = os.path.join(DIR_PATH, corpus_path)
        pairs += load_pairs(datafile)

    # Trim pairs with words not in embedding
    if trim:
        pairs = trim_unk_data(pairs, voc)

    return [data_2_indexes(pair, voc) for pair in pairs]

def build_model(checkpoint):
    if checkpoint:
        voc = checkpoint['voc']

    else:
        # Initialize word embeddings
        voc, pre_we_weight = init_word_embedding(config.WORD_EMBEDDING_FILES)

    print(f'word embedding size {config.WORD_EMBEDDING_SIZE}, hidden size {config.HIDDEN_SIZE}, layers {config.MODEL_LAYERS}')

    word_ebd_shape = (voc.size(), config.WORD_EMBEDDING_SIZE)

    model = Seq2Seq(word_ebd_shape, config.HIDDEN_SIZE, config.MODEL_LAYERS, config.MODEL_DROPOUT_RATE, config.RNN_TYPE, config.ATTN_TYPE)

    if checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_pretrained_word_ebd(pre_we_weight)

    # Use appropriate device
    model = model.to(device)

    return model, voc



def train(mode, model, voc, checkpoint, checkpoint_mng):
    trainer = Trainer(model, voc, checkpoint_mng)

    if checkpoint:
        trainer.resume(checkpoint)
    else:
        checkpoint_mng.save_meta(voc=voc)

    corpus = config.CORPUS
    n_iter = config.N_ITER

    pairs = load_data(corpus, voc)

    # Ensure dropout layers are in train mode
    model.train()

    trainer.train(pairs, n_iter, config.BATCH_SIZE, stage=mode)


def run(model, voc):
    # Set dropout layers to eval mode
    model.eval()

    # Initialize search module
    if config.BEAM_SEARCH_ON:
        searcher = BeamSearchDecoder(model)
    else:
        searcher = GreedySearchDecoder(model)

    # Begin chatting (uncomment and run the following line to begin)
    evaluateInput(searcher, voc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices={'train', 'run'}, help="mode to run the network")
    parser.add_argument('-cp', '--checkpoint')
    args = parser.parse_args()

    print('Saving path:', SAVE_PATH)
    checkpoint_mng = CheckpointManager(SAVE_PATH)

    checkpoint = None
    if args.checkpoint:
        print('Load checkpoint:', args.checkpoint)
        checkpoint = checkpoint_mng.load(args.checkpoint, device)

    model, voc = build_model(checkpoint)

    if args.mode == 'train':
        train(args.mode, model, voc, checkpoint, checkpoint_mng)

    elif args.mode == 'run':
        run(model, voc)


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--checkpoint')
    args = parser.parse_args()

    checkpoint_mng = CheckpointManager(SAVE_PATH)
    checkpoint = None if not args.checkpoint else checkpoint_mng.load(args.checkpoint, device)

    model, voc = build_model(checkpoint)
    # Set dropout layers to eval mode
    model.eval()
    # Initialize search module
    if config.BEAM_SEARCH_ON:
        searcher = BeamSearchDecoder(model)
    else:
        searcher = GreedySearchDecoder(model)
    return searcher, voc


if __name__ == '__main__':
    main()
