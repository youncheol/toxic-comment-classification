from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_fname", required=True, help="file name of GloVe format vectors")
    parser.add_argument("--target_fname", required=True, help="output file name")

    return parser.parse_args()


def main():
    args = get_args()
    glove_file = datapath(args.source_fname)
    tmp_file = get_tmpfile('glove_model.txt')
    glove2word2vec(glove_file, tmp_file)
    glove_model = KeyedVectors.load_word2vec_format(tmp_file)
    glove_model.save(args.target_fname)


if __name__ == "__main__":
    main()
