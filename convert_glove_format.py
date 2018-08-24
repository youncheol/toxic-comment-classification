from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
import argparse


def convert_glove_format(source_fname, target_fname):
    glove_file = datapath(source_fname)
    tmp_file = get_tmpfile('glove_model.txt')
    glove2word2vec(glove_file, tmp_file)
    glove_model = KeyedVectors.load_word2vec_format(tmp_file)
    glove_model.save(target_fname)

    return glove_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_fname", required=True, help="GloVe 포맷 파일 이름")
    parser.add_argument("--target_fname", required=True, help="Word2Vec 포맷 파일 이름")
    args = parser.parse_args()

    convert_glove_format(args.source_fname, args.target_fname)

