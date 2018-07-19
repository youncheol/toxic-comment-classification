import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import gensim
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from dictionary import contraction_map, unnecessary_patterns
from tqdm import tqdm
from collections import defaultdict
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import argparse


class PreProcessing():
    def __init__(self, train_fname, test_fname, glove_fname, model_fname, max_length=100):
        self.train_fname = train_fname
        self.test_fname = test_fname
        self.train = None
        self.test = None
        self.train_y = None
        self.contraction_map = contraction_map
        self.contraction_object = re.compile("|".join(contraction_map.keys()))
        self.sub_patterns = '|'.join(unnecessary_patterns)
        self.stop_words = stopwords.words('english')
        self.wnl = WordNetLemmatizer()
        self.word_tokenize = word_tokenize
        self.word_count = defaultdict(lambda: 0)
        # self.embedding_dim = embedding_dim
        self.glove_fname = glove_fname
        self.model_fname = model_fname
        self.max_length = max_length

        self.use_file = None

        self.model = None
        self.tokenized_train = None
        self.tokenized_test = None
        self.indexed_train = None
        self.indexed_test = None

    def _load_data(self):
        self.train = pd.read_csv(self.train_fname)
        self.test = pd.read_csv(self.test_fname)
        self.train_y = self.train.as_matrix()[:, 2:].astype("int32")

    def _expand_contraction(self, sentence):
        def matching_case(match):
            return self.contraction_map[match.group(0)]

        return self.contraction_object.sub(matching_case, sentence)

    def _cleaning_text(self, text):
        text = re.sub(self.sub_patterns, ' ', text)
        text = re.sub('[0-9]+', 'NUM', text)
        return text.strip()

    def _tokenizer(self, sentence):
        tokenized_sentence = self.word_tokenize(sentence)

        return [self.wnl.lemmatize(token) for token in tokenized_sentence if token not in self.stop_words]

    def _tokenizing(self, dataset):
        comment_list = []
        total = len(dataset.comment_text)
        for i, comment in enumerate(tqdm(dataset.comment_text, desc="Tokeninzing")):
            comment = comment.lower()
            tokenized_comment = self._tokenizer(self._cleaning_text(self._expand_contraction(comment)))
            for token in tokenized_comment:
                self.word_count[token] += 1
            comment_list.append(tokenized_comment)

        #     if (i > 0) and (i % 100 == 0):
        #         sys.stdout.write('\rtokenizing...{}%'.format(int((i / total) * 100)))
        #
        # print("\rComplete")
        return comment_list

    def _load_glove(self):
        glove_file = datapath(self.glove_fname)
        tmp_file = get_tmpfile("glove_model.txt")
        glove2word2vec(glove_file, tmp_file)
        glove_model = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)
        self.model = glove_model
        glove_model.save(self.model_fname)

        print("GloVe model is saved.")

    # def _train_embedding(self, data, fname):
    #     model = gensim.models.Word2Vec(size=self.embedding_dim, window=5, min_count=5, sg=1)
    #     model.build_vocab(data)
    #     model.train(data, total_examples=model.corpus_count, epochs=5)
    #     model.save(fname)
    #
    #     print("Embedding mode is saved.")
    #
    #     return model

    def _word2index(self, data):
        size = len(data)
        matrix = np.zeros((size, self.max_length), dtype="int32")
        oov_index = len(self.model.vocab) + 1
        for i, sent in enumerate(tqdm(data)):
            for j, word in enumerate(sent):
                if j < self.max_length:
                    try:
                        matrix[i][j] = self.model.vocab[word].index + 1
                    except KeyError:
                        matrix[i][j] = oov_index
                else:
                    continue

        return matrix


    # def _word2index(self, data, model, max_length):
    #     output_list = []
    #     oov_index = model.wv.vectors.shape[0]
    #     for sentence in data:
    #         word_list = []
    #         for i, word in enumerate(sentence):
    #             if i > (max_length - 1):
    #                 continue
    #             else:
    #                 try:
    #                     word_list.append(model.wv.vocab[word].index)
    #                 except KeyError:
    #                     word_list.append(oov_index)
    #         output_list.append(word_list)
    #
    #     return output_list

    # def _train_valid_split(self, X, y, valid_ratio):
    #     toxic_idx = y.sum(axis=1) > 0
    #
    #     toxic_train_X = [x for i, x in zip(toxic_idx, X) if i]
    #     non_toxic_train_X = [x for i, x in zip(toxic_idx, X) if not i]
    #
    #     toxic_train_y = y[toxic_idx]
    #     non_toxic_train_y = y[toxic_idx == False]
    #
    #     toxic_valid_idx = np.random.choice(a=[True, False], size=len(toxic_train_X), p=[valid_ratio, 1 - valid_ratio])
    #     non_toxic_valid_idx = np.random.choice(a=[True, False], size=len(non_toxic_train_X),
    #                                            p=[valid_ratio, 1 - valid_ratio])
    #
    #     train_X0 = [x for i, x in zip(toxic_valid_idx, toxic_train_X) if not i]
    #     train_X1 = [x for i, x in zip(non_toxic_valid_idx, non_toxic_train_X) if not i]
    #     valid_X0 = [x for i, x in zip(toxic_valid_idx, toxic_train_X) if i]
    #     valid_X1 = [x for i, x in zip(non_toxic_valid_idx, non_toxic_train_X) if i]
    #
    #     train_y0 = toxic_train_y[toxic_valid_idx == False]
    #     train_y1 = non_toxic_train_y[non_toxic_valid_idx == False]
    #     valid_y0 = toxic_train_y[toxic_valid_idx]
    #     valid_y1 = non_toxic_train_y[non_toxic_valid_idx]
    #
    #     train_X = train_X0 + train_X1
    #     valid_X = valid_X0 + valid_X1
    #     train_y = np.vstack([train_y0, train_y1])
    #     valid_y = np.vstack([valid_y0, valid_y1])
    #
    #     return train_X, valid_X, train_y, valid_y

    def processing(self):
        self._load_data()

        if not self.use_file:
            self.tokenized_train = self._tokenizing(self.train)
            self.tokenized_test = self._tokenizing(self.test)

        # tokenized_data = self.tokenized_train + self.tokenized_test
        # print("Start training vector model...")
        # self.model = self._train_embedding(tokenized_data, model_fname)

        print("Loading GloVe model.")
        self._load_glove()

        print("Converting word to index.")
        self.indexed_train = self._word2index(self.tokenized_train)
        self.indexed_test = self._word2index(self.tokenized_test)

        # print("Train valid split")
        # train_X, valid_X, train_y, valid_y = self._train_valid_split(self.indexed_train, y_train, self.valid_ratio)

        print("Finished.")

        # return train_X, valid_X, train_y, valid_y, self.indexed_test

    def load_save(self, train_fname, test_fname, mode='load', train_obj=None, test_obj=None):
        if mode == 'load':
            with open(train_fname, 'rb') as f:
                self.tokenized_train = pickle.load(f)
            with open(test_fname, 'rb') as f:
                self.tokenized_test = pickle.load(f)

            self.use_file = True

        elif mode == 'save':
            with open(train_fname, 'wb') as f:
                pickle.dump(train_obj, f)
            with open(test_fname, 'wb') as f:
                pickle.dump(test_obj, f)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fname", required=True)
    parser.add_argument("--test_fname", required=True)
    parser.add_argument("--glove_fname", required=True)
    parser.add_argument("--glove_savepath", required=True)
    parser.add_argument("--max_length", type=int, required=True)
    args = parser.parse_args()

    prep = PreProcessing(args.train_fname, args.test_fname, args.glove_fname,
                         args.glove_savepath, args.max_length)

    prep.load_save("tokenized_train.pkl", "tokenized_test.pkl", mode="load")

    prep.processing()

    print(prep.indexed_train.shape)
    print(prep.indexed_test.shape)
    print(prep.indexed_train[0])


if __name__ == "__main__":
    main()