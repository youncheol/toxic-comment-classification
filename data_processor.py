import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
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
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import argparse


class PreProcessing:
    def __init__(self):
        self.contraction_map = contraction_map
        self.contraction_object = re.compile("|".join(contraction_map.keys()))
        self.sub_patterns = '|'.join(unnecessary_patterns)
        self.stop_words = stopwords.words('english')
        self.wnl = WordNetLemmatizer()
        self.word_tokenize = word_tokenize

        self.train = None
        self.test = None
        self.train_y = None

        self.model = None

        self.tokenized_train = None
        self.tokenized_test = None
        self.indexed_train = None
        self.indexed_test = None

        self.use_file = None

    def _load_data(self, train_fname, test_fname):
        self.train = pd.read_csv(train_fname)
        self.test = pd.read_csv(test_fname)
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
        for i, comment in enumerate(tqdm(dataset.comment_text, desc="Tokeninzing")):
            comment = comment.lower()
            tokenized_comment = self._tokenizer(self._cleaning_text(self._expand_contraction(comment)))
            comment_list.append(tokenized_comment)

        return comment_list

    def _load_glove(self, glove_fname, glove_savepath):
        glove_file = datapath(glove_fname)
        tmp_file = get_tmpfile("glove_model.txt")
        glove2word2vec(glove_file, tmp_file)
        glove_model = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)
        self.model = glove_model
        glove_model.save(glove_savepath)

    def _word2index(self, data, max_length):
        size = len(data)
        matrix = np.zeros((size, max_length), dtype="int32")
        oov_index = len(self.model.vocab) + 1
        for i, sent in enumerate(tqdm(data)):
            for j, word in enumerate(sent):
                if j < max_length:
                    try:
                        matrix[i][j] = self.model.vocab[word].index + 1
                    except KeyError:
                        matrix[i][j] = oov_index
                else:
                    continue

        return matrix

    def processing(self, train_fname, test_fname, glove_fname, glove_savepath, max_length):
        self._load_data(train_fname, test_fname)

        if not self.use_file:
            self.tokenized_train = self._tokenizing(self.train)
            self.tokenized_test = self._tokenizing(self.test)

        print("Loading GloVe model.")
        self._load_glove(glove_fname, glove_savepath)

        print("Converting word to index.")
        self.indexed_train = self._word2index(self.tokenized_train, max_length)
        self.indexed_test = self._word2index(self.tokenized_test, max_length)

        print("Finished.")

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


class TFRecord:
    def __init__(self, max_length=50, num_classes=6):
        self.max_length = max_length
        self.num_classes = num_classes

        self.comment = None
        self.label = None
        self.init_op = None

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _make_example(self, training, comment, label=None):
        if training:
            feature  = {
                "comment": self._int64_feature(comment),
                "label": self._int64_feature(label)
            }

        else:
            feature = {
                "comment": self._int64_feature(comment)
            }

        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)

        return example
    #
    # def _make_example(self, comment, label=None, training=True):
    #     se = tf.train.SequenceExample()
    #     comment_feature = se.feature_lists.feature_list["comment"]
    #     for token in comment:
    #         comment_feature.feature.add().int64_list.value.append(token)
    #     if training:
    #         label_feature = se.feature_lists.feature_list["label"]
    #         for binary in label:
    #             label_feature.feature.add().int64_list.value.append(binary)
    #
    #     return se

    def save(self, fname, comments, labels=None, training=True):
        writer = tf.python_io.TFRecordWriter(fname)

        if training:
            for comment, label in tqdm(zip(comments, labels)):
                example = self._make_example(comment=comment, label=label, training=training)
                writer.write(example.SerializeToString())
        else:
            for comment in tqdm(comments):
                example = self._make_example(comment=comment, training=training)
                writer.write(example.SerializeToString())

        writer.close()

    def _train_parser(self, example):
        features = {
            "comment": tf.FixedLenFeature(shape=[self.max_length], dtype=tf.int64),
            "label": tf.FixedLenFeature(shape=[self.num_classes], dtype=tf.int64)
        }

        parsed_features = tf.parse_single_example(example, features)

        comment = parsed_features["comment"]
        label = parsed_features["label"]

        return comment, label

    def _test_parser(self, example):
        features = {
            "comment": tf.FixedLenFeature(shape=[self.max_length], dtype=tf.int64)
        }

        parsed_features = tf.parse_single_example(example, features)

        comment = parsed_features["comment"]

        return comment

    def make_iterator(self, fname, training=True, shuffle_size=140000, batch_size=64):
        with tf.name_scope("TFRecord"):
            if training:
                data = tf.data.TFRecordDataset(fname).map(self._train_parser)
                data = data.shuffle(shuffle_size, reshuffle_each_iteration=True)
            else:
                data = tf.data.TFRecordDataset(fname).map(self._test_parser)

            data = data.batch(batch_size)
            iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)

            if training:
                self.comment, self.label = iterator.get_next()
            else:
                self.comment = iterator.get_next()

            self.init_op = iterator.make_initializer(data)

    def load(self, session, training=True):
        if training:
            return session.run([self.comment, self.label])
        else:
            return session.run(self.comment)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_fname", required=True)
    parser.add_argument("--test_fname", required=True)
    parser.add_argument("--glove_fname", required=True)
    parser.add_argument("--glove_savepath", default="glove.model")
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--tfr_train_fname", default="train.tfrecord")
    parser.add_argument("--tfr_test_fname", default="test.tfrecord")

    args = parser.parse_args()

    prep = PreProcessing()

    prep.load_save("tokenized_train.pkl", "tokenized_test.pkl", mode="load")
    prep.processing(args.train_fname, args.test_fname, args.glove_fname, args.glove_savepath, args.max_length)

    tfrecord = TFRecord()

    tfrecord.save(args.tfr_train_fname, prep.indexed_train, prep.train_y, training=True)
    tfrecord.save(args.tfr_test_fname, prep.indexed_test, training=False)

if __name__ == "__main__":
    main()