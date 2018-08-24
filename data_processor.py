import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import tensorflow as tf
import gensim
import pickle
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


class PreProcessing:
    def __init__(self):
        self.contraction_map = contraction_map
        self.contraction_object = re.compile("|".join(contraction_map.keys()))
        self.sub_patterns = '|'.join(unnecessary_patterns)
        self.stop_words = stopwords.words('english')
        self.wnl = WordNetLemmatizer()
        self.word_tokenize = word_tokenize

        self.model = None
        self.tokenized_data = None
        self.indexed_data = None

        self.use_file = None

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
        output_list = []
        oov_index = len(self.model.vocab) + 1
        for sentence in tqdm(data, desc="Converting word to index"):
            word_list = []
            for i, word in enumerate(sentence):
                if i > (max_length - 1):
                    continue
                else:
                    try:
                        word_list.append(self.model.vocab[word].index + 1)
                    except KeyError:
                        word_list.append(oov_index)

            output_list.append(word_list)

        return output_list

    def processing(self, data, glove_fname, max_length=100):
        if not self.use_file:
            self.tokenized_data = self._tokenizing(data)
            # self._load_glove(glove_fname, glove_savepath)

        self.model = gensim.models.KeyedVectors.load(glove_fname)

        self.indexed_data = self._word2index(self.tokenized_data, max_length)

    def load_save(self, fname, mode='load', obj=None):
        if mode == 'load':
            with open(fname, 'rb') as f:
                self.tokenized_data = pickle.load(f)

            self.use_file = True

        elif mode == 'save':
            with open(fname, 'wb') as f:
                pickle.dump(obj, f)


class TFRecord:
    def __init__(self):
        self.comment = None
        self.label = None
        self.init_op = None

    def _make_example(self, comment, label=None, training=True):
        se = tf.train.SequenceExample()
        comment_feature = se.feature_lists.feature_list["comment"]
        for token in comment:
            comment_feature.feature.add().int64_list.value.append(token)
        if training:
            label_feature = se.feature_lists.feature_list["label"]
            for binary in label:
                label_feature.feature.add().int64_list.value.append(binary)

        return se

    def save(self, fname, comments, labels=None, training=True):
        writer = tf.python_io.TFRecordWriter(fname)

        if training:
            for comment, label in tqdm(zip(comments, labels), total=len(labels), desc="Writing TFRecord"):
                over_sample = 1

                if label.sum() >=1:
                    over_sample = 2

                for _ in range(over_sample):
                    example = self._make_example(comment=comment, label=label, training=training)
                    writer.write(example.SerializeToString())
        else:
            for comment in tqdm(comments, desc="Writing TFRecord"):
                example = self._make_example(comment=comment, training=training)
                writer.write(example.SerializeToString())

        writer.close()

    def _train_parser(self, example):
        features = {
            "comment": tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64),
            "label": tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64)
        }

        _, parsed_features = tf.parse_single_sequence_example(
            serialized=example,
            sequence_features=features
        )

        comment = parsed_features["comment"]
        label = parsed_features["label"]

        return comment, label

    def _test_parser(self, example):
        features = {
            "comment": tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64)
        }

        _, parsed_features = tf.parse_single_sequence_example(
            serialized=example,
            sequence_features=features
        )

        comment = parsed_features["comment"]

        return comment

    def make_iterator(self, fname, padding_value, training=True, shuffle_size=100000, batch_size=64):
        with tf.name_scope("TFRecord"):
            if training:
                padded_shapes = (tf.Dimension(None), tf.Dimension(None))
                padding_values = (tf.constant(padding_value, dtype=tf.int64),
                                  tf.constant(padding_value, dtype=tf.int64))
                data = tf.data.TFRecordDataset(fname).map(self._train_parser)
                data = data.shuffle(shuffle_size, reshuffle_each_iteration=True)
            else:
                padded_shapes = (tf.Dimension(None))
                padding_values = (tf.constant(padding_value, dtype=tf.int64))
                data = tf.data.TFRecordDataset(fname).map(self._test_parser)

            data = data.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
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


def make_submission(predict, sample_fname, sub_fname):
    sample = pd.read_csv(sample_fname)
    index = sample["id"]
    columns = sample.columns.tolist()[1:]
    sub = pd.DataFrame(predict, index=index, columns=columns)
    sub.to_csv(sub_fname, index=True)
    print("Submission file is created.")
