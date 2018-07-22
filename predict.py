import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import tensorflow as tf
# import numpy as np
# import pandas as pd
# import gensim
import argparse
from data_processor import TFRecord, load_embedding_matrix, make_submission
from model import CRAN


def predict(embedding_model_fname, model_fname, data_fname, sample_fname, sub_fname):
    embedding_matrix = load_embedding_matrix(embedding_model_fname)

    with tf.device("/cpu:0"):
        model = CRAN(embedding_matrix, use_bn=True)

        tfrecord = TFRecord()
        tfrecord.make_iterator(data_fname, training=False, batch_size=160000)

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, model_fname)

        sess.run(tfrecord.init_op)

        while True:
            try:
                comment = tfrecord.load(sess, training=False)
                predict = model.predict(sess, comment)

            except tf.errors.OutOfRangeError:
                break

    make_submission(predict, sample_fname, sub_fname)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model_fname", required=True)
    parser.add_argument("--model_fname", required=True)
    parser.add_argument("--data_fname", required=True)
    parser.add_argument("--sample_fname", required=True)
    parser.add_argument("--sub_fname", required=True)
    args = parser.parse_args()

    predict(args.embedding_model_fname, args.model_fname, args.data_fname, args.sample_fname, args.sub_fname)


if __name__ == "__main__":
    main()

