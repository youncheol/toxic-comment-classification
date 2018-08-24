import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import tensorflow as tf
import numpy as np
import gensim
import argparse
from data_processor import TFRecord, make_submission
from model import CRAN


def predict(glove_fname, model_fname, tfr_fname, filter_size, num_filters, hidden_size,
            learning_rate, dropout_prob, proba=True):
    glove_model = gensim.models.KeyedVectors.load(glove_fname)

    with tf.device("/gpu:0"):
        model = CRAN(glove_model, filter_size=filter_size, num_filters=num_filters, hidden_size=hidden_size,
                     learning_rate=learning_rate, dropout_prob=dropout_prob)

        tfrecord = TFRecord()
        tfrecord.make_iterator(tfr_fname, len(glove_model.vocab) + 1, training=False)

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, model_fname)

        sess.run(tfrecord.init_op)

        comment = tfrecord.load(sess, training=False)
        predict = model.predict(sess, comment)

        while True:
            try:
                comment = tfrecord.load(sess, training=False)
                predict = np.vstack([predict, model.predict(sess, comment)])

            except tf.errors.OutOfRangeError:
                break

        if not proba:
            predict = np.vectorize(lambda x: 1 if x > 0.5 else 0)(predict)

    return predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove_fname", required=True)
    parser.add_argument("--model_fname", required=True)
    parser.add_argument("--tfr_fname", required=True)
    parser.add_argument("--sample_fname", required=True)
    parser.add_argument("--sub_fname", required=True)
    parser.add_argument("--filter_size", type=int, default=3)
    parser.add_argument("--num_filters", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout_prob", type=float, default=0.5)
    args = parser.parse_args()

    pred = predict(args.glove_fname, args.model_fname, args.tfr_fname, args.filter_size, args.num_filters,
                   args.hidden_size, args.learning_rate, args.dropout_prob, proba=True)

    make_submission(pred, args.sample_fname, args.sub_fname)

