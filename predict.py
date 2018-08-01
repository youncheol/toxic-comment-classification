import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import tensorflow as tf
import numpy as np
import gensim
import argparse
from data_processor import TFRecord, make_submission
from model import CRAN


def predict(glove_fname, model_fname, data_fname, proba=True):
    glove_model = gensim.models.KeyedVectors.load(glove_fname)

    with tf.device("/gpu:0"):
        model = CRAN(glove_model, use_bn=False, dropout_prob=0.5)

        tfrecord = TFRecord()
        tfrecord.make_iterator(data_fname, len(glove_model.vocab) + 1, training=False)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove_fname", required=True)
    parser.add_argument("--model_fname", required=True)
    parser.add_argument("--data_fname", required=True)
    parser.add_argument("--sample_fname", required=True)
    parser.add_argument("--sub_fname", required=True)
    args = parser.parse_args()

    pred = predict(args.glove_fname, args.model_fname, args.data_fname, proba=True)

    make_submission(pred, args.sample_fname, args.sub_fname)


if __name__ == "__main__":
    main()

