import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import pandas as pd
import gensim
import argparse
import logging
from tqdm import tqdm
from data_processor import TFRecord
from model import CRAN


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove_fname", required=True,
                        help="file name of pre-trained GloVe embedding model")
    parser.add_argument("--model_fname", required=True,
                        help="file name of model to restore")
    parser.add_argument("--tfr_fname", required=True,
                        help="file name of TFRecord to predict")
    parser.add_argument("--sample_fname", required=True,
                        help="file name of kaggle sample submission")
    parser.add_argument("--output_fname", required=True,
                        help="file name of submission to be created")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size to load data (default: 64)")
    parser.add_argument("--filter_size", type=int, default=3,
                        help="size of convolution filters (default: 3)")
    parser.add_argument("--num_filters", type=int, default=100,
                        help="number of convolution channels (default: 100)")
    parser.add_argument("--hidden_size", type=int, default=100,
                        help="number of GRU hidden units (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate for optimization (default: 0.001)")
    parser.add_argument("--dropout_prob", type=float, default=0.5,
                        help="dropout probability of RNN layer (default: 0.5)")
    parser.add_argument("--proba", default=False, action="store_true",
                        help="predict probabilities (default: False)")

    return parser.parse_args()


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    return logger


def make_submission(predict, sample_fname, output_fname):
    sample = pd.read_csv(sample_fname)
    index = sample["id"]
    columns = sample.columns.tolist()[1:]
    sub = pd.DataFrame(predict, index=index, columns=columns)
    sub.to_csv(output_fname, index=True)


def main():
    args = get_args()
    logger = get_logger()

    glove_model = gensim.models.KeyedVectors.load(args.glove_fname)

    with tf.device("/gpu:0"):
        model = CRAN(glove_model, filter_size=args.filter_size, num_filters=args.num_filters, hidden_size=args.hidden_size,
                     learning_rate=args.learning_rate, dropout_prob=args.dropout_prob)

    tfrecord = TFRecord()
    tfrecord.make_iterator(args.tfr_fname, len(glove_model.vocab) + 1, training=False)

    total = sum(1 for _ in tf.python_io.tf_record_iterator(args.tfr_fname)) // args.batch_size

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, args.model_fname)

        sess.run(tfrecord.init_op)

        comment = tfrecord.load(sess, training=False)
        predict = model.predict(sess, comment)

        progress_bar = tqdm(total=total, desc="[PREDICT]", unit="batch", leave=False)

        while True:
            try:
                comment = tfrecord.load(sess, training=False)
                predict = np.vstack([predict, model.predict(sess, comment)])

                progress_bar.update(1)

            except tf.errors.OutOfRangeError:
                break

        if not args.proba:
            predict = np.vectorize(lambda x: 1 if x > 0.5 else 0)(predict)

    make_submission(predict, args.sample_fname, args.output_fname)
    logger.info(f"{args.output_fname} is created.")


if __name__ == "__main__":
    main()
