import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import gensim
import argparse
import logging
from tqdm import tqdm
from data_processor import TFRecord
from model import CRAN


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove_fname", default="glove.model",
                        help="file name of pre-trained GloVe embedding model (default: glove.model)")
    parser.add_argument("--tfr_fname", default="train.tfrecord",
                        help="file name of TFRecord to train (default: train.tfrecord)")
    parser.add_argument("--num_epochs", default=5, type=int,
                        help="number of training epochs (default: 5)")
    parser.add_argument("--logdir", default="logs",
                        help="directory name where to write log files (default: logs)")
    parser.add_argument("--save_fname", default="model",
                        help="prefix of model to be saved (default: model")
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
    parser.add_argument("--class_weight", nargs="+", type=int, default=None,
                        help="class weight of loss function (default: None)")

    return parser.parse_args()


def get_logger(logdir):
    try:
        os.mkdir(logdir)
    except FileExistsError:
        pass

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")

    file_handler = logging.FileHandler(logdir + "/" + "log.txt")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def main():
    args = get_args()
    logger = get_logger(args.logdir)

    logger.info(vars(args))

    glove_model = gensim.models.KeyedVectors.load(args.glove_fname)

    with tf.device("/gpu:0"):
        model = CRAN(glove_model, filter_size=args.filter_size, num_filters=args.num_filters, hidden_size=args.hidden_size,
                     learning_rate=args.learning_rate, dropout_prob=args.dropout_prob, class_weight=args.class_weight)

    tfrecord = TFRecord()
    tfrecord.make_iterator(args.tfr_fname, len(glove_model.vocab) + 1, batch_size=args.batch_size)

    total = sum(1 for _ in tf.python_io.tf_record_iterator(args.tfr_fname)) // args.batch_size

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        writer = tf.summary.FileWriter(args.logdir, sess.graph)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for epoch in range(args.num_epochs):
            logger.info(f"Epoch {epoch+1}")

            sess.run(tfrecord.init_op)
            loss_list = []

            progress_bar = tqdm(total=total, desc="[TRAIN] Loss: 0", unit="batch", leave=False)

            while True:
                try:
                    step = sess.run(model.global_step)

                    comment, label = tfrecord.load(sess, training=True)
                    _, loss, merged = model.train(sess, comment, label)

                    progress_bar.update(1)
                    progress_bar.set_description(f"[TRAIN] Batch Loss: {loss:.4f}")

                    loss_list.append(loss)

                    writer.add_summary(summary=merged, global_step=step)

                except tf.errors.OutOfRangeError:
                    break

            progress_bar.close()

            mean_loss = np.mean(loss_list)
            logger.info(f"  -  [TRAIN] Mean Loss: {mean_loss:.4f}")

            saver.save(sess, args.logdir + "/" + args.save_fname + ".ckpt", global_step=sess.run(model.global_step))


if __name__ == "__main__":
    main()
