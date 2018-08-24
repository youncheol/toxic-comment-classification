import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import tensorflow as tf
import numpy as np
import gensim
import argparse
from data_processor import TFRecord
from model import CRAN


def train(glove_fname, tfr_fname, epochs, logdir, save_fname, filter_size, num_filters, hidden_size,
          learning_rate, dropout_prob):
    glove_model = gensim.models.KeyedVectors.load(glove_fname)

    with tf.device("/gpu:0"):
        model = CRAN(glove_model, filter_size=filter_size, num_filters=num_filters, hidden_size=hidden_size,
                     learning_rate=learning_rate, dropout_prob=dropout_prob)

        tfrecord = TFRecord()
        tfrecord.make_iterator(tfr_fname, len(glove_model.vocab) + 1, shuffle_size=140000)

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        writer = tf.summary.FileWriter(logdir, sess.graph)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for epoch in range(epochs):
            sess.run(tfrecord.init_op)
            loss_list = []

            while True:
                try:
                    step = sess.run(model.global_step)

                    comment, label = tfrecord.load(sess, training=True)
                    _, loss, merged = model.train(sess, comment, label)

                    loss_list.append(loss)

                    writer.add_summary(summary=merged, global_step=step)

                    if (step > 0) and (step % 200 == 0):
                        print("Step {:4} Mean Loss: {}".format(step, np.mean(loss_list)))

                except tf.errors.OutOfRangeError:
                    break

            saver.save(sess, logdir + "/" + save_fname + ".ckpt", global_step=sess.run(model.global_step))
            print("Epoch {} model is saved.".format(epoch + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove_fname", required=True)
    parser.add_argument("--tfr_fname", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--save_fname", required=True)
    parser.add_argument("--filter_size", type=int, default=3)
    parser.add_argument("--num_filters", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout_prob", type=float, default=0.5)
    args = parser.parse_args()

    train(args.glove_fname, args.tfr_fname, args.epochs, args.logdir, args.save_fname,
          args.filter_size, args.num_filters, args.hidden_size, args.learning_rate, args.dropout_prob)


