import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import tensorflow as tf
import numpy as np
import gensim
import argparse
from data_processor import TFRecord
from model import CRAN


def train(glove_fname, data_fname, epochs, log_dir, model_name, filter_size):
    glove_model = gensim.models.KeyedVectors.load(glove_fname)

    with tf.device("/gpu:0"):
        model = CRAN(glove_model, use_bn=False, dropout_prob=0.5, filter_size=filter_size)

        tfrecord = TFRecord()
        tfrecord.make_iterator(data_fname, len(glove_model.vocab) + 1, shuffle_size=140000)

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        writer = tf.summary.FileWriter(log_dir, sess.graph)

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

            saver.save(sess, log_dir + "/" + model_name + ".ckpt", global_step=sess.run(model.global_step))
            print("Epoch {} model is saved.".format(epoch + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove_fname", required=True)
    parser.add_argument("--tfr_fname", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--filter_size", type=int, default=3)
    args = parser.parse_args()

    train(args.glove_fname, args.tfr_fname, args.epochs, args.log_dir, args.model_name, args.filter_size)


if __name__ == "__main__":
    main()

