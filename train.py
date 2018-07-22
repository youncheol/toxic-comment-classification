import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import tensorflow as tf
import numpy as np
import gensim
import argparse
from data_processor import TFRecord, load_embedding_matrix
from model import CRAN


def train(embedding_model_fname, data_fname, epochs, log_dir, model_name):
    embedding_matrix = load_embedding_matrix(embedding_model_fname)

    with tf.device("/cpu:0"):
        model = CRAN(embedding_matrix, use_bn=True, dropout_prob=0.5)

        tfrecord = TFRecord()
        tfrecord.make_iterator(data_fname, training=True)

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
    parser.add_argument("--embedding_model_fname", required=True)
    parser.add_argument("--tfr_fname", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()

    train(args.embedding_model_fname, args.tfr_fname, args.epochs, args.log_dir, args.model_name)


if __name__ == "__main__":
    main()

