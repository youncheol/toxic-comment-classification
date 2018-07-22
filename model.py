import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import tensorflow as tf
# import numpy as np
# import gensim


class CRAN:


    def __init__(self, embedding_matrix, filter_size=3,
                 num_filters=100, hidden_size=100, num_classes=6, learning_rate=0.001,
                 use_bn=False, dropout_prob=None, class_weights=None):

        self.embedding_matrix = embedding_matrix
        # self.use_pretrained = use_pretrained
        self.embedding_dim = embedding_matrix.shape[1]

        self.filter_size = filter_size
        self.num_filters = num_filters
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.use_bn = use_bn
        self.dropout_prob = dropout_prob
        self.class_weights = class_weights

        self.X = tf.placeholder(tf.int64, [None, None], name="comment")
        self.y = tf.placeholder(tf.float32, [None, num_classes], name="label")
        self.training = tf.placeholder_with_default(False, shape=(), name='is_training')

        # self.valid_pred = tf.placeholder(tf.float32, [None, None], name="valid_pred")
        # self.valid_label = tf.placeholder(tf.float32, [None, None], name="valid_label")

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.learning_rate = learning_rate

        self.loss = None
        self.train_op = None
        self.predict_proba = None
        self.auc = None

        # self.valid_auc = None

        self._build_graph()

    def _word_embedding(self, inputs):
        with tf.name_scope("word_embedding"):
            embedding_W = tf.get_variable("embedding_W",
                                          shape=[self.embedding_matrix.shape[0], self.embedding_matrix.shape[1]],
                                          initializer= tf.constant_initializer(self.embedding_matrix),
                                          trainable=True)
            embedded_X = tf.nn.embedding_lookup(embedding_W, inputs)

        return embedded_X

    def _attention_extraction(self, inputs, training):
        with tf.name_scope("attention_extraction"):
            seq_length = tf.shape(inputs)[1]
            input_reshaped = tf.reshape(inputs, [-1, seq_length, self.embedding_dim, 1])
            paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]], dtype="int32")
            input_padded = tf.pad(input_reshaped, paddings, "CONSTANT")

            filter_shape = [self.filter_size, self.embedding_dim, 1, self.num_filters]

            W = tf.get_variable("cnn_filter",
                                shape=filter_shape,
                                initializer=tf.contrib.layers.variance_scaling_initializer())

            conv = tf.nn.conv2d(input=input_padded,
                                filter=W,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="convolution")

            if self.use_bn:
                bn_conv = tf.layers.batch_normalization(conv, momentum=0.9, training=training)
                conv_output = tf.nn.relu(bn_conv, name='relu')

            else:
                b = tf.get_variable("cnn_bias",
                                    shape=[self.num_filters],
                                    initializer=tf.constant_initializer(0.0))
                conv_output = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            attention_signal = tf.reduce_mean(tf.transpose(tf.squeeze(conv_output), perm=[0, 2, 1]),
                                              axis=1,
                                              name="attention_signal")

        return attention_signal

    def _lstm_encoder(self, inputs):
        with tf.name_scope("lstm_encoder"):
            sign = tf.sign(tf.reduce_max(tf.abs(inputs), axis=2))
            length = tf.cast(tf.reduce_sum(sign, axis=1), tf.int32)

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, name="lstm_cell")

            if self.dropout_prob:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_prob)

            hiddens, _ = tf.nn.dynamic_rnn(lstm_cell, inputs, sequence_length=length, dtype=tf.float32)

        return hiddens

    def _classification(self, attention_signal, hiddens):
        with tf.name_scope("classification"):
            whole_seq = tf.reduce_mean(tf.multiply(hiddens, tf.expand_dims(attention_signal, axis=2)), axis=1)

            W = tf.get_variable("output_weights",
                                shape=[self.hidden_size, self.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.get_variable("output_bias",
                                shape=[self.num_classes],
                                initializer=tf.constant_initializer(0.0))

            output = tf.nn.bias_add(tf.matmul(whole_seq, W), b)

        return output

    def _optimization(self, inputs, targets):
        with tf.name_scope("loss"):
            if self.class_weights:
                weights = tf.constant(self.class_weights, dtype=tf.float32)
                weighted_loss = tf.nn.weighted_cross_entropy_with_logits(targets, inputs, pos_weight=weights)
                loss = tf.reduce_mean(weighted_loss)
            else:
                loss = tf.losses.sigmoid_cross_entropy(targets, inputs)
                tf.summary.scalar("loss", loss)

        with tf.name_scope("train"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step=self.global_step)

        with tf.name_scope("predict"):
            predict_proba = tf.nn.sigmoid(inputs)

        with tf.name_scope("AUC"):
            auc = tf.metrics.auc(targets, predict_proba, name="train_auc")
            tf.summary.scalar("AUC", auc[1])

        # with tf.name_scope("validation_AUC"):
        #     valid_auc = tf.metrics.auc(self.valid_label, self.valid_pred, name="valid_auc")

        return loss, train_op, predict_proba, auc

    def _build_graph(self):
        embedded_X = self._word_embedding(self.X)
        attention_signal = self._attention_extraction(embedded_X, self.training)
        hiddens = self._lstm_encoder(embedded_X)
        logits = self._classification(attention_signal, hiddens)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        self.loss, self.train_op, self.predict_proba, self.auc = self._optimization(logits, self.y)

        return logits

    def train(self, session, X, y):
        merged = tf.summary.merge_all()
        return session.run([self.train_op, self.loss, merged], feed_dict={self.X: X, self.y: y, self.training: True})

    # def validation(self, session, target, predict):
    #     return session.run(self.valid_auc, feed_dict={self.valid_label: target, self.valid_pred: predict})

    def predict(self, session, X):
        return session.run(self.predict_proba, feed_dict={self.X: X, self.training: False})



