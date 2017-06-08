import logging

import numpy as np

import tensorflow as tf
import tensorflow.contrib.rnn as rnn

import functools

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

class LstmConfig:

    def __init__(self, name):

        self.name = name
        self.batch_size = 256
        self.seq_len = 20
        self.learning_rate = 0.01
        self.state_size = 130
        self.num_layers = 1
        self.num_epochs = 10
        self.dropout_train = 0.25
        self.dropout_eval = 1

class LstmModel:

    def __init__(self, config, sess):

        self.logger = logging.getLogger()
        self.config = config
        self.add_placeholders()

        last_output = self.add_LSTM_layer()
        last_output = last_output[:, last_output.shape[1] - 1, :]
        last_output = self.add_dense_layer(last_output, self.config.state_size, 1)
        self.model = last_output
        self.optimize  

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()
        self.sess = sess

    def add_placeholders(self):

        with tf.variable_scope(self.config.name + "_placeholders"):

            self.input_placeholder = tf.placeholder(tf.float32, [None, self.config.seq_len, 1], "input")
            self.dropout_placeholder = tf.placeholder(tf.float32, None, "dropout")
            self.target_placeholder = tf.placeholder(tf.float32, [None, 1, 1], "target")

    def add_LSTM_layer(self):

        with tf.variable_scope(self.config.name + "_lstm_layers"):

            # The following is replaced by the generator pattern cf. https://github.com/tensorflow/tensorflow/issues/8191            
            #onecell = rnn.GRUCell(self.config.state_size)
            #onecell = tf.contrib.rnn.DropoutWrapper(onecell, output_keep_prob=self.dropout_placeholder)            
            #multicell = tf.contrib.rnn.MultiRNNCell([onecell] * self.config.num_layers, state_is_tuple=False)

            multicell = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.GRUCell(self.config.state_size), output_keep_prob=self.dropout_placeholder) for _ in range(self.config.num_layers)], state_is_tuple=False)

            outputs, _ = tf.nn.dynamic_rnn(multicell, self.input_placeholder, dtype=tf.float32)
            return outputs
        
    def add_dense_layer(self, _input, hidden_size, out_size):

        weight = tf.Variable(tf.truncated_normal([hidden_size, out_size], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[out_size]))
        return tf.matmul(_input, weight) + bias

    @lazy_property
    def cost(self):
        """Add loss function
        """
        mse = tf.reduce_mean(tf.pow(tf.subtract(self.model, self.target_placeholder), 2.0))
        return mse

    @lazy_property
    def optimize(self):
        """Sets up the training Ops.
        """
        optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate)
        return optimizer.minimize(self.cost)

    def batch_train_generator(self, X, y):
        """Consecutive mini
        batch generator
        """
        for i in range(len(X) // self.config.batch_size):
            batch_X = X[i:i+self.config.batch_size, :]
            batch_y = y[i:i+self.config.batch_size]
            yield batch_X, batch_y

    def load(self):
        self.saver.restore(self.sess, "models/" + self.config.name + ".ckpt")
        self.logger.info("%s: Model loaded." % self.config.name)
        
    def train(self, X, y):
                     
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(self.config.num_epochs):
            # mini batch generator for training
            gen_train = self.batch_train_generator(X, y)

            for batch in range(len(X) // self.config.batch_size):
                #logger.debug("Optimizing (epoch, batch) = ({0}, {1})".format(epoch, batch));
                batch_X, batch_y = next(gen_train);
                _ = self.sess.run(self.optimize, feed_dict={
                        self.input_placeholder: batch_X.reshape(-1, self.config.seq_len, 1),
                        self.target_placeholder: batch_y.reshape(-1, 1, 1),
                        self.dropout_placeholder: self.config.dropout_train
                })

            train_error = self.sess.run(self.cost, feed_dict={
                    self.input_placeholder: X.reshape(-1, self.config.seq_len, 1),
                    self.target_placeholder: y.reshape(-1, 1, 1),
                    self.dropout_placeholder: self.config.dropout_eval
            })

            self.logger.info("%s: epoch: %d, train error: %f", self.config.name, epoch, train_error)

        # Save the variables to disk.
        save_path = self.saver.save(self.sess, "models/" + self.config.name + ".ckpt")
        self.logger.info("%s: Model saved in file: %s", self.config.name, save_path)

    def predict(self, X):

        if self.model is None:
            raise RuntimeError("Model is not initialized.")
       
        preds = self.sess.run(self.model, feed_dict={
                self.input_placeholder: X.reshape(-1, self.config.seq_len, 1),
                self.dropout_placeholder: self.config.dropout_eval
        })

        return preds
