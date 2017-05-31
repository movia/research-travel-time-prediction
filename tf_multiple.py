import logging
import datetime

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.contrib.rnn as rnn

import sklearn.preprocessing as pp

import functools

from common import *

# initialize and configure logging
logger = logging.getLogger('tf_multiple')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('tf_multiple.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# prevent tensorflow from allocating the entire GPU memory at once
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

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
        self.batch_size = 64
        self.seq_len = 20
        self.learning_rate = 0.0003
        self.state_size = 128
        self.num_layers = 2
        self.num_epochs = 1
        self.dropout_train = 0.25
        self.dropout_eval = 1

class LstmModel:

    def __init__(self, config, sess):

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
            batch_X = X[i:i+self.config.batch_size, :].reshape(-1, self.config.seq_len, 1)
            batch_y = y[i:i+self.config.batch_size].reshape(-1, 1, 1)
            yield batch_X, batch_y

    def load(self):
        self.saver.restore(self.sess, "models/" + self.config.name + ".ckpt")
        logger.info("%s: Model loaded." % self.config.name)
        
    def train(self, X, y):
                     
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(self.config.num_epochs):
            # mini batch generator for training
            gen_train = self.batch_train_generator(X, y)

            for batch in range(len(X) // self.config.batch_size):
                #logger.debug("Optimizing (epoch, batch) = ({0}, {1})".format(epoch, batch));
                batch_X, batch_y = next(gen_train);
                _ = self.sess.run(self.optimize, feed_dict={
                        self.input_placeholder: batch_X,
                        self.target_placeholder: batch_y,
                        self.dropout_placeholder: self.config.dropout_train
                })

            train_error = self.sess.run(self.cost, feed_dict={
                    self.input_placeholder: X.reshape(-1, self.config.seq_len, 1),
                    self.target_placeholder: y.reshape(-1, 1, 1),
                    self.dropout_placeholder: self.config.dropout_eval
            })

            logger.info("%s: epoch: %d, train error: %f", self.config.name, epoch, train_error)

        # Save the variables to disk.
        save_path = self.saver.save(sess, "models/" + self.config.name + ".ckpt")
        logger.info("%s: Model saved in file: %s", self.config.name, save_path)

    def predict(self, X):

        if self.model is None:
            raise RuntimeError("Model is not initialized.")
       
        preds = self.sess.run(self.model, feed_dict={
                self.input_placeholder: X.reshape(-1, self.config.seq_len, 1),
                self.dropout_placeholder: self.config.dropout_eval
        })

        return preds

def main():    
    logger.info("Using TensorFlow " + tf.VERSION)

    logger.info("Loading data ...")
    data = pd.read_csv('data/4A_201701_Consistent.csv', sep=';')
    # Initial data-slicing
    data = data[(data.LinkTravelTime > 0) & (data.LineDirectionCode == 1)]
    data = data[(26 <= data.LineDirectionLinkOrder) & (data.LineDirectionLinkOrder <= 32)]
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data.set_index(pd.DatetimeIndex(data['DateTime']), inplace = True)

    logger.info("Transforming data ...")
    # Create and 2d matrix of traveltime with (x, y) = (space, time) = (linkRef, journeyRef)
    ts = data.pivot(index='JourneyRef', columns='LinkRef', values='LinkTravelTime')
    ts = ts[~np.isnan(ts).any(axis=1)]
    
    # TODO: Refactor 
    i = int(len(ts) * 0.8)
    n_test = len(ts) - i

    train = ts[0:i]
    test = ts[i:i + n_test]
       
    scaler = pp.StandardScaler()
    train_norm = scaler.fit_transform(train)
    test_norm = scaler.transform(test)

    # Create lags travel time
    X_train_norm = np.stack([np.roll(train_norm, i) for i in range(20, 0, -1)], axis = -1)
    X_train_norm = X_train_norm[20:, ...]
    y_train_norm = train_norm[20:, ...]

    # Create lags travel time
    X_test_norm = np.stack([np.roll(test_norm, i) for i in range(20, 0, -1)], axis = -1)
    X_test_norm = X_test_norm[20:, ...]
    y_test_norm = test_norm[20:, ...]
    y_test = test.iloc[20:, :]

    logger.info("Train size (X, y) = (" + str(X_train_norm.shape) + ", " + str(y_train_norm.shape) + ")")
    logger.info("Test size (X, y) = (" + str(X_test_norm.shape) + ", " + str(y_test_norm.shape) + ")")

    logger.info("Initializing model graph ...")
    tf.reset_default_graph()

    submodels = []
    preds_norm = []

    with tf.Session() as sess:

        # Create submodel for each space (link) and train and evel independently
        for space in range(y_train_norm.shape[1]):
            config = LstmConfig("lstm_model_" + str(space))
            submodels.append(LstmModel(config, sess));

        if True:
            logger.info("Loading models ...")
            for space in range(X_train_norm.shape[1]):
                submodels[space].load()
        else:
            logger.info("Running training epochs ...")
            for space in range(X_train_norm.shape[1]):
                submodels[space].train(X_train_norm[:,space,:], y_train_norm[:,space])

        for space in range(X_test_norm.shape[1]):
            preds_norm.append(submodels[space].predict(X_test_norm[:,space,:]))

    preds_norm = np.stack(preds_norm, axis = 1).reshape(-1, X_test_norm.shape[1])    
    preds = scaler.inverse_transform(preds_norm)
   
    for space in range(y_test_norm.shape[1]):
        logger.info("Results: %s (MAPE) = (%f)", test.columns[space], mean_absolute_percentage_error(y_test.iloc[:, space], preds[:, space]))

    results = pd.DataFrame(data=preds, index = test.index[20:], columns = test.columns)
    results.to_csv('data/results_lstm_independent.csv', index = True, encoding = 'utf-8')

if __name__ == "__main__": 
    try:
        main()
    except Exception as e:
        logging.exception("message")
