import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import functools
import numpy as np
import pandas as pd

import datetime

print("Using TensorFlow " + tf.VERSION)

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

class Config_LSTM_1:

    def __init__(self):

        self.batch_size = 100
        self.seq_len = 5
        self.learning_rate = 0.001
        self.state_size = 120
        self.num_layers = 2
        self.dropout_train = 0.25
        self.dropout_eval = 1

class LSTM_Model_1:

    def __init__(self, config):

        self.config = config
        self.add_placeholders()
        #self.get_input_placeholder
        #self.get_dropout_placeholder
        last_output = self.add_LSTM_layer()
        last_output = last_output[:, last_output.shape[1] - 1, :]
        last_output = self.add_dense_layer(last_output, self.config.state_size, 1)
        self.model = last_output

    def add_placeholders(self):

        with tf.variable_scope("lstm_placeholders_model_1"):

            self.input_placeholder = tf.placeholder(tf.float32, [None, self.config.seq_len, 1], "input")
            self.dropout_placeholder = tf.placeholder(tf.float32, None, "dropout")
            self.target_placeholder = tf.placeholder(tf.float32, [None, 1, 1], "target")

    def add_LSTM_layer(self):

        with tf.variable_scope("lstm_layer_model_1"):

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


    def batch_train_generator(self, X, location):
        """Consecutive mini
        batch generator
        """
        for i in range(len(X) // self.config.batch_size):
            batch_X = X[i:i+self.config.batch_size, location, :self.config.seq_len].reshape(self.config.batch_size, self.config.seq_len, 1)
            batch_y = X[i:i+self.config.batch_size, location, -1].reshape(self.config.batch_size, 1, 1)
            yield batch_X, batch_y

    def run_epochs(self, X):
        # TODO: Refactor 
        i = int(len(X) * 0.8)
        n_test = len(X) - i

        X_train = X[:i]
        X_test = X[i:i + n_test]
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # mini batch generator for training
        gen_train = self.batch_train_generator(X_train, 0)

        for batch_X_train, batch_y_train in gen_train:
            _ = sess.run(self.optimize, feed_dict={
                    self.input_placeholder: batch_X_train,
                    self.target_placeholder: batch_y_train,
                    self.dropout_placeholder: self.config.dropout_train
            })




######################################################


print("Loading data ...")
data = pd.read_csv('data/4A_201701_Consistent.csv', sep=';')
# Initial data-slicing
data = data[(data.LinkTravelTime > 0) & (data.LineDirectionCode == 1)]
data = data[(26 <= data.LineDirectionLinkOrder) & (data.LineDirectionLinkOrder <= 32)]
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index(pd.DatetimeIndex(data['DateTime']), inplace = True)


print("Transforming data ...")
travel_time_ts = data.pivot(index='JourneyRef', columns='LinkRef', values='LinkTravelTime')
travel_time_ts = travel_time_ts[~np.isnan(travel_time_ts).any(axis=1)]

# Create lags travel time
lags = np.stack([travel_time_ts.shift(i) for i in range(5, -1, -1)], axis = -1)
lags = lags[5:, ...]

print("Initializing model graph ...")
tf.reset_default_graph()
config_lstm_1 = Config_LSTM_1()
model_1 = LSTM_Model_1(config_lstm_1)

print("Running ...")
model_1.run_epochs(lags)


