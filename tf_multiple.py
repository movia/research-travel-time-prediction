import tensorflow as tf
import functools
import numpy as np
import pandas as pd
#from rmv_seas import remove_trend
import datetime

# prevent tensorflow from allocating the entire GPU memory at once
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

class Config_LSTM_1:

    def __init__(self):

        self.batch_size = 120
        self.seq_len = 20
        self.state_size = 120
        self.num_layers = 2
        self.dropout_train = 0.25
        self.dropout_eval = 1


class LSTM_Model_1:

    def __init__(self, config):

        self.config = config
        self.add_placeholders()
        self.get_input_placeholder
        self.get_dropout_placeholder
        self.add_LSTM_layer

    def add_placeholders(self):

        with tf.variable_scope("lstm_placeholders_model_1"):

            self.input_placeholder = tf.placeholder(
                tf.float32, [None, self.config.seq_len, 1])
            self.dropout_placeholder = tf.placeholder(tf.float32)

    def get_input_placeholder(self):
        return self.input_placeholder

    def get_dropout_placeholder(self):
        return self.dropout_placeholder

    def add_LSTM_layer(self):

        with tf.variable_scope("lstm_layer_model_1"):

            onecell = tf.contrib.rnn.GRUCell(self.config.state_size)
            onecell = tf.contrib.rnn.DropoutWrapper(
                onecell, output_keep_prob=self.dropout_placeholder)
            multicell = tf.contrib.rnn.MultiRNNCell(
                [onecell] * self.config.num_layers, state_is_tuple=False)
            outputs, _ = tf.nn.dynamic_rnn(
                multicell, self.input_placeholder, dtype=tf.float32)
            return outputs

######################################################


config_lstm_1 = Config_LSTM_1()
model_1 = LSTM_Model_1(config_lstm_1)

df = pd.read_csv("C:/Users/ncp/Downloads/yellow_tripdata_2016-01.csv")

data = pd.read_csv('data/4A_201701_Consistent.csv', sep=';')

start_date = datetime.datetime.strptime("2017-01-01 00:00", "%Y-%m-%d %H:%M")
end_date = datetime.datetime.strptime("2017-02-01 00:00", "%Y-%m-%d %H:%M")
date_index = pd.date_range(start_date, end_date, freq='30min').values
len(date_index)

data.iloc[1, :]