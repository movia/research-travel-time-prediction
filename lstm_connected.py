import argparse
import logging
import datetime

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.contrib.rnn as rnn

import sklearn.preprocessing as pp

from common import *
from lstm_common import *

def main():   
    # parse arguments
    parser = argparse.ArgumentParser(description='Train LSTM neural network.')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    # initialize and configure logging
    logFormatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler("logs/lstm_connected.log")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(logging.INFO)
    logger.addHandler(consoleHandler)

    # begin
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
    train.iloc[20:, :].to_csv('data/train_lstm.csv', index = True, encoding = 'utf-8')
    test = ts[i:i + n_test]
    test.iloc[20:, :].to_csv('data/test_lstm.csv', index = True, encoding = 'utf-8')
       
    scaler = pp.StandardScaler()
    train_norm = scaler.fit_transform(train)
    test_norm = scaler.transform(test)
    pd.DataFrame(test_norm[20:, :], index = test.index[20:], columns = test.columns).to_csv('data/test_lstm_scaled.csv', index = True, encoding = 'utf-8')

    # Create lags travel time
    X_train_norm = np.stack([np.roll(train_norm, i) for i in range(20, 0, -1)], axis = -1)
    X_train_norm = X_train_norm[20:, ...]
    y_train_norm = train_norm[20:, ...]

    # Create lags travel time
    X_test_norm = np.stack([np.roll(test_norm, i) for i in range(20, 0, -1)], axis = -1)
    X_test_norm = X_test_norm[20:, ...]
    y_test_norm = test_norm[20:, ...]
    y_test = test.iloc[20:, :].as_matrix()

    logger.info("Train size (X, y) = (" + str(X_train_norm.shape) + ", " + str(y_train_norm.shape) + ")")
    logger.info("Test size (X, y) = (" + str(X_test_norm.shape) + ", " + str(y_test_norm.shape) + ")")

    logger.info("Initializing model graph ...")
    tf.reset_default_graph()

    submodels = []
    preds_norm = []

    with tf.Session() as sess:

        # Create submodel for each space (link) and train and evel independently
        for space in range(y_train_norm.shape[1]):
            config = LstmConfig("lstm_connected_" + str(space))
            submodels.append(LstmModel(config, sess));

        connected_config = ConnectedLstmConfig("lstm_connected")
        connected_model = ConnectedLstmModel(connected_config, sess, submodels)

        if args.train:
            logger.info("Running training epochs ...")
            connected_model.train(X_train_norm, y_train_norm)
        else:
            logger.info("Loading models ...")
            connected_model.load()

        preds_norm = connected_model.predict(X_test_norm)
    
    preds = scaler.inverse_transform(preds_norm)
    
    for space in range(y_test_norm.shape[1]):
        logger.info("Results: %s (MAPE / MAE / RMSE) = (%.1f %%, %.1f, %.1f)", 
                    test.columns[space], 
                    mean_absolute_percentage_error(y_test[:, space], preds[:, space]) * 100,
                    mean_absolute_error(y_test[:, space], preds[:, space]),
                    root_mean_square_error(y_test[:, space], preds[:, space]))
    
    logger.info("Results TOTAL: (MAPE / MAE / RMSE) = (%.1f %%, %.1f, %.1f)", 
                mean_absolute_percentage_error(y_test.sum(axis = 1), preds.sum(axis = 1)) * 100,
                mean_absolute_error(y_test.sum(axis = 1), preds.sum(axis = 1)),
                root_mean_square_error(y_test.sum(axis = 1), preds.sum(axis = 1)))
    
    results = pd.DataFrame(data=preds, index = test.index[20:], columns = test.columns)
    results.to_csv('data/results_lstm_connected.csv', index = True, encoding = 'utf-8')

if __name__ == "__main__": 
    try:
        main()
    except Exception as e:
        logging.exception("error in main()")
