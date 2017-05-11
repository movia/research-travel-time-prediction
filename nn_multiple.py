import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
from common import *
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation

model = Sequential()
model.add(Dense(250, input_dim = 14))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

# Configuration
group_columns = ['LinkRef']
categorial_columns = ['DayType', 'TimeOfDayClass']
meta_columns = ['JourneyLinkRef', 'JourneyRef', 'DateTime', 'DayType', 'LineDirectionLinkOrder', 'LinkName']

results = pd.DataFrame()

data = load_csv('data/4A_201701_Consistent.csv', group_columns = group_columns, categorial_columns = categorial_columns, meta_columns = meta_columns)
#group, X, Y, meta = next(g)
for group, X, Y, meta in data:
    # Split data into train and test    
    X_train, X_test = np.split(X, [int(.8*len(X))])
    Y_train, Y_test = np.split(Y, [int(.8*len(Y))])
    meta_train, meta_test = np.split(meta, [int(.8*len(meta))])

    print('Train data set (size, features):',  X_train.shape)

    # Normalizing X and y:
    X_scaler = pp.RobustScaler(with_centering = False, quantile_range = (5, 95)).fit(X_train)
    Y_scaler = pp.RobustScaler(with_centering = False, quantile_range = (5, 95)).fit(Y_train)
    #X_scaler = pp.MinMaxScaler(feature_range=(0, 1)).fit(X_train)
    #Y_scaler = pp.MinMaxScaler(feature_range=(0, 1)).fit(Y_train)

    X_train_norm = X_scaler.transform(X_train)
    Y_train_norm = Y_scaler.transform(Y_train)

    model.fit(X_train_norm, Y_train_norm, epochs=25, batch_size=64, verbose=2)

    print('Test data set (size, features):',  X_test.shape)

    X_test_norm = X_scaler.transform(X_test)
    Y_test_norm = Y_scaler.transform(Y_test)

    Y_test_pred_norm = model.predict(X_test_norm)
    Y_test_pred = Y_scaler.inverse_transform(Y_test_pred_norm)

    meta_test['Observed'] = Y_test
    meta_test['Predicted'] = Y_test_pred
    results = results.append(meta_test, ignore_index = True)

# Write predictions to CSV
results.to_csv('data/results_nn_multiple.csv', index = False, encoding = 'utf-8')
# Write predictions to TEX
write_results_table(results, 'paper/results_nn_multiple.tex', group_columns = ['LineDirectionLinkOrder', 'LinkName'], key_index = 1, true_colomn_name = 'Observed', predicted_column_name = 'Predicted')