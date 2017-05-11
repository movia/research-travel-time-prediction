import numpy as np
import pandas as pd
from sklearn import svm
import sklearn.preprocessing as pp
from common import load_csv, write_results_table

# Configuration
group_columns = ['LinkRef']
categorial_columns = ['DayType', 'TimeOfDayClass']
meta_columns = ['JourneyLinkRef', 'JourneyRef', 'DateTime', 'DayType', 'LineDirectionLinkOrder', 'LinkName']

results = pd.DataFrame()

# Load and pre-process data
print('Loading data ...')
for group, X, Y, meta in load_csv('data/4A_201701_Consistent.csv', group_columns = group_columns, categorial_columns = categorial_columns, meta_columns = meta_columns):

    # Split data into train and test    
    X_train, X_test = np.split(X, [int(.8*len(X))])
    Y_train, Y_test = np.split(Y, [int(.8*len(Y))])
    meta_train, meta_test = np.split(meta, [int(.8*len(meta))])

    # Train

    print('Train data set (size, features):',  X_train.shape)

    # Normalizing X and y:
    X_scaler = pp.RobustScaler(with_centering = False, quantile_range = (5, 95)).fit(X_train)
    Y_scaler = pp.RobustScaler(with_centering = False, quantile_range = (5, 95)).fit(Y_train)

    X_train_norm = X_scaler.transform(X_train)
    Y_train_norm = Y_scaler.transform(Y_train)

    clf = svm.SVR(C = 100.0, gamma = 0.01)
    clf.fit(X_train_norm, Y_train_norm[:,0]) 

    # Test

    print('Test data set (size, features):',  X_test.shape)

    X_test_norm = X_scaler.transform(X_test)
    Y_test_norm = Y_scaler.transform(Y_test)

    Y_test_pred_norm = clf.predict(X_test_norm)
    Y_test_pred = Y_scaler.inverse_transform(Y_test_pred_norm.reshape(-1, 1))

    meta_test['Observed'] = Y_test
    meta_test['Predicted'] = Y_test_pred
    results = results.append(meta_test, ignore_index = True)

# Write predictions to CSV
results.to_csv('data/results_svr_multiple.csv', index = False, encoding = 'utf-8')
# Write predictions to TEX
write_results_table(results, 'paper/results_svr_multiple.tex', group_columns = ['LineDirectionLinkOrder', 'LinkName'], key_index = 1, true_colomn_name = 'Observed', predicted_column_name = 'Predicted')
