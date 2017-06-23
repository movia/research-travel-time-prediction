import numpy as np
from sklearn import linear_model
import sklearn.preprocessing as pp
from common import *

# Configuration
group_columns = []
categorial_columns = ['LinkRef'] # ['LinkRef', 'DayType', 'TimeOfDayClass']
meta_columns = ['JourneyLinkRef', 'JourneyRef', 'DateTime', 'LineDirectionLinkOrder', 'LinkName']

results = pd.DataFrame()

# Load and pre-process data
data = load_csv('data/4A_201701_Consistent.csv',
                group_columns = group_columns, 
                categorial_columns = categorial_columns,
                meta_columns = meta_columns,
                n_lags = 20,
                n_headways = 0)

for group, X, Y, meta in data:

    print('Group:', group)

    # Split data into train and test    
    X_train, X_test = np.split(X, [int(.8*len(X))])
    Y_train, Y_test = np.split(Y, [int(.8*len(Y))])
    meta_train, meta_test = np.split(meta, [int(.8*len(meta))])
    print('\tTrain data set (size, features):',  X_train.shape)

    clf = linear_model.LinearRegression(copy_X = False, n_jobs = -1)
    clf.fit(X_train, Y_train[:,0]) 

    Y_train_pred = clf.predict(X_train).reshape(-1, 1)
    
    # Test
    print('\tTest data set (size, features):',  X_test.shape)

    Y_test_pred = clf.predict(X_test).reshape(-1, 1)

    meta_test['LinkTravelTime_Predicted'] = Y_test_pred
    results = results.append(meta_test, ignore_index = True)

# Write predictions to CSV
results.to_csv('data/results_lr_single.csv', index = False, encoding = 'utf-8')
# Write predictions to TEX
write_results_table(results, 'paper/results_lr_single.tex', group_columns = ['LineDirectionLinkOrder', 'LinkName'], key_index = 1, true_colomn_name = 'LinkTravelTime', predicted_column_name = 'LinkTravelTime_Predicted')
