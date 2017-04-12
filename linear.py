import numpy as np
from sklearn import linear_model
import sklearn.preprocessing as pp
from pp import load_csv
import multiprocessing

# Load and pre-process data
print('Loading data ...')
X, Y = load_csv('data/4A_201701.csv')

# Split data into train and test
X_train, X_test = X[:int(.8*len(X)),:], X[int(.8*len(X)):,:]
Y_train, Y_test = Y[:int(.8*len(X)),:], Y[int(.8*len(X)):,:]
print('Train data set (size, features):',  X_train.shape)

X_scaler = pp.MinMaxScaler().fit(X_train)
Y_scaler = pp.MinMaxScaler().fit(Y_train)

X_train_norm = X_scaler.transform(X_train)
Y_train_norm = Y_scaler.transform(Y_train)

clf = linear_model.LinearRegression(copy_X = False, n_jobs = multiprocessing.cpu_count())
clf.fit(X_train_norm, Y_train_norm[:,0]) 

Y_train_pred_norm = clf.predict(X_train_norm).reshape(-1, 1)
Y_train_pred = Y_scaler.inverse_transform(Y_train_pred_norm)

metric_train_mape = (np.abs(Y_train_pred - Y_train)/Y_train).mean()
print('Train MAPE:', metric_train_mape)

# Test
print('Test data set (size, features):',  X_test.shape)

X_test_norm = X_scaler.transform(X_test)
Y_test_pred_norm = clf.predict(X_test_norm).reshape(-1, 1)
Y_test_pred = Y_scaler.inverse_transform(Y_test_pred_norm)

metric_test_mape = (np.abs(Y_test_pred - Y_test)/Y_test).mean()
metric_test_mae = np.abs(Y_test_pred - Y_test).mean()
metric_test_rmse = np.sqrt(((Y_test_pred - Y_test) ** 2).mean())
print('MAPE:', metric_test_mape, '\nMAE:', metric_test_mae, '\nRMSE:', metric_test_rmse)

print("$\\text{LR}$ & %0.1f\\%% & %0.1f & %0.1f \\\\ \\hline" % ((metric_test_mape * 100), metric_test_mae, metric_test_rmse))
