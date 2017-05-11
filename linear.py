import numpy as np
from sklearn import linear_model
import sklearn.preprocessing as pp
from pp import load_csv, silentremove
import multiprocessing

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true))

Y_test_all = None
Y_pred_all = None

silentremove('paper/results_lr.tex')

# Load and pre-process data
print('Loading data ...')
for group, X, Y in load_csv('data/4A_201701.csv'):

    # Split data into train and test
    X_train, X_test = X[:int(.8*len(X)),:], X[int(.8*len(X)):,:]
    Y_train, Y_test = Y[:int(.8*len(X)),:], Y[int(.8*len(X)):,:]
    print('Train data set (size, features):',  X_train.shape)

    clf = linear_model.LinearRegression(copy_X = False, n_jobs = multiprocessing.cpu_count())
    clf.fit(X_train, Y_train[:,0]) 

    Y_train_pred = clf.predict(X_train).reshape(-1, 1)
    
    metric_train_mape = (np.abs(Y_train_pred - Y_train)/Y_train).mean()
    print('Train MAPE:', metric_train_mape)

    # Test
    print('Group:', group, '\n\tTest data set (size, features):',  X_test.shape)

    Y_test_pred = clf.predict(X_test).reshape(-1, 1)

    metric_test_mape = (np.abs(Y_test_pred - Y_test)/Y_test).mean()
    metric_test_mae = np.abs(Y_test_pred - Y_test).mean()
    metric_test_rmse = np.sqrt(((Y_test_pred - Y_test) ** 2).mean())

    with open('paper/results_lr.tex', 'a') as file:
        file.write("%s & %0.1f\\%% & %0.1f & %0.1f \\\\ \\hline \n" % (group[1], (metric_test_mape * 100), metric_test_mae, metric_test_rmse))

    print('\tMAPE:', metric_test_mape, '\n\tMAE:', metric_test_mae, '\n\tRMSE:', metric_test_rmse, '\n')

    if not type(Y_test_all) is np.ndarray:
        Y_test_all = Y_test
        Y_pred_all = Y_test_pred
    else:
        Y_test_all = np.append(Y_test_all, Y_test, axis = 0)
        Y_pred_all = np.append(Y_pred_all, Y_test_pred, axis = 0)
    #print("$\\text{LR}$ & %0.1f\\%% & %0.1f & %0.1f \\\\ \\hline" % ((metric_test_mape * 100), metric_test_mae, metric_test_rmse))

metric_test_mape = mean_absolute_percentage_error(Y_test_all, Y_pred_all)
metric_test_mae = np.abs(Y_pred_all - Y_test_all).mean()
metric_test_rmse = np.sqrt(((Y_pred_all - Y_test_all) ** 2).mean())

with open('paper/results_lr.tex', 'a') as file:
    file.write('\\hline \n')
    file.write("%s & %0.1f\\%% & %0.1f & %0.1f \n" % ('Avg.', (metric_test_mape * 100), metric_test_mae, metric_test_rmse))

print('\tMAPE:', metric_test_mape, '\n\tMAE:', metric_test_mae, '\n\tRMSE:', metric_test_rmse, '\n')