import numpy as np
from sklearn import tree
import sklearn.preprocessing as pp
from pp import load_csv

# Load and pre-process data
print('Loading data ...')
X, Y = load_csv('data/4A_201701.csv')

# Split data into train and test
X_train, X_test = X[:int(.8*len(X)),:], X[int(.8*len(X)):,:]
Y_train, Y_test = Y[:int(.8*len(X)),:], Y[int(.8*len(X)):,:]
print('Train data set (size, features):',  X_train.shape)

max_leaf_nodes = 50
clf = tree.DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes)
clf.fit(X_train, Y_train[:,0]) 

Y_train_pred = clf.predict(X_train).reshape(-1, 1)
metric_train_mape = (np.abs(Y_train_pred - Y_train)/Y_train).mean()
print('Train MAPE:', metric_train_mape)

# Test
print('Test data set (size, features):',  X_test.shape)

Y_test_pred = clf.predict(X_test).reshape(-1, 1)

metric_test_mape = (np.abs(Y_test_pred - Y_test)/Y_test).mean()
metric_test_mae = np.abs(Y_test_pred - Y_test).mean()
metric_test_rmse = np.sqrt(((Y_test_pred - Y_test) ** 2).mean())
print('MAPE:', metric_test_mape, '\nMAE:', metric_test_mae, '\nRMSE:', metric_test_rmse)

print("$\\text{DT}_{%d}$ & %0.1f\\%% & %0.1f & %0.1f \\\\ \\hline" % (max_leaf_nodes, (metric_test_mape * 100), metric_test_mae, metric_test_rmse))
tree.export_graphviz(clf, 'plots/decision_tree.dot')