import numpy as np
from sklearn import svm
import sklearn.preprocessing as pp
from sklearn.model_selection import KFold, GridSearchCV
from pp import load_csv
import multiprocessing

X, Y = load_csv('data/4A_201701.csv')
X_train, X_test = X[:int(.8*len(X)),:], X[int(.8*len(X)):,:]
Y_train, Y_test = Y[:int(.8*len(X)),:], Y[int(.8*len(X)):,:]

print('Train data set (size, features):',  X_train.shape)

# Normilizing X and y:
X_scaler = pp.RobustScaler(with_centering = False, quantile_range = (5, 95)).fit(X_train)
Y_scaler = pp.RobustScaler(with_centering = False, quantile_range = (5, 95)).fit(Y_train)

X_train_norm = X_scaler.transform(X_train)
Y_train_norm = Y_scaler.transform(Y_train)

C_range = np.logspace(-2, 3, 6)
gamma_range = np.logspace(-2, 3, 6)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = KFold(n_splits=2, random_state=42, shuffle=False)
grid = GridSearchCV(svm.SVR(cache_size = 32*1024), param_grid=param_grid, n_jobs = multiprocessing.cpu_count(), verbose = 5)
grid.fit(X_train_norm, Y_train_norm[:,0])

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
