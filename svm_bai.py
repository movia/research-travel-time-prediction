import numpy as np
import pandas as pd
from sklearn import svm
import sklearn.preprocessing as pp
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('data/4A_201701_08+12.csv', sep=';')
data = data[(data.LinkTravelTime > 0) & (data.LineDirectionCode == 1)]
data['DateTime'] = pd.to_datetime(data['DateTime'])
time = pd.DatetimeIndex(data['DateTime']) 
data['TimeOfDay'] = time.hour + time.minute / 60
categorial_columns = ['PeekClass', 'LinkRef']
numerical_columns = ['TimeOfDay', 'LinkTravelTime_L']
#numerical_columns = ['LinkTravelTime_L1', 'LinkTravelTime_L2', 'LinkTravelTime_L3']
input_columns = categorial_columns + numerical_columns
output_column = 'LinkTravelTime'

grouping = data.groupby(categorial_columns)

# Equivilent of (15) in Bai, C. et al., 2015. Dynamic bus travel time prediction models on road with multiple bus routes. Computational Intelligence and Neuroscience, 2015.
m = 3
Gamma = np.zeros(len(data))
LinkTravelTime_L = np.zeros(len(data))
for i in range(1,m):
    Gamma += 1/((data['DateTime'] - grouping['DateTime'].shift(i)) / np.timedelta64(1, 's'))
for i in range(1,m):
    LinkTravelTime_L += 1/((data['DateTime'] - grouping['DateTime'].shift(i)) / np.timedelta64(1, 's')) / Gamma * grouping['LinkTravelTime'].shift(i)
data['LinkTravelTime_L'] = LinkTravelTime_L

#data = data[(data.LinkTravelTime_L1 > 0) & (data.LinkTravelTime_L2 > 0) & (data.LinkTravelTime_L3 > 0)]
data = data[(data.LinkTravelTime_L > 0)]

train = data.head(n=int(.8 * len(data)))
test = data.tail(n=int(.2 * len(data)))

X_train = pd.get_dummies(train[input_columns], columns = categorial_columns).as_matrix()
y_train = train.as_matrix(columns = [output_column])
#X_scaler = pp.RobustScaler(with_centering = False, quantile_range = (5, 95)).fit(X_train)
#y_scaler = pp.RobustScaler(with_centering = False, quantile_range = (5, 95)).fit(y_train)
X_scaler = pp.MinMaxScaler().fit(X_train) #.RobustScaler(with_centering = False, quantile_range = (5, 95)).fit(X_train)
y_scaler = pp.MinMaxScaler().fit(y_train) #.RobustScaler(with_centering = False, quantile_range = (5, 95)).fit(y_train)

X_train_norm = X_scaler.transform(X_train)
y_train_norm = y_scaler.transform(y_train)

np.savetxt("foo.csv", np.c_[X_train_norm, y_train_norm], delimiter=";", fmt = "%10.5f")
np.savetxt("bar.csv", np.c_[X_train, y_train], delimiter=";", fmt = "%10.5f")

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = KFold(n_splits=2, random_state=42, shuffle=False)
#cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(svm.SVR(), param_grid=param_grid, n_jobs = 4, verbose = 5)
grid.fit(X_train_norm, y_train_norm[:,0])

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

clf = svm.SVR(C = 0.1, gamma = 4.28575)
clf.fit(X_train_norm, y_train_norm[:,0]) 

y_train_pred_norm = clf.predict(X_train_norm)
y_train_pred = y_scaler.inverse_transform(y_train_pred_norm.reshape(-1, 1))

metric_train_mape = (np.abs(y_train_pred - y_train)/y_train).mean()
print('Train MAPE:', metric_train_mape)

X_test = pd.get_dummies(test[input_columns], columns = categorial_columns).as_matrix()
y_test = test.as_matrix(columns = [output_column])
X_test_norm = X_scaler.transform(X_test)
y_test_norm = y_scaler.transform(y_test)

clf.score(X_test_norm, y_test_norm[:,0])

y_test_pred_norm = clf.predict(X_test_norm)
y_test_pred = y_scaler.inverse_transform(y_test_pred_norm.reshape(-1, 1))

test['LinkTravelTime_Predicate'] = y_test_pred[:,0]
test.to_csv('data/4A_201701_08+12_test.csv', encoding = 'utf-8', index = False)

metric_test_mae = np.abs(y_test_pred - y_test).mean()
print('MAE:', metric_test_mae)

metric_test_mape = (np.abs(y_test_pred - y_test)/y_test).mean()
print('MAPE:', metric_test_mape)

metric_test_rmse = np.sqrt(((y_test_pred - y_test) ** 2).mean())
print('RMSE:', metric_test_rmse)