# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 06:38:33 2022

"""

from pandas import read_csv
from pandas import read_excel
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
from numpy import concatenate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr as psr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values 
	if dropnan:
		agg.dropna(inplace=True)
	return agg

dataset = read_excel('D:/python/New folder/Dataframe.xlsx', header=0, index_col=0)
dataset = dataset.drop(['Exchange_Rate'], axis = 1)
values = dataset.values
# values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 12, 1)
reframed.drop(reframed.columns[[73,74,75,76,77]], axis=1, inplace=True)
values = reframed.values

n_train_month = 132
train = values[:n_train_month, :]
test = values[n_train_month:, :]

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

train_X = train_X.reshape((train_X.shape[0], train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]))

model = LinearRegression()
model.fit(train_X, train_y)

predTrain = model.predict(train_X)

predTest = model.predict(test_X)

inv_predTest = predTest*(max(dataset.iloc[:,0])-min(dataset.iloc[:,0]))+min(dataset.iloc[:,0])
inv_yTest = test_y*(max(dataset.iloc[:,0])-min(dataset.iloc[:,0]))+min(dataset.iloc[:,0])

mape = mean_absolute_percentage_error(inv_yTest, inv_predTest)
rmse = sqrt(mean_squared_error(inv_yTest, inv_predTest))
Rsq = r2_score(inv_yTest,inv_predTest)
print('Test RMSE: %.3f' % rmse)
print('Test MAPE: %.3f' % mape)
print('Test R-square: %.3f' % Rsq)
pyplot.plot(inv_predTest, label='Predict')
pyplot.plot(inv_yTest, label='Actual')
pyplot.legend()
