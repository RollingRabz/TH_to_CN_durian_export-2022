# prepare data for lstm
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
from sklearn.metrics import r2_score
from math import sqrt


# convert series to supervised learning
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

# load dataset
dataset = read_excel('D:/python/New folder/Dataframe.xlsx', header=0, index_col=0)
dataset = dataset.drop(['Exchange_Rate'], axis = 1)
values = dataset.values
# integer encode direction
#encoder = LabelEncoder()
#values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 12, 1)# in = use x month to predict = forecast
# drop columns we don't want to predict
reframed.drop(reframed.columns[[73,74,75,76,77]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values
n_train_hours = 132 #month in our data
train = values[:n_train_hours, :]#first 365*24 row
test = values[n_train_hours:, :]# after that  current run

# split into input and outputs
train_X, train_y = train[:,:-1], train[:, -1]#everything except column-1, everything on column -1
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))#make input in same array
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


# design network
model = Sequential()
model.add(LSTM(4, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=1, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))#revert back to normal shape
# invert scaling 
inv_yhat = yhat*(max(dataset.iloc[:,0])-min(dataset.iloc[:,0]))+min(dataset.iloc[:,0])
inv_y = test_y*(max(dataset.iloc[:,0])-min(dataset.iloc[:,0]))+min(dataset.iloc[:,0])



# calculate error

mape = mean_absolute_percentage_error(inv_y, inv_yhat)
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
Rsq = r2_score(inv_y,inv_yhat)
print('Test RMSE: %.3f' % rmse)
print('Test MAPE: %.3f' % mape)
print('Test R-square: %.3f' % Rsq)



pyplot.plot(inv_yhat, label='Predict')
pyplot.plot(inv_y, label='Actual')
pyplot.legend()
