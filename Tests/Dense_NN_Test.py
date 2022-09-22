import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

# Load data

raw_xmr_bust = pd.read_csv('training_data/Binance_XMRUSDT_d.csv')

# Scale data

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(raw_xmr_bust['close'].values.reshape(-1, 1))

# Create train data

prediction_registers = 30

x_train, y_train = [], []

for x in range(prediction_registers, len(scaled_close)):
    x_train.append(scaled_close[x-prediction_registers:x, 0])
    y_train.append(scaled_close[x, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Create the model

model = Sequential()

model.add(Input(shape=(30,)))
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=30, batch_size=10)

# Testing the model

actual_data = raw_xmr_bust['close'].values

total_dataset = pd.concat((raw_xmr_bust['close'], raw_xmr_bust['close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(raw_xmr_bust) - prediction_registers:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)


x_test = []

for x in range(prediction_registers, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_registers:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices_min = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices_min)

# calculate error metrics

error = []
for i in range(len(prediction_prices_min)):
    error.append(abs((scaled_close[i] - prediction_prices_min[i])/scaled_close[i]) * 100)

error_average = sum(error)/len(error)

print("Error average: " + str(error_average))
print("Max error: " + str(max(error)))
print("Min error: " + str(min(error)))

# Print plot

plt.plot(actual_data, color='black', label='Actual Prices')
plt.plot(prediction_prices, color='green', label='Predicted Prices')
# plt.plot(error, color='red', label='Error')
plt.title('Price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.savefig("prediction.png")
