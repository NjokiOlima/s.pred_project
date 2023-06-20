import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler

# ML Model
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

start = "2010-01-01"
end = "2019-12-31"

df = yf.download("AAPL", start, end)

# Plot Apple Closing Price
# plt.figure(figsize=(12,6))
# plt.plot(df.Close)
# plt.title("Apple Closing Price", fontsize=16)
# plt.xlabel("Year")
# plt.ylabel("Price")
# plt.show()

# ma - Moving Average, mean for a specified period of time, e.g 100 days, 200 days
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

# plt.figure(figsize=(12,6))
# plt.plot(df.Close, label="Original")
# plt.plot(ma100, "r", label="MA100")
# plt.plot(ma200, "g", label="MA200")
# plt.legend()
# plt.title("Apple Closing Price, MA100, MA200", fontsize=16)
# plt.xlabel("Year")
# plt.ylabel("Price")
# plt.show()

df = df.reset_index()
# Split data into train and test => 70% training data and 30% testing data
data_train = pd.DataFrame(df["Close"][0 : int(len(df) * 0.70)])
data_test = pd.DataFrame(df["Close"][int(len(df) * 0.70) : int(len(df))])

# print(data_train.shape)
# print(data_test.shape)
# print(data_train.head())
# print(data_test.head())

scaler = MinMaxScaler(feature_range=(0,1))
data_train_array = scaler.fit_transform(data_train)

x_train = []
y_train = []

for i in range(100, data_train_array.shape[0]):
    x_train.append(data_train_array[i - 100: i])
    y_train.append(data_train_array[i, 0])
    
# Convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
# print(x_train.shape)
# print(y_train.shape)

model = Sequential()

# First layer
model.add(LSTM(units=50, activation="relu", return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

# Second layer
model.add(LSTM(units=60, activation="relu", return_sequences=True))
model.add(Dropout(0.3))

# Third layer
model.add(LSTM(units=80, activation="relu", return_sequences=True))
model.add(Dropout(0.4))

# Fourth layer
model.add(LSTM(units=120, activation="relu"))
model.add(Dropout(0.5))

# Connect all layers
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x_train, y_train, epochs=50)

# Model Trained! Save model for re-use
# model.save("s_trend_model")

# Append the past 100 days to testing data
past_100_days = data_train.tail(100)
final_df = pd.concat([past_100_days, data_test], ignore_index=True)

# Scale down testing data
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])
    
# Convert to numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)

# Making Predictions
y_predicted = model.predict(x_test)

# Scale up
scale_factor = 1/0.02123255
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

plt.figure(figsize=(12,6))
plt.plot(y_test, "b", label="Original Price")
plt.plot(y_predicted, "r", label="Predicted Price")
plt.title("Apple Original Closing Price Vs Predicted Closing Price", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Price")
plt.legend()
plt.show()



