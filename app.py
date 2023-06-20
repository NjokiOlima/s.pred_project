import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

from sklearn.preprocessing import MinMaxScaler

start = "2010-01-01"
end = "2019-12-31"

st.title("Stock Prediction Project")
stock = st.text_input("Enter stock ticker", "AAPL")

df = yf.download(stock, start, end)

st.subheader("From 2010 - 2019")
st.write(df.describe())

# Visuals
st.subheader("Closing Price Over Time Chart")
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.title("Closing Price Chart", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Price")
st.pyplot(fig)

st.subheader("Closing Price With MA Over Time Chart")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

fig2 = plt.figure(figsize=(12,6))
plt.plot(df.Close, label="Original")
plt.plot(ma100, "g", label="MA100")
plt.plot(ma200, "r", label="MA200")
plt.title("Closing Price, MA100, MA200 Chart", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)

# Split data into train and test => 70% training data and 30% testing data
data_train = pd.DataFrame(df["Close"][0 : int(len(df) * 0.70)])
data_test = pd.DataFrame(df["Close"][int(len(df) * 0.70) : int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))

# Load Prediction Model
model = load_model("s_trend_model")

# Testing part
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
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader("Predictions Vs Original")
fig3 = plt.figure(figsize=(12,6))
plt.plot(y_test, "b", label="Original Price")
plt.plot(y_predicted, "r", label="Predicted Price")
plt.title("Original Vs Predicted Closing Price", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig3)
