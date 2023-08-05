import numpy as np
import pandas as pd
import datetime as dtm

import yfinance as yf

# ML libraries
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

import math

import streamlit as st

# Global Variables
start = "2012-01-01"
end = "2023-07-01"

st.title("Stock Prediction Project")

is_ke_stocks = st.checkbox('Uncheck to view international stocks', value=True) # Show KE stocks by default

if is_ke_stocks:
    st.subheader("Showing KE stocks")

    filename = "SCOM08.csv" # Read SCOM data by default
    tckr = "Safaricom"
    model_name = "ke_lstm_model3"

    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])

    with col1:
        if st.button("Safaricom"):
            filename = "SCOM08.csv"
            tckr = "Safaricom"
    with col2:
        if st.button("KCB", type= "primary" if (tckr == "KCB") else "secondary"):
            filename = "KCB.csv"
            tckr = "KCB"
    with col3:
        if st.button("DTB"):
            filename = "DTB.csv"
            tckr = "DiamondTrustBank"
    with col4:
        if st.button("EVEREADY"):
            filename = "EVEREADY.csv"
            tckr = "EVEREADY"
    with col5:
        if st.button("SASINI"):
            filename = "SASINI.csv"
            tckr = "SASINI"
    
    # LOAD DATA
    df = pd.read_csv(filename, skipinitialspace=True)

    # CLEANING DATA
    # Reverse dataframe starting with earliest stock as first index
    df = df.iloc[::-1]
    # Drop reversed index and add default integer index
    df = df.reset_index()
    df = df.drop("index", axis=1)

    # Convert date column to standard date format
    for idx, row in df.iterrows():
        row.Date = dtm.datetime.strptime(row.Date, "%m/%d/%y")
        # row.Date = row.Date.strftime("%Y-%m-%d")
        df.at[idx,"Date"] = row.Date

    # Get start date and end date
    start = df["Date"].iloc[0].strftime("%Y-%m-%d")
    end = df["Date"].iloc[len(df) - 1].strftime("%Y-%m-%d")

    # Set Date as default index
    df.set_index("Date", inplace=True)
else:
    st.subheader("International stocks")
    model_name = "us_lstm_model2"
    tckr = st.text_input("Enter stock ticker", "AAPL")  # Choose a ticker if showing international stocks
    df = yf.download(tckr, start, end)

# Format date to standard format
new_start = dtm.datetime.strptime(start, "%Y-%m-%d")
new_end = dtm.datetime.strptime(end, "%Y-%m-%d")

st.subheader("{} Historical Data".format(tckr))
st.text("From {} - {}".format(new_start.strftime("%Y"), new_end.strftime("%Y")))

# Show latest 10 values
# st.write(df.head(10))
st.text("Showing the last 10 values")
st.dataframe(df.tail(10), use_container_width=True)

# Visuals
st.subheader("{} - Closing Price History".format(tckr))
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.title("Closing Price Historical Data", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Price")
st.pyplot(fig)

# TEST 
# New dataframe with only close column
data = df.filter(["Close"])
# Convert the dataframe into a numpy array
dataset = data.values
# Get length of training and testing data
training_data_len = math.ceil(len(dataset) * 0.8)
test_data_len = len(dataset) - training_data_len
# Scale down data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Testing dataset
test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60: i, 0])

# Convert to numpy arrays
x_test = np.array(x_test)

# Reshape to 3D format
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Load Saved Model
from keras.models import load_model
model = load_model(model_name)

# Get the models predicted values
predictions = model.predict(x_test)
# Scale down data
predictions = scaler.inverse_transform(predictions)

# rmse = np.sqrt(np.mean(predictions - y_test)**2)
# st.text("{} Root Mean Square Error - RMSE".format(rmse))
# st.text("The closer the number to zero, the closer the predicted values to original values, the better it was predicting")

# Plot data
train = data[0: training_data_len] # from zero
valid = data[training_data_len:] # validation data
valid["Predictions"] = predictions

st.subheader("Predictions")
# Visualise Predictions data
st.subheader("{} - Predictions Vs Original".format(tckr))
fig2 = plt.figure(figsize=(12,6))
plt.title("Original Vs Predicted Closing Price", fontsize=16)
plt.xlabel("Date", fontsize=16)
plt.ylabel("Close Price", fontsize=16)
plt.plot(train.Close)
plt.plot(valid[["Close", "Predictions"]])
plt.legend(["Train", "Validations", "Predictions"])
st.pyplot(fig2)

# COMPARE ACTUAL CLOSING AND PREDICTED CLOSING PRICE
st.subheader("{} - Original Closing Vs Predicted Closing Price".format(tckr))
st.text("Showing the last 10 values")
st.dataframe(valid.tail(10), use_container_width=True)

# PREDICT FOR THE FUTURE DAYS
# Predict for the next n_days => n_days can be 10, 20, 30, 60, 100 days e.t.c
n_days = st.slider('Use slider change future days ', 5, 90, 10)

# Create new dataframe
new_df = df.filter(["Close"])

# Combined dataframe: Contains original close prices and predicted prices
combined = new_df

data = {
    "Date": [],
    "Close": []
}
next_n_days = pd.DataFrame(data)
next_n_days.set_index('Date', inplace=True)

last_day = dtm.datetime.strptime(end, "%Y-%m-%d")
if not is_ke_stocks:
    last_day = last_day - dtm.timedelta(days=1)
last_day = last_day.strftime("%Y-%m-%d")
last_day = dtm.datetime.strptime(last_day, "%Y-%m-%d")

for i in range(n_days):
    # Get the last 100 days closing price and convert dataframe to an array
    last_n_days = combined[-100:].values
    # Scale down data
    last_n_days_scaled = scaler.transform(last_n_days)
    X_test = []
    X_test.append(last_n_days_scaled)
    # Convert to numpy arrays
    X_test = np.array(X_test)
    # Reshape to 3D format
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # Predicted price
    pred_price = model.predict(X_test)
    # Scale up
    pred_price = scaler.inverse_transform(pred_price)
    
    #. Update dataframe with predicted value
    tomorrow = last_day + dtm.timedelta(days=1)
    new_row = {
        "Date": [tomorrow],
        "Close": [pred_price[0].tolist()[0]]
    }
    last_day = tomorrow
    next_day = pd.DataFrame(new_row)
    next_day.set_index('Date', inplace=True)
    next_n_days = pd.concat([next_n_days, next_day])
    combined = pd.concat([combined, next_day])


st.subheader("{} - Next {} Days Predicted Closing Price ".format(tckr, n_days))
st.dataframe(next_n_days, use_container_width=True)

# Show previous 3 months plus n_days (10, 30, 60 e.t.c) predictions
prev_days = st.slider('Use slider change number of previous days to view', 30, len(valid), 90)
prev_n_days = new_df[-prev_days:]

# Visualise Future days predictions and original data
st.subheader("{} - Previous {} Days Original Closing Price and Next {} Days Predictions Graph".format(tckr, prev_days, n_days))
fig3 = plt.figure(figsize=(16,8))
plt.title("Prediction Model", fontsize=16)
plt.xlabel("Date", fontsize=16)
plt.ylabel("Close Price", fontsize=16)
plt.plot(prev_n_days.Close)
plt.plot(next_n_days.Close)
plt.legend(["Original", "Predictions"])
st.pyplot(fig3)

