import math
import numpy as np
import pandas as pd
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from keras.models import load_model
import streamlit as st



today = datetime.today()
yesterday = today - timedelta(days=1)

start = '2010-01-01'
end = yesterday

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)

df=df.reset_index ()
Close = df['Close']
Date = df['Date']


#DEscribing Data
st.subheader ('Data From 2010 - yesterday')
st.write(df.describe())


#Visualizations
st.subheader('Closing Price vs Time chart')
fig=plt.figure(figsize=(16,8))
plt.title('Close Price Movement')
plt.plot(Date, Close)
plt.xlabel('Date', fontsize=18)
plt.ylabel('close price in $', fontsize=18)

st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 50MA')
ma50 = df.Close.rolling(50).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(Date, Close)
plt.plot(Date, ma50, 'r')
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 50MA & 200MA')
ma50 = df.Close.rolling(50).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(Date, Close)
plt.plot(Date, ma50, 'g')
plt.plot(Date, ma200, 'r')
st.pyplot(fig)
 
# Spliting data into Traing and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


#Load my model

model = load_model('keras_model.h5')

#Testing Part 

past_100_days = data_training.tail(100) 
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range (100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)


scaler=scaler.scale_


scale_factor = 1/0.00690691
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor
Date1 = df['Date'][int(len(df)*0.70):int(len(df))]

 

#Final Graph

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(Date1, y_test, 'b', label = 'Original Price')
plt.plot(Date1, y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

y_test1 = y_test[int(len(y_test)*0.95): int(len(y_test)*0.98)]

y_predicted1 = y_predicted[int(len(y_predicted)*0.95): len(y_predicted)]

Date2 = Date1[int(len(Date1)*0.95): int(len(Date1)*0.98)]

Date3 = Date1[int(len(Date1)*0.95): len(Date1)] 

st.subheader('Predictions')
fig3 = plt.figure(figsize=(12,6))
plt.plot(Date2, y_test1, 'b', label = 'Original Price')
plt.plot(Date3, y_predicted1, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)
