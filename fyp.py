import math 
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
plt.style.use('fivethirtyeight')
from arch import arch_model
from datetime import datetime, timedelta
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_predict
from pmdarima import auto_arima


start = '2010-01-01'
end = '2022-08-11'

st.title('Stock Trend Prediction')

user_input1 = st.text_input('Enter Stock Ticker', 'AAPL')
prediction_date = st.text_input('Enter Date: ', '2022-07-24')
buying_price = st.number_input('Enter the Buying Price: ')
profit_margin = st.number_input("Enter the Profit Margin: ")

df = web.DataReader(user_input1, 'yahoo', start, end)

#DEscribing Data
st.subheader ('Data From 2010')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time chart')
#visulaize the closing price history
fig=plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Data', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 50MA')
ma50 = df.Close.rolling(50).mean()
fig=plt.figure(figsize=(16,8))
plt.title('Close Price History With 50 Moving Average')
plt.plot(df['Close'])
plt.plot(ma50, 'g')
plt.xlabel('Data', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 50MA & 200MA')
ma200 = df.Close.rolling(200).mean()
fig=plt.figure(figsize=(16,8))
plt.title('Close Price History With 50 & 200 Moving Average')
plt.plot(df['Close'])
plt.plot(ma50, 'g')
plt.plot(ma200, 'r')
plt.xlabel('Data', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
st.pyplot(fig)

returns = 100 * df.Close.pct_change().dropna()
st.subheader( 'Partial Autocorrelation')
fig14=plot_pacf(returns**2)
st.pyplot(fig14)

model = arch_model(returns, p=2, q=2)
model_fit = model.fit()

rolling_predictions = []
test_size = 365*5

for i in range(test_size):
    train = returns[:-(test_size-i)]
    model = arch_model(train, p=2, q=2)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))

rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-365*5:])

st.subheader('Volatility Prediction - Rolling Forecast')
fig15=plt.figure(figsize=(10,4))
true, = plt.plot(returns[-365*5:])
preds, = plt.plot(rolling_predictions)
plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)
st.pyplot(fig15)

train = returns
model = arch_model(train, p=2, q=2)
model_fit = model.fit(disp='off')

pred = model_fit.forecast(horizon=7)
future_dates = [returns.index[-1] + timedelta(days=i) for i in range(1,8)]
pred = pd.Series(np.sqrt(pred.variance.values[-1,:]), index=future_dates)

st.subheader('GARCH MODEL')
fig16=plt.figure(figsize=(10,4))
plt.plot(pred)
plt.title('Volatility Prediction - Next 7 Days', fontsize=20)
st.pyplot(fig16)

from statsmodels.tsa.stattools import adfuller
def adf_test(dataset):
     dftest = adfuller(dataset, autolag = 'AIC')
     for key, val in dftest[4].items():
         print("\t",key, ": ", val)
adf_test(df['Close'])


stepwise_fit = auto_arima(df['Close'], trace=True,
suppress_warnings=True)

train=df.iloc[:-30]
test=df.iloc[-30:]

from statsmodels.tsa.arima.model import ARIMA
model=ARIMA(train['Close'],order=(7,1,4))
model=model.fit()
start=len(train)
end=len(train)+len(test)-1
pred=model.predict(start=start,end=end,typ='levels').rename('ARIMA Predictions')

pred.index=df.index[start:end+1]

st.subheader('ARIMA Predictions')


fig20=plt.figure(figsize=(10,4))
plt.plot(pred)
plt.title('Arima Prediction', fontsize=20)
st.pyplot(fig20)



Start = "2010-01-01"
End = "2022-08-22"
df = web.DataReader("AAPL", "yahoo", Start, End)
df = df[["Close"]].copy()

result = adfuller(df.Close.dropna())
arima_model = ARIMA(df.Close, order=(7,1,4))
result = arima_model.fit()

fig21=plot_predict(
    result,
    start=3100,
    end=3181,
    dynamic=False,
);
st.pyplot(fig21)

#creat a new dataframe with only the 'Close column'
data = df.filter(['Close'])

#convert the dataframe to a numpy array
dataset = data.values

#Get the number of rows to train the model on 
training_data_len = math.ceil(len(dataset) * .8)

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Creat the training data set
#Creat the scaled training data set
train_data  = scaled_data[0:training_data_len , :]
#Split the data set into x_train and y_train data sets
x_train = []
y_train =[]

for i in range(70, len(train_data)):
    x_train.append(train_data[i-70:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 71:
        print(x_train)
        print(y_train)


#Convert the x_train and y_train to numpy arrays

x_train = np.array(x_train)
y_train = np.array(y_train)
#Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


#Load my model

model = load_model('keras_model.h1')

#Create the testing data set
test_data = scaled_data[training_data_len - 70: , :]
#Creat the data set x_test and y_test
x_test = []
y_test = dataset[training_data_len: , :]
for i in range(70, len(test_data)):
    x_test.append(test_data[i-70:i, 0])


#Convert the data into numpy array
x_test = np.array(x_test)


#Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
st.subheader('LSTM Predictions vs Original')

#Visualize the data 
#Final Graph

fig2=plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Data', fontsize=18)
plt.ylabel('Cose Price', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
st.pyplot(fig2)

#Show the valid and predicted price 
valid

#Get the quote
start = '2010-01-01'
apple_quote = web.DataReader(user_input1, 'yahoo', start, prediction_date)
#create a mew data frame
new_df = apple_quote.filter(['Close'])
last_60_days = new_df[-70:].values
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
X_test.append(last_60_days_scaled)                             
#Convert the X_test data set to a numpy array
X_test= np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the prediction scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
st.write(pred_price)
if (pred_price>= buying_price * (1 + profit_margin/100)):
    st.write("Sell this! The stock is", (buying_price * (1 + profit_margin/100)))
else: 
    st.write("Hold this! The stock is", (buying_price * (1 + profit_margin/100)))

st.title('Trading Indicators')
start = '2010-01-01'
end = '2022-08-11'
df = web.DataReader('AAPL', 'yahoo', start, end)

df=df.reset_index ()

Close = df['Close']

Date = df['Date']

Date2 = Date[int(len(Date)*0.99): len(Date)]

returns = 100 * df.Close.pct_change().dropna()

returns2 = returns[int(len(returns)*0.99): len(returns)]

#visulaize the Stock Returns
st.subheader('Stock Returns')
fig3=plt.figure(figsize=(16,8))
plt.title('Stock Returns')
plt.plot(Date2, returns2)
plt.xlabel('Data', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
st.pyplot(fig3)

ema_period = 50  # defining the ema period to 10

# calculating sma in pandas using df.rolling().mean() applied on the close price
# rolling() defines the window for the period where ema_period is passed
# .ewm() creates an exponential weighted window with 'span is equal to our ema_period'
ema50 = df['Close'].ewm(span=ema_period, min_periods=ema_period).mean()

ema_period = 200  # defining the ema period to 10

# calculating sma in pandas using df.rolling().mean() applied on the close price
# rolling() defines the window for the period where ema_period is passed
# .ewm() creates an exponential weighted window with 'span is equal to our ema_period'
ema200 = df['Close'].ewm(span=ema_period, min_periods=ema_period).mean()


ema200 = ema200[int(len(ema200)*0.99): len(ema200)]

ema50 = ema50[int(len(ema50)*0.99): len(ema50)]

Close = Close[int(len(Close)*0.99): len(Close)]

#visulaize the Close Price History With 50 & 200 Exponential Moving Average
st.subheader('Close Price History With 50 & 200 Exponential Moving Average')
fig4=plt.figure(figsize=(16,8))
plt.title('Close Price History With 50 & 200 Exponential Moving Average')
plt.plot(Date2, Close)
plt.plot(Date2, ema50, 'y')
plt.plot(Date2, ema200, 'b')
plt.xlabel('Data', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.legend(['Close', 'ema50', 'ema200'], loc='upper right')
st.pyplot(fig4)

ma50 = df.Close.rolling(50).mean()
ma50 = ma50[int(len(ma50)*0.99): len(ma50)]

ma200 = df.Close.rolling(200).mean()
ma200 = ma200[int(len(ma200)*0.99): len(ma200)]

st.subheader('Close Price History With 50 & 200 EMA VS MA')
fig5=plt.figure(figsize=(16,8))
plt.title('Close Price History With 50 & 200 EMA VS MA')
plt.plot(Date2, Close)
plt.plot(Date2, ema50, 'y')
plt.plot(Date2, ema200, 'b')
plt.plot(Date2, ma50, 'g')
plt.plot(Date2, ma200, 'r')
plt.xlabel('Data', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.legend(['Close', 'ema50', 'ema200', 'ma50', 'ma200'], loc='upper right')
st.pyplot(fig5)

atr_period = 14  # defining the atr period to 14

# calculating the range of each candle
range = df['High'] - df['Low']

# calculating the average value of ranges
atr_14 = range.rolling(atr_period).mean()

atr_14 = atr_14[int(len(atr_14)*0.99): len(atr_14)]

# plotting the ATR Indicator
st.subheader('ATR Indicator')
fig6=plt.figure(figsize=(16,8))
plt.title('ATR Indicator')
plt.plot(Date2, atr_14)
plt.xlabel('Data', fontsize=18)
st.pyplot(fig6)

# setting the RSI Period
rsi_period = 14

# to calculate RSI, we first need to calculate the exponential weighted aveage gain and loss during the period
df['gain'] = (df['Close'] - df['Open']).apply(lambda x: x if x > 0 else 0)
df['loss'] = (df['Close'] - df['Open']).apply(lambda x: -x if x < 0 else 0)

# here we use the same formula to calculate Exponential Moving Average
df['ema_gain'] = df['gain'].ewm(span=rsi_period, min_periods=rsi_period).mean()
df['ema_loss'] = df['loss'].ewm(span=rsi_period, min_periods=rsi_period).mean()

# the Relative Strength is the ratio between the exponential avg gain divided by the exponential avg loss
df['rs'] = df['ema_gain'] / df['ema_loss']

# the RSI is calculated based on the Relative Strength using the following formula
rsi_14 = 100 - (100 / (df['rs'] + 1))
rsi_14 = rsi_14[int(len(rsi_14)*0.99): len(rsi_14)]

# plotting the RSI Indicator
st.subheader('RSI Indicator')
fig7=plt.figure(figsize=(16,8))
plt.title('RSI Indicator')
plt.plot(Date2, rsi_14)
plt.xlabel('Data', fontsize=18)
# RSI commonly uses oversold and overbought levels, usually at 70 and 30
overbought_level = 70
orversold_level = 30

plt.axhline(y=70, color='r', linestyle='-')
plt.axhline(y=30, color='g', linestyle='-')
st.pyplot(fig7)

# to calculate the previous High/Low, we can simply use shift() to check values of previous rows
prev_high = df['High'].shift(1)
prev_low = df['Low'].shift(1)
prev_high = prev_high[int(len(prev_high)*0.99): len(prev_high)]
prev_low = prev_low[int(len(prev_low)*0.99): len(prev_low)]

# High/Low of Previous Session
st.subheader('High/Low of Previous Session')
fig8=plt.figure(figsize=(16,8))
plt.title('High/Low of Previous Session')
plt.plot(Date2, prev_high)
plt.plot(Date2, prev_low)
plt.plot(Date2, Close)
plt.xlabel('Data', fontsize=18)
plt.legend(['Close', 'prev_high', 'prev_low'], loc='upper right')
st.pyplot(fig8)

# setting the deviation period
deviation_period = 20

# simple way to calculate Standard Deviation is to use std() 
std_20 = df['Close'].rolling(20).std()
std_201 = std_20[int(len(std_20)*0.99): len(std_20)]

# plotting the deviation period
st.subheader('Standard Deviation')
fig9=plt.figure(figsize=(16,8))
plt.title('Standard Deviation')
plt.plot( Date2, std_201)
plt.xlabel('Data', fontsize=18)
st.pyplot(fig9)

# setting SMA Period to 20
sma_period = 20

# calculating individual components of Bollinger Bands
sma_20 = df['Close'].rolling(sma_period).mean()
upper_band_20 = sma_20 + 2 * std_20
lower_band_20 = sma_20 - 2 * std_20

upper_band_20 = upper_band_20[int(len(upper_band_20)*0.99): len(upper_band_20)]
lower_band_20 = lower_band_20[int(len(lower_band_20)*0.99): len(lower_band_20)]
sma_20 = sma_20[int(len(sma_20)*0.99): len(sma_20)]

# plotting the Bollinger Bands
st.subheader('Bollinger Bands')
fig10=plt.figure(figsize=(16,8))
plt.title('Bollinger Bands')
plt.plot(Date2, Close)
plt.plot(Date2, sma_20)
plt.plot(Date2, upper_band_20)
plt.plot(Date2, lower_band_20)
plt.xlabel('Data', fontsize=18)
plt.legend(['Close','sma_20', 'upper_band_20', 'lower_band_20'], loc='upper right')
st.pyplot(fig10)

# setting Moving Average Convergence/Divergence (MACD)
# setting the EMA periods
fast_ema_period = 12
slow_ema_period = 26

# calculating EMAs
df['ema_12'] = df['Close'].ewm(span=fast_ema_period, min_periods=fast_ema_period).mean()
df['ema_26'] = df['Close'].ewm(span=slow_ema_period, min_periods=slow_ema_period).mean()

# calculating MACD by subtracting the EMAs
macd = df['ema_26'] - df['ema_12']

# calculating to Signal Line by taking the EMA of the MACD
signal_period = 9
macd_signal = macd.ewm(span=signal_period, min_periods=signal_period).mean()
macd = macd[int(len(macd)*0.99): len(macd)]
macd_signal = macd_signal[int(len(macd_signal)*0.99): len(macd_signal)]

# plotting the Moving Average Convergence/Divergence (MACD)
st.subheader('Moving Average Convergence/Divergence (MACD)')
fig11=plt.figure(figsize=(16,8))
plt.title('Moving Average Convergence/Divergence (MACD)')
plt.plot(Date2, macd_signal)
plt.plot(Date2, macd)
plt.xlabel('Data', fontsize=18)
plt.legend(['macd', 'macd_signal'], loc='upper right')
st.pyplot(fig11)

# setting the SMA Periods
fast_sma_period = 10
slow_sma_period = 20

# calculating fast SMA
df['sma_10'] = df['Close'].rolling(fast_sma_period).mean()

# To find crossovers, previous SMA value is necessary using shift()
df['prev_sma_10'] = df['sma_10'].shift(1)

# calculating slow SMA
df['sma_20'] = df['Close'].rolling(slow_sma_period).mean()

# function to find crossovers
def sma_cross(row):
    
    bullish_crossover = row['sma_10'] >= row['sma_20'] and row['prev_sma_10'] < row['sma_20']
    bearish_crossover = row['sma_10'] <= row['sma_20'] and row['prev_sma_10'] > row['sma_20']
    
    if bullish_crossover or bearish_crossover:
        return True

# applying function to dataframe
df['crossover'] = df.apply(sma_cross, axis=1)

sma_20 = df['sma_20']
sma_10 = df['sma_10']
sma_20 = sma_20[int(len(sma_20)*0.99): len(sma_20)]
sma_10 = sma_10[int(len(sma_10)*0.99): len(sma_10)]

# plotting the SMA Crossover
st.subheader('SMA Crossover')
fig12=plt.figure(figsize=(16,8))
plt.title('SMA Crossover')
plt.plot(Date2, sma_10)
plt.plot(Date2, sma_20)
plt.plot(Date2, Close)
plt.xlabel('Data', fontsize=18)
plt.legend(['sma_10','sma_20', 'Close'], loc='upper right')
st.pyplot(fig12)


# setting the period
stochastic_period = 14

# calculating maximum high and minimum low for the period
df['14_period_low'] = df['Low'].rolling(stochastic_period).min()
df['14_period_high'] = df['High'].rolling(stochastic_period).max()

# formula to calculate the Stochastic Oscillator
stoch_osc = (df['Close'] - df['14_period_low']) / (df['14_period_high'] - df['14_period_low'])
stoch_osc = stoch_osc[int(len(stoch_osc)*0.99): len(stoch_osc)]

# Stochastic Oscillator
st.subheader('Stochastic Oscillator')
fig13=plt.figure(figsize=(16,8))
plt.title('Stochastic Oscillator')
plt.plot(Date2, stoch_osc)
plt.xlabel('Data', fontsize=18)
st.pyplot(fig13)
