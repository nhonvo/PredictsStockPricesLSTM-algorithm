import streamlit as st
import datetime
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
from pandas.tseries.offsets import BDay
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title('Stock forecast dashboard')



      
# ------ layout setting---------------------------
window_selection_c = st.sidebar.container() # create an empty container in the sidebar
window_selection_c.markdown("## TNT Stock Group") # add a title to the sidebar container
sub_columns = window_selection_c.columns(2) #Split the container into two columns for start and end date

today = datetime.datetime.today()
YESTERDAY = today - BDay(0)

DEFAULT_START=today -  BDay(700)
START = sub_columns[0].date_input("From", value=DEFAULT_START, max_value=YESTERDAY)
END = sub_columns[1].date_input("To", value=YESTERDAY, max_value=YESTERDAY, min_value=START)


STOCKS = np.array([ "GOOG", "AMZN", "FB","AAPL","TSLA"])  # TODO : include all stocks
SYMB = window_selection_c.selectbox("select stock", STOCKS)


# # # # ------------------------Plot stock linecharts--------------------

fig=go.Figure()
tickerData = yf.Ticker(SYMB)
stock = tickerData.history(period='1d', start=START, end=END)
field = np.array([ "Open", "High", "Low","Close","Volume"])  # TODO : include all stocks
fd = window_selection_c.selectbox("select field", field)
st.line_chart(stock[fd])
st.write(stock[:][:])

# ----------------------------------------------------------------
# Tính chuỗi return
fig_1=go.Figure()
r_t = np.log((stock['Open']/stock['Open'].shift(1)))
mean = np.mean(r_t)
r_t[0] = mean
plt.figure(figsize=(20, 4))
plt.plot(r_t, linestyle='--', marker='o')
plt.axhline(y=mean, label='mean return', c='red')
plt.legend()
st.plotly_chart(fig_1)

# #----part-1--------------------------------Session state intializations---------------------------------------------------------------

if "TEST_INTERVAL_LENGTH" not in st.session_state:
    # set the initial default value of test interval
    st.session_state.TEST_INTERVAL_LENGTH = 60

if "TRAIN_INTERVAL_LENGTH" not in st.session_state:
    # set the initial default value of the training length widget
    st.session_state.TRAIN_INTERVAL_LENGTH = 500

if "HORIZON" not in st.session_state:
    # set the initial default value of horizon length widget
    st.session_state.HORIZON = 60

if 'TRAINED' not in st.session_state:
    st.session_state.TRAINED=False

# #---------------------------------------------------------Train_test_forecast_splits---------------------------------------------------
st.sidebar.markdown("## Predict")
train_test_forecast_c = st.sidebar.container()
st.title('Predict stock price')

day = train_test_forecast_c.date_input("Day need to predict")
def got_data(data):
    input = list()
    input.append(data[:])
    return np.asarray(input)
def xuly(data):
    input = got_data(data)
    scaler = MinMaxScaler()
    scaler.fit_transform(input)
    input.reshape(-1,30,1)
    # ???
    model = keras.models.load_model(SYMB+'_'+fd+'.h5')
    testk = model.predict(input)
    m = float(testk)
    mn = min(scaler.data_min_)
    e = (max(scaler.data_max_)-min(scaler.data_min_))
    return m*e+mn
day = day-BDay(0)
today = today - BDay(0)
d30bf = today - BDay(31)


data = tickerData.history(period='1d', start=d30bf, end=today)
data.reset_index(inplace=True)
data = data[:][fd]
data = data[-30:]
L = list()
for i in data:
  L.append(i)
date1 = list()
ketqua = list()
mix = list()
while today < day:
    date1.append(today)
    x = xuly(data)
    L = L[1:] + [x]
    data = np.asarray(L)
    ketqua.append(x)
    today = today - BDay(-1)
    mix.append((today,x))

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=date1, y=ketqua, name="stock_open"))
	fig.layout.update(title_text='Predict Chart of '+ str(SYMB), xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()
bang = np.asarray(mix)
st.write(bang)