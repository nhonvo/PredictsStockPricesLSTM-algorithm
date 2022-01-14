import streamlit as st
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import yfinance as yf
from pandas.tseries.offsets import BDay
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from tensorflow import keras
import warnings
import seaborn as sns
from function import *
from train import *
plt.style.use("fivethirtyeight")
warnings.filterwarnings("ignore")



st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.image("abc.png",width = 150) #logo
st.markdown('# Nhóm Rạp Xiếc \n** Prestige and quality **') # brand name

   
# ------ layout setting---------------------------
window_selection_c = st.sidebar.container() # create an empty container in the sidebar
window_selection_c.markdown("## Nhóm Rạp Xiếc") # add a title to the sidebar container
sub_columns = window_selection_c.columns(2) #Split the container into two columns for start and end date

today = datetime.datetime.today()
YESTERDAY = today - BDay(0)

DEFAULT_START=today -  BDay(365)
START = sub_columns[0].date_input("From", value=DEFAULT_START, max_value=YESTERDAY)
END = sub_columns[1].date_input("To", value=YESTERDAY, max_value=YESTERDAY, min_value=START)


STOCKS = np.array([ "GOOG", "AMZN", "FB","AAPL","TSLA","Another Choice"])  # TODO : include all stocks
SYMB = window_selection_c.selectbox("select stock", STOCKS)

if SYMB != "Another Choice":
# # # # ------------------------Plot stock linecharts--------------------
    st.title('Price data of '+SYMB+' stock')
    tickerData = yf.Ticker(SYMB)
    stock = tickerData.history(period='1d', start=START, end=END)
    field = np.array([ "Open", "High", "Low","Close","Volume"])  # TODO : include all stocks
    fd = window_selection_c.selectbox("select field", field)
    xuatdothi_1(stock[fd])
    st.title('Price data of '+SYMB+' stock')
    stock = stock.drop('Stock Splits',1)
    stock = stock.drop('Dividends',1)
    st.write(stock[:][:])

    #--------------------------------------------------------------------------------------------------------------------------------------
    r_t,mean = fig_1(stock,fd,SYMB)

   

    #--------------------------------------------------------------------------------------------------------------------------------------
    fig_3(stock,fd,SYMB,r_t,mean)



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
    today = today - BDay(0)
    d30bf = today - BDay(100)
    data = tickerData.history(period='1d', start=d30bf, end=today)
    data.reset_index(inplace=True)
    day = got_day()
    predict(data,fd,SYMB,day)
    # THêm bảng testing error ss dữ liệu test vs dự đoán
else:
    uploaded_file = window_selection_c.file_uploader("Choose a file")
    stock = pd.DataFrame()
    if uploaded_file is not None:
        stock = pd.read_csv(uploaded_file,index_col=0,parse_dates=True,infer_datetime_format=True)
        # print(stock)
        
        sl = stock.columns
        fd = window_selection_c.selectbox("select field to show", sl)
        st.title('Price data of your stock')
        xuatdothi_1(stock[fd])
        st.write(stock)
        SYMB = 'Your_stock'
        r_t,mean = fig_1(stock,fd,SYMB)
        fig_3(stock,fd,SYMB,r_t,mean)
        Size = stock[fd].shape[0]
        stock.reset_index(inplace=True)
        Button = window_selection_c.button("Train")
        if Button:
            for ld in sl:
                train(stock,ld)
        day = got_day()
        fdd = window_selection_c.selectbox("select field to predict",sl)
        
        predict(stock[:30],fd,'Your_stock',day)
        