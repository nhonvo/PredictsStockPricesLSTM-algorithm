import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from tensorflow import keras
import streamlit as st
import plotly.graph_objects as go
from pandas.tseries.offsets import BDay
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def got_data(data):
    input = list()
    input.append(data[:])
    return np.asarray(input)
def xuly(data,SYMB,fd):
    input = got_data(data)
    scaler = MinMaxScaler()
    scaler.fit_transform(input)
    input.reshape(-1,30,1)
    model = keras.models.load_model('Model\\' + SYMB+'_'+fd+'.h5')
    testk = model.predict(input)
    m = float(testk)
    mn = min(scaler.data_min_)
    e = (max(scaler.data_max_)-min(scaler.data_min_))
    return m*e+mn
def xuatdothi_1(bang):
    st.line_chart(bang)
def plot_raw_data(date1,ketqua,SYMB):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=date1, y=ketqua, name="stock_open"))
        fig.layout.update(title_text='Predict Chart of '+ str(SYMB), xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
def got_day():
    st.sidebar.markdown("## Predict")
    train_test_forecast_c = st.sidebar.container()
    st.title('Predict stock price')

    day = train_test_forecast_c.date_input("Day need to predict")

    day = day-BDay(0) + datetime.timedelta(days=1) - datetime.timedelta(hours=1)
    return day
def predict(data,fd,SYMB,day):
    
    today = datetime.datetime.today()
    today = today - BDay(0)
    


    data = data[:][fd]
    data = data[-30:]
    L = list()
    for i in data:
        L.append(i)
    # print(L)
    # print(day,today)
    date1 = list()
    ketqua = list()
    mix = {}
    while today <= day:
        date1.append(today)
        x = xuly(data,SYMB,fd)
        L = L[1:] + [x]
        data = np.asarray(L)
        ketqua.append(x)
        mix[today]=x
        today = today - BDay(-1)
        
    plot_raw_data(date1,ketqua,SYMB)
    data_item = mix.items()
    data_list = list(data_item)
    df = pd.DataFrame(data_list,columns=['Day','Value'])
    st.write(df)
def fig_1(stock,fd,SYMB):
    fig_1,ax = plt.subplots()
    fig_1.set_figheight(5)
    fig_1.set_figwidth(12)
    r_t = np.log((stock[fd]/stock[fd].shift(1)))
    mean = np.mean(r_t)
    r_t[0] = mean
    ax.plot(r_t, linestyle='--', marker='o')
    ax.axhline(y=mean, label='mean return', c='red')
    ax.legend()
    st.write('\n')
    st.title('Prive movement chart '+str(fd)+' of '+SYMB)
    st.pyplot(fig_1)
    return r_t,mean
def fig_3(stock,fd,SYMB,r_t,mean):
    st.title('Overall average daily profit of '+SYMB)
    fig_3,ax = plt.subplots()
    fig_3.set_figheight(5)
    fig_3.set_figwidth(12)
    sns.distplot(r_t, bins = 20)
    plt.axvline(x=mean, label='mean return', c='red')
    plt.legend()
    plt.xlabel('return rate')
    plt.ylabel('frequency')
    st.pyplot(fig_3)