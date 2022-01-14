import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow import keras
plt.style.use("fivethirtyeight")
import warnings
import streamlit as st

warnings.filterwarnings("ignore")
def get_data(train,test,time_step,num_predict,date):
    x_train= list()
    y_train = list()
    x_test = list()
    y_test = list()
    date_test= list()

    for i in range(0,len(train) - time_step - num_predict):
        x_train.append(train[i:i+time_step])
        y_train.append(train[i+time_step:i+time_step+num_predict])

    for i in range(0, len(test) - time_step - num_predict):
        x_test.append(test[i:i+time_step])
        y_test.append(test[i+time_step:i+time_step+num_predict])
        date_test.append(date[i+time_step])
    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), np.asarray(date_test)
def train(data,prop):
    data_end = int(np.floor(0.8*(data.shape[0])))
    train = data[0:data_end][prop] 
    test = data[data_end:][prop]
    date_test = data[data_end:]['Date']
    train = train.values.reshape(-1)
    test = test.values.reshape(-1)
    date_test = date_test.values.reshape(-1)
    x_train, y_train, x_test, y_test, date_test = get_data(train,test,30,1, date_test)

    x_train = x_train.reshape(-1,30)
    x_test = x_test.reshape(-1,30)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    y_train = scaler.fit_transform(y_train)

    x_test = scaler.fit_transform(x_test)
    y_test = scaler.fit_transform(y_test)
    x_train = x_train.reshape(-1,30,1)
    y_train = y_train.reshape(-1,1)

    x_test = x_test.reshape(-1,30,1)
    y_test = y_test.reshape(-1,1)
    date_test = date_test.reshape(-1,1)
    n_input = 30
    n_features = 1

    model = Sequential()
    model.add(LSTM(units = 50, input_shape=(n_input, n_features), return_sequences=True))

    model.add(Dropout(0.3))
    model.add(LSTM(units = 50, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer ='adam', loss ='mse')
    model.fit(x_train, y_train, epochs=500, validation_split=0.2, verbose=1, batch_size=30)
    model.save(f'Model\\Your_stock_{prop}.h5')
    model = keras.models.load_model(f'Model\\Your_stock_{prop}.h5')
    test_output = model.predict(x_test)

    test_1 = scaler.inverse_transform(test_output)
    test_2 =scaler.inverse_transform(y_test)
    fig,ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(12)
    plt.title(prop)
    plt.plot(test_1, color='r')
    plt.plot(test_2, color='b')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(('prediction', 'reality'),loc='upper right')
    st.pyplot(fig)
    # plt.show()
# stock = pd.read_csv("NFLX.csv")
# train(stock,'Open')