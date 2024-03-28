import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.regularizers import l2
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import plotly.graph_objs as go

st.title('Stock Price Prediction')

ticker_symbol = st.text_input('Enter Stock Symbol (e.g., RELIANCE.NS)', 'RELIANCE.NS')

start_date = st.date_input('Start Date', value=pd.to_datetime('2010-01-01'))
end_date = st.date_input('End Date', value=pd.to_datetime('2024-01-28'))

stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)['Close']

# Check if data is empty
if len(stock_data) == 0:
    st.error("No data available for the specified stock symbol and date range. Please try again with different inputs.")
else:
    data = stock_data.values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    time_step = st.slider('Time Step', min_value=1, max_value=100, value=100)
    X, y = [], []
    for i in range(len(scaled_data) - time_step - 1):
        X.append(scaled_data[i:(i + time_step), 0])
        y.append(scaled_data[i + time_step, 0])
    X, y = np.array(X), np.array(y)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(X.shape[1], 1), kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(LSTM(units=64, return_sequences=True, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(LSTM(units=32, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))

    st.write(model.summary())

    from keras.optimizers import RMSprop

    model.compile(loss='mse', optimizer=RMSprop(learning_rate=0.0005), metrics=['mean_absolute_error'])

    model.fit(X, y, epochs=20, batch_size=64)

    # Extend the prediction to next 100 days
    future_time_steps = 100

    future_predictions = []
    last_sequence = scaled_data[-time_step:]
    for i in range(future_time_steps):
        next_prediction = model.predict(np.array([last_sequence]))[0, 0]
        future_predictions.append(next_prediction)
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_prediction

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    next_dates = pd.date_range(start=end_date, periods=future_time_steps + 1)[1:]

    # Plotting the graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data.values.flatten(), mode='lines', name='Original Prices'))
    fig.add_trace(go.Scatter(x=next_dates, y=future_predictions.flatten(), mode='lines', name='Predicted Prices'))
    fig.update_layout(title='Stock Price Prediction', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

    # Predicted Prices Table
    df_predictions = pd.DataFrame({'Date': next_dates, 'Predicted Price': future_predictions.flatten()})
    st.subheader('Predicted Prices for the Next 100 Days:')
    st.dataframe(df_predictions.style.format({'Predicted Price': '{:.2f}'}), height=300)
