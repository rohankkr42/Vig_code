import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Function to load data
@st.cache_data
def load_data(file_path):
    gold_data = pd.read_csv(file_path)
    gold_data['Date'] = pd.to_datetime(gold_data['Date'], format='%m/%d/%Y')
    gold_data['Price'] = gold_data['Price'].str.replace(',', '').astype(float)
    gold_data = gold_data.sort_values('Date')  # Ensure data is sorted by date
    return gold_data

# Function to train ARIMA model
def train_arima_model(data):
    prices = data['Price'].values
    arima_order = (5, 1, 0)
    arima_model = ARIMA(prices, order=arima_order)
    arima_model_fit = arima_model.fit()
    return arima_model_fit

# Function to predict with ARIMA model
def predict_arima(model, steps):
    predictions = model.forecast(steps=steps)
    return predictions

# Function to train SVR model
def train_svr_model(data, sequence_length=7):
    prices = data['Price'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)
    X_train, y_train = [], []
    for i in range(len(prices_scaled) - sequence_length):
        X_train.append(prices_scaled[i:i + sequence_length, 0])
        y_train.append(prices_scaled[i + sequence_length, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train, y_train)
    return svr_model, scaler

# Function to predict with SVR model
def predict_svr(model, scaler, data, sequence_length=7, steps=1):
    prices = data['Price'].values.reshape(-1, 1)
    prices_scaled = scaler.transform(prices)
    last_sequence = prices_scaled[-sequence_length:]
    predictions = []
    for _ in range(steps):
        prediction = model.predict(last_sequence.reshape(1, -1))
        predictions.append(prediction[0])
        last_sequence = np.append(last_sequence[1:], prediction)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Function to train Random Forest model
def train_rf_model(data, sequence_length=7):
    prices = data['Price'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)
    X_train, y_train = [], []
    for i in range(len(prices_scaled) - sequence_length):
        X_train.append(prices_scaled[i:i + sequence_length, 0])
        y_train.append(prices_scaled[i + sequence_length, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train, y_train)
    return rf_model, scaler

# Function to predict with Random Forest model
def predict_rf(model, scaler, data, sequence_length=7, steps=1):
    prices = data['Price'].values.reshape(-1, 1)
    prices_scaled = scaler.transform(prices)
    last_sequence = prices_scaled[-sequence_length:]
    predictions = []
    for _ in range(steps):
        prediction = model.predict(last_sequence.reshape(1, -1))
        predictions.append(prediction[0])
        last_sequence = np.append(last_sequence[1:], prediction)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Plot predictions
def plot_predictions(actual_dates, actual_prices, predicted_dates, predicted_prices, model_name):
    fig, ax = plt.subplots()
    ax.plot(actual_dates, actual_prices, label='Actual Price', color='blue')
    ax.plot(predicted_dates, predicted_prices, label=f'{model_name} Predicted Price', color='orange')
    ax.set_title(f'Actual vs Predicted Gold Prices ({model_name})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (INR)')
    ax.legend()
    st.pyplot(fig)

def main():
    st.title('Gold Price Prediction and Visualization')

    # Load data
    file_path = 'Gold Price (2013-2023).csv'  # Update with your CSV file path
    gold_data = load_data(file_path)

    # Sidebar for user input
    st.sidebar.header('Prediction Date Input')
    prediction_date_input = st.sidebar.date_input("Enter the prediction date:")

    # Convert prediction_date_input to Timestamp
    end_date = pd.Timestamp(prediction_date_input)

    # Train models
    with st.spinner('Training ARIMA model...'):
        arima_model = train_arima_model(gold_data)
    with st.spinner('Training SVR model...'):
        svr_model, svr_scaler = train_svr_model(gold_data)
    with st.spinner('Training RandomForest model...'):
        rf_model, rf_scaler = train_rf_model(gold_data)

    # Predictions
    start_date = gold_data['Date'].max()
    days_to_predict = (end_date - start_date).days

    if days_to_predict > 0:
        # Predictions
        arima_predicted_prices = predict_arima(arima_model, days_to_predict)
        svr_predicted_prices = predict_svr(svr_model, svr_scaler, gold_data, steps=days_to_predict)
        rf_predicted_prices = predict_rf(rf_model, rf_scaler, gold_data, steps=days_to_predict)

        # Get actual dates for plotting
        actual_dates = gold_data['Date']
        actual_prices = gold_data['Price']

        # Get predicted dates for plotting
        predicted_dates = pd.date_range(start=start_date + pd.Timedelta(days=1), periods=days_to_predict)

        # Display predicted prices for the input date
        st.subheader('Predicted Prices for Input Date:')
        st.write(f'ARIMA Predicted Price: {arima_predicted_prices[-1]:.2f} INR')
        st.write(f'SVR Predicted Price: {svr_predicted_prices[-1][0]:.2f} INR')
        st.write(f'RandomForest Predicted Price: {rf_predicted_prices[-1][0]:.2f} INR')

        # Visualization of Predicted Prices
        st.subheader('Visualization of Predicted Prices:')
        plot_predictions(actual_dates, actual_prices, predicted_dates, arima_predicted_prices, 'ARIMA')
        plot_predictions(actual_dates, actual_prices, predicted_dates, svr_predicted_prices, 'SVR')
        plot_predictions(actual_dates, actual_prices, predicted_dates, rf_predicted_prices, 'RandomForest')

if __name__ == '__main__':
    main()
