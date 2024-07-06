import requests
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import plotly.express as px

# Function to load data
@st.cache_data
def load_data(file_path):
    gold_data = pd.read_csv(file_path)
    gold_data['Date'] = pd.to_datetime(gold_data['Date'], format='%Y-%m-%d')
    gold_data['Price'] = gold_data['Price'].str.replace(',', '').astype(float)
    gold_data = gold_data.sort_values('Date')
    return gold_data

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

# Function to predict with RandomForest model for a range of dates
def predict_rf(model, scaler, data, prediction_date, sequence_length=7):
    prices = data['Price'].values.reshape(-1, 1)
    prices_scaled = scaler.transform(prices)
    X_pred = []
    for i in range(len(prices_scaled) - sequence_length, len(prices_scaled)):
        X_pred.append(prices_scaled[i - sequence_length:i, 0])
    X_pred = np.array(X_pred)
    future_steps = (prediction_date - data['Date'].max()).days
    if future_steps < 0:
        st.error("Prediction date is in the past.")
        return None
    
    try:
        for _ in range(future_steps):
            prediction = model.predict(X_pred[-1].reshape(1, -1))
            prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()[0]
            prices = np.append(prices, prediction).reshape(-1, 1)
            prices_scaled = scaler.transform(prices)
            X_pred = np.append(X_pred, prices_scaled[-sequence_length:, 0].reshape(1, -1), axis=0)
        
        st.subheader(f"Random Forest Prediction for {prediction_date.date()}: {prediction:.2f} INR")
        return prediction
    
    except Exception as e:
        st.error(f"Failed to make a prediction: {str(e)}")
        return None



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

# Function to predict with SVR model for a range of dates
def predict_svr(model, scaler, data, prediction_date, sequence_length=7):
    prices = data['Price'].values.reshape(-1, 1)
    prices_scaled = scaler.transform(prices)
    X_pred = []
    for i in range(len(prices_scaled) - sequence_length, len(prices_scaled)):
        X_pred.append(prices_scaled[i - sequence_length:i, 0])
    X_pred = np.array(X_pred)
    future_steps = (prediction_date - data['Date'].max()).days
    if future_steps < 0:
        st.error("Prediction date is in the past.")
        return None
    for _ in range(future_steps):
        prediction = model.predict(X_pred[-1].reshape(1, -1))
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()[0]
        prices = np.append(prices, prediction).reshape(-1, 1)
        prices_scaled = scaler.transform(prices)
        X_pred = np.append(X_pred, prices_scaled[-sequence_length:, 0].reshape(1, -1), axis=0)
    st.subheader(f"SVR Prediction for {prediction_date.date()}: {prediction:.2f} INR")
    return prediction

# Function to train ARIMA model
def train_arima_model(data):
    prices = data['Price'].values
    arima_order = (5, 1, 0)
    arima_model = ARIMA(prices, order=arima_order)
    arima_model_fit = arima_model.fit()
    return arima_model_fit

# Function to predict with ARIMA model for a range of dates
def predict_arima(model, data, prediction_date):
    start_date = data['Date'].min()
    end_date = data['Date'].max()
    future_dates = pd.date_range(start=end_date, end=prediction_date)
    future_steps = len(future_dates)
    predictions = model.forecast(steps=future_steps)
    prediction = predictions[-1]
    st.subheader(f"ARIMA Prediction for {prediction_date.date()}: {prediction:.2f} INR")
    return prediction

# Function to calculate error metrics
def calculate_errors(actual, predicted):
    errors = {
        'MAE': np.mean(np.abs(actual - predicted)),
        'MBE': np.mean(predicted - actual),
        'ARE': np.mean(np.abs((actual - predicted) / actual)) * 100,
        'MPE': np.mean((actual - predicted) / actual) * 100,
        'MSE': np.mean((actual - predicted)**2),
        'RMSE': np.sqrt(np.mean((actual - predicted)**2)),
        'RSE': np.sum((actual - predicted)**2) / np.sum((actual - np.mean(actual))**2)
    }
    return errors

# Function to plot actual vs predicted prices
def plot_predictions(actual, rf_pred, svr_pred, arima_pred):
    df = pd.DataFrame({
        'Date': actual.index,
        'Actual': actual.values,
        'Random Forest': rf_pred,
        'SVR': svr_pred,
        'ARIMA': arima_pred
    })
    df = df.melt(id_vars='Date', value_vars=['Actual', 'Random Forest', 'SVR', 'ARIMA'], 
                 var_name='Model', value_name='Price')
    fig = px.line(df, x='Date', y='Price', color='Model', title='Actual vs Predicted Prices')
    st.plotly_chart(fig)

# Function to save today's price to the main CSV file
def save_today_price_to_csv(today_price, file_path):
    today = datetime.now().strftime('%Y-%m-%d')
    new_data = pd.DataFrame({
        'Date': [today],
        'Price': [today_price],
        'Open': [np.nan],
        'High': [np.nan],
        'Low': [np.nan],
        'Vol.': [np.nan],
        'Change %': [np.nan]
    })
    try:
        existing_data = pd.read_csv(file_path)
        if today in existing_data['Date'].values:
            existing_data.loc[existing_data['Date'] == today, 'Price'] = today_price
        else:
            existing_data = pd.concat([new_data, existing_data], ignore_index=True)
        existing_data.to_csv(file_path, index=False)
    except FileNotFoundError:
        new_data.to_csv(file_path, index=False)

# Function to make a Gold API request for the current day's price
def make_gapi_request():
    api_key = "goldapi-3x26usly2hlttk-io"
    symbol = "XAU"
    curr = "INR"
    date = ""
    url = f"https://www.goldapi.io/api/{symbol}/{curr}{date}"
    headers = {
        "x-access-token": api_key,
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        result = response.json()
        if 'price' in result:
            return result['price'] / 31.1035  # Convert price per ounce to price per gram
        else:
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return None

# Main function
def main():
    st.title('Gold Price Prediction')

    # Load data
    file_path = 'mainfinalcsv.csv'
    gold_data = load_data(file_path)

    # Sidebar for user input
    st.sidebar.header('Prediction Date Input')
    prediction_date_input = st.sidebar.date_input("Enter the prediction date:")
    prediction_date = pd.Timestamp(prediction_date_input)

    # Train models if data is loaded
    if 'gold_data' in locals():
        rf_model, rf_scaler = train_rf_model(gold_data)
        svr_model, svr_scaler = train_svr_model(gold_data)
        arima_model = train_arima_model(gold_data)

        # Predict based on user input date
        if prediction_date < gold_data['Date'].min():
            st.error("Prediction date is before the available data range.")
        else:
            st.subheader("Predictions for Selected Date:")
            rf_pred = predict_rf(rf_model, rf_scaler, gold_data, prediction_date)
            svr_pred = predict_svr(svr_model, svr_scaler, gold_data, prediction_date)
            arima_pred = predict_arima(arima_model, gold_data, prediction_date)

            actual = gold_data.set_index('Date')['Price']
            errors = calculate_errors(actual, rf_pred)
            st.write("Error Metrics (Random Forest):", errors)
            errors = calculate_errors(actual, svr_pred)
            st.write("Error Metrics (SVR):", errors)
            errors = calculate_errors(actual, arima_pred)
            st.write("Error Metrics (ARIMA):", errors)

            plot_predictions(actual, rf_pred, svr_pred, arima_pred)

    st.title("Gold Price in INR per Gram for Today")

    # Display today's gold price
    today_price = make_gapi_request()
    if today_price:
        st.header(f"Today's Gold Price: {today_price:.2f} INR per gram")
        save_today_price_to_csv(today_price, file_path)
    else:
        st.error("Failed to retrieve today's gold price.")

if __name__ == "__main__":
    main()
