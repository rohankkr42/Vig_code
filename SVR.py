import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

# Function to load data
@st.cache
def load_data(file_path):
    gold_data = pd.read_csv(file_path)
    gold_data['Date'] = pd.to_datetime(gold_data['Date'], format='%m/%d/%Y')
    gold_data['Price'] = gold_data['Price'].str.replace(',', '').astype(float)
    gold_data = gold_data.sort_values('Date')  # Ensure data is sorted by date
    return gold_data

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
def predict_svr(model, scaler, data, start_date, end_date, sequence_length=7):
    prices = data['Price'].values.reshape(-1, 1)
    prices_scaled = scaler.transform(prices)
    
    # Prepare input sequences for prediction
    X_pred = []
    for i in range(len(prices_scaled) - sequence_length, len(prices_scaled)):
        X_pred.append(prices_scaled[i - sequence_length:i, 0])
    X_pred = np.array(X_pred)
    
    # Predict for each day in the forecast period
    predictions = []
    current_sequence = X_pred[-1].reshape(1, -1)
    for _ in range((end_date - start_date).days):
        prediction = model.predict(current_sequence)
        predictions.append(prediction[0])  # Append single prediction value
        current_sequence = np.append(current_sequence[:, 1:], prediction).reshape(1, -1)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    forecast_dates = pd.date_range(start=start_date + pd.Timedelta(days=1), end=end_date)
    return forecast_dates, predictions

# Main function
def main():
    st.title('Gold Price Prediction - SVR Model')

    # Load data
    file_path = 'Gold Price (2013-2023).csv'  # Update with your CSV file path
    gold_data = load_data(file_path)

    # Sidebar for user input
    st.sidebar.header('Prediction Date Input')
    prediction_date_input = st.sidebar.date_input("Enter the prediction date:")

    # Convert prediction_date_input to Timestamp
    end_date = pd.Timestamp(prediction_date_input)

    # Train SVR model
    with st.spinner('Training SVR model...'):
        svr_model, svr_scaler = train_svr_model(gold_data)

    # Predictions
    start_date = gold_data['Date'].max()

    if end_date > start_date:
        st.subheader('Predictions:')
        try:
            svr_dates, svr_predictions = predict_svr(svr_model, svr_scaler, gold_data, start_date, end_date)
            for date, prediction in zip(svr_dates, svr_predictions):
                st.write(f'{date.date()}: {prediction[0]:.2f} INR')  # Access individual prediction value
        except Exception as e:
            st.write(f'Error predicting with SVR: {str(e)}')

if __name__ == "__main__":
    main()
