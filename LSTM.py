import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os


def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="max")
    return data


def preprocess_data(data):
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['Volatility'] = data['Close'].rolling(window=50).std()
    data['RSI'] = compute_rsi(data['Close'])
    data = data.dropna()
    return data


def compute_rsi(series, window=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def add_features(data):
    data['Daily Return'] = data['Close'].pct_change()
    data['Daily Return'].fillna(0, inplace=True)
    data['MA Ratio'] = data['MA50'] / data['MA200']
    data['High-Low'] = data['High'] - data['Low']
    data['Open-Close'] = data['Open'] - data['Close']
    return data


def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length, 0])  # Use only the 'Close' price for y
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model


def plot_predictions(actual, predicted, title, filename):
    plt.figure(figsize=(14, 7))
    plt.plot(actual.index, actual.values, label='Actual')
    plt.plot(predicted.index, predicted.values, label='Predicted')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.savefig(filename)
    plt.close()


# List of some NSE stocks for demonstration purposes
nse_stock_list = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']


# Function to get the best stock
def get_best_stock(stock_list, plot_dir='plots'):
    best_mae = np.inf
    best_stock = None
    best_model = None
    best_data = None
    stock_accuracies = {}

    # Create plot directory if it doesn't exist
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for ticker in stock_list:
        print(f"Processing {ticker}...")
        try:
            data = get_stock_data(ticker)
            last_entry_date = data.index[-1]  # Get the date of the last entry
            print(f"Last data entry for {ticker}: {last_entry_date}")
            data = preprocess_data(data)
            data = add_features(data)

            # Normalize the data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data[['Close', 'MA50', 'MA200', 'Volatility', 'Daily Return', 'MA Ratio', 'RSI', 'High-Low', 'Open-Close']])

            # Create sequences
            sequence_length = 60
            X, y = create_sequences(scaled_data, sequence_length)

            # Splitting the data
            split_index = int(len(X) * 0.9)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            # Train the model
            model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
            model.fit(X_train, y_train, epochs=20, batch_size=32)

            # Predict and evaluate on test data
            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], scaled_data.shape[1] - 1))), axis=1))[:, 0]
            y_test_actual = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1))), axis=1))[:, 0]

            mae = mean_absolute_error(y_test_actual, predictions)

            # Store the accuracy for this stock
            stock_accuracies[ticker] = mae

            # Plotting actual vs. predicted close prices and save plot
            test_data = data.iloc[split_index + sequence_length:]
            test_data['Predicted_Close'] = predictions
            plot_filename = os.path.join(plot_dir, f"{ticker}_actual_vs_predicted.png")
            plot_predictions(test_data['Close'], test_data['Predicted_Close'], f"{ticker} - Actual vs Predicted", plot_filename)

            if mae < best_mae:
                best_mae = mae
                best_stock = ticker
                best_model = model
                best_data = test_data
        except Exception as e:
            print(f"Failed to process {ticker}: {e}")
            continue

    return best_stock, best_mae, best_model, best_data, stock_accuracies


best_stock, best_mae, best_model, best_data, stock_accuracies = get_best_stock(nse_stock_list)

if best_stock is not None:
    print(f"Best stock: {best_stock}")
    print(f"Mean Absolute Error: {best_mae}")
    print("Accuracy for each stock:")
    for stock, mae in stock_accuracies.items():
        print(f"{stock}: MAE = {mae}")
    print(best_data.tail())
else:
    print("No valid results were obtained.")
