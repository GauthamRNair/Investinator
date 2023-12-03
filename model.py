import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import yfinance as yf


class Investinator:
    def __init__(self, stock_name):
        self.stock_name = stock_name
        self.model = Sequential()
        self.dataset = None
        self.sc = MinMaxScaler(feature_range=(0, 1))
        self.X_train = None
        self.y_train = None
        self.test_period = None
        pass

    def prepare_dataset(self, period="1y", test_period="60d"):
        self.dataset = yf.download(self.stock_name, period=period, interval="1d")

        # Get training data
        training_set = self.dataset.iloc[:, 1:2].values[0:-int(test_period[:-1])]
        training_set_scaled = self.sc.fit_transform(training_set)

        # Prepare training data
        X_train = []
        y_train = []
        for i in range(60, len(training_set_scaled)):
            X_train.append(training_set_scaled[i - 60:i, 0])
            y_train.append(training_set_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        self.X_train, self.y_train = X_train, y_train

    def train(self, model=None, period="1y", test_period="60d"):
        self.prepare_dataset(period=period, test_period=test_period)
        self.test_period = test_period
        # Initialize model
        if model:
            self.model = model
            return
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Train model
        self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=32)

    def predict(self):
        test_dataset = yf.download(self.stock_name, period=self.test_period, interval="1d")

        real_stock_price = test_dataset.iloc[:, 1:2].values
        dataset_total = pd.concat((self.dataset['Open'], test_dataset['Open']), axis=0)
        inputs = dataset_total[len(dataset_total) - len(test_dataset) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = self.sc.transform(inputs)
        X_test = []
        for i in range(60, 60 + len(test_dataset)):
            X_test.append(inputs[i - 60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = self.model.predict(X_test)
        predicted_stock_price = self.sc.inverse_transform(predicted_stock_price)

        plt.plot(real_stock_price, color='red', label='Real Stock Price')
        plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()
