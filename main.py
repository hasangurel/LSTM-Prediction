import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
import math
from sklearn.metrics import mean_squared_error
from plot import Plotting

class MultiLSTMModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.scalers = {}
        self.models = {}

    def load_data(self):
        # Load the data
        self.data = pd.read_csv(self.file_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)

    def preprocess_data(self, column):
        # Preprocess the data for a given column
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.data[column].values.reshape(-1, 1))
        self.scalers[column] = scaler

        # Split the data into train and test sets
        train_size = int(len(scaled_data) * 0.6)  # Changed from 0.7 to 0.6
        train_data = scaled_data[0:train_size, :]
        test_data = scaled_data[train_size:len(scaled_data), :]

        return train_data, test_data

    def create_dataset(self, dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    def build_model(self, input_shape):
        # Build a single LSTM model
        model = Sequential()
        model.add(LSTM(45, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(45, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_individual_models(self, batch_size=16, epochs=100):
        self.load_data()
        columns = ['Open', 'High', 'Low']

        for column in columns:
            train_data, test_data = self.preprocess_data(column)
            X_train, Y_train = self.create_dataset(train_data)
            X_test, Y_test = self.create_dataset(test_data)

            # Reshape input to be [samples, time steps, features]
            X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

            # Build and train the model
            model = self.build_model((X_train.shape[1], X_train.shape[2]))
            early_stopping = EarlyStopping(monitor='val_loss', patience=5)
            model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,
                      callbacks=[early_stopping])

            # Save the model and test data
            self.models[column] = model
            self.models[f'{column}_test'] = (X_test, Y_test)

    def train_close_model(self, look_back=1, batch_size=16, epochs=100):
        # Preprocess the data for 'Close'
        train_data, test_data = self.preprocess_data('Close')

        # Get the predictions from individual models
        predictions = []
        for column in ['Open', 'High', 'Low']:
            X_test, Y_test = self.models[f'{column}_test']
            prediction = self.models[column].predict(X_test)
            predictions.append(prediction)

        # Concatenate predictions to use as features for 'Close' model
        combined_features = np.concatenate(predictions, axis=1)

        # Create dataset for 'Close' model
        X_train, Y_train = self.create_dataset(combined_features, look_back)
        X_test, Y_test = self.create_dataset(combined_features, look_back)

        # Reshape input to be [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        # Build and train the model for 'Close'
        model = self.build_model((X_train.shape[1], X_train.shape[2]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,
                  callbacks=[early_stopping])

        # Save the close model
        self.models['Close'] = model
        self.models['Close_test'] = (X_test, Y_test)

    def evaluate_model(self, column):
        X_test, Y_test = self.models[f'{column}_test']
        model = self.models[column]

        # Make predictions
        predictions = model.predict(X_test)

        # Invert the predictions to original scale
        predictions = self.scalers[column].inverse_transform(predictions)
        Y_test = self.scalers[column].inverse_transform([Y_test])

        # Calculate RMSE
        score = math.sqrt(mean_squared_error(Y_test[0], predictions[:, 0]))
        print(f'{column} Model Score: {score:.2f} RMSE')

        return predictions, Y_test

    def calculate_ma200(self, data):
        return data.rolling(window=200).mean()

# Usage example
model = MultiLSTMModel('HistoricalData_1716418071988.csv')
model.train_individual_models(batch_size=16, epochs=100)
model.train_close_model(batch_size=16, epochs=100)

# Evaluate and plot individual models
plotter = Plotting(model.data)

for column in ['Open', 'High', 'Low']:
    predictions, Y_test = model.evaluate_model(column)
    plotter.plot_results(predictions, Y_test, column)

# Evaluate and plot the 'Close' model
predictions, Y_test = model.evaluate_model('Close')
plotter.plot_results(predictions, Y_test, 'Close')

# Plot MA200 comparison for 'Close' model
plotter.plot_ma200_comparison(predictions, Y_test, 'Close')
