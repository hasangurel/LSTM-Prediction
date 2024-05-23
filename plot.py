import matplotlib.pyplot as plt
import pandas as pd

class Plotting:
    def __init__(self, data):
        self.data = data

    def plot_results(self, predictions, Y_test, column):
        # Plot the data
        plt.figure(figsize=(7, 3.5))
        plt.title(f'{column} Model Prediction')
        plt.xlabel('Date')
        plt.ylabel(column)
        plt.plot(Y_test[0], label='True Data')
        plt.plot(predictions, label='Predicted Data')
        plt.legend()
        plt.show()

    def calculate_ma200(self, data):
        return data.rolling(window=200).mean()

    def plot_ma200_comparison(self, predictions, Y_test, column):
        # Convert predictions and Y_test to pandas Series
        predictions_series = pd.Series(predictions.flatten(), index=self.data.index[-len(predictions):])
        Y_test_series = pd.Series(Y_test[0], index=self.data.index[-len(Y_test[0]):])

        # Calculate MA200
        ma200_predictions = self.calculate_ma200(predictions_series)
        ma200_Y_test = self.calculate_ma200(Y_test_series)

        # Plot the MA200 comparison
        plt.figure(figsize=(10, 6))
        plt.plot(ma200_Y_test, label='True Data MA200')
        plt.plot(ma200_predictions, label='Predicted Data MA200')
        plt.title(f'{column} Model MA200 Comparison')
        plt.xlabel('Date')
        plt.ylabel('MA200')
        plt.legend()
        plt.show()
