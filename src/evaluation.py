# src/evaluation.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

def evaluate_lstm_model(y_test_lstm_rescaled, y_pred_lstm_rescaled):
    """Evaluates LSTM model on test data."""
    mae_lstm = mean_absolute_error(y_test_lstm_rescaled, y_pred_lstm_rescaled)
    mse_lstm = mean_squared_error(y_test_lstm_rescaled, y_pred_lstm_rescaled)
    r2_lstm = r2_score(y_test_lstm_rescaled, y_pred_lstm_rescaled)

    print(f"LSTM Model - Test Set MAE: {mae_lstm:.2f}")
    print(f"LSTM Model - Test Set MSE: {mse_lstm:.2f}")
    print(f"LSTM Model - Test Set R²: {r2_lstm:.2f}")

    # Plot the results
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_lstm_rescaled, label='Actual Sales', color='blue')
    plt.plot(y_pred_lstm_rescaled, label='Predicted Sales', color='orange')
    plt.title('LSTM Model Predictions vs Actual Sales')
    plt.xlabel('Samples')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

def plot_rf_confidence_interval(rf_model, X_test, y_test, y_pred_test):
    """Plots predictions with confidence intervals for Random Forest model."""
    # Confidence Interval estimation (based on the standard deviation of predictions)
    y_pred_std = np.std([tree.predict(X_test) for tree in rf_model.estimators_], axis=0)
    confidence_interval = 1.96 * y_pred_std  # 95% confidence interval

    # Plotting the predictions with confidence intervals
    plt.figure(figsize=(10, 6))
    plt.errorbar(y_test.index, y_pred_test, yerr=confidence_interval, fmt='o', ecolor='r', capthick=2, label="Confidence Interval")
    plt.scatter(y_test.index, y_test, color='blue', label='Actual Values', alpha=0.5)
    plt.title("Random Forest Predictions with 95% Confidence Interval")
    plt.xlabel("Sample Index")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.show()