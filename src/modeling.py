import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

def train_models(df):
    # Define target and features
    X = df.drop(columns=['Sales', 'Date'])
    y = df['Sales']
    
    # One-Hot Encoding for categorical columns
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # RandomForest model pipeline
    rf_pipeline = Pipeline([
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # XGBoost model pipeline
    xgb_pipeline = Pipeline([
        ('model', XGBRegressor(n_estimators=100, random_state=42))
    ])

    # Fit the models
    rf_pipeline.fit(X_train, y_train)
    xgb_pipeline.fit(X_train, y_train)

    return rf_pipeline, xgb_pipeline, X_test, y_test


def train_lstm_model(df):
    """Trains an LSTM model on sales data."""
    sales = df['Sales'].values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_sales = scaler.fit_transform(sales.reshape(-1, 1))

    def create_lagged_data(data, time_steps=60):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps])
            y.append(data[i + time_steps])
        return np.array(X), np.array(y)

    time_steps = 60
    X_lstm, y_lstm = create_lagged_data(scaled_sales, time_steps)

    # Split data into train and test sets
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

    # Reshape input to be 3D [samples, time steps, features]
    X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
    X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

    # Build LSTM model
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dense(50, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=64, validation_data=(X_test_lstm, y_test_lstm), 
              callbacks=[early_stop], verbose=2)

    # Predict on test data
    y_pred_lstm = model.predict(X_test_lstm)

    # Rescale predictions back to original scale
    y_pred_lstm_rescaled = scaler.inverse_transform(y_pred_lstm)
    y_test_lstm_rescaled = scaler.inverse_transform(y_test_lstm)

    return model, y_test_lstm_rescaled, y_pred_lstm_rescaled
