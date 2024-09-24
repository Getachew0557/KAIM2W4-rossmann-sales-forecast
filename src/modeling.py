import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

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
