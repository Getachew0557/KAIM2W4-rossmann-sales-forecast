# Import necessary libraries
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
from sklearn.preprocessing import LabelEncoder
# Date and time
from datetime import datetime

# Warnings and logging
import warnings
import logging
warnings.filterwarnings("ignore")

# Append the correct src path for custom module imports
sys.path.append(os.path.abspath('../src'))
sys.path.append(os.path.abspath('../data'))

from data_processing import load_data, clean_data, handle_missing_values
from sale_analysis import (
    plot_sales_distribution,
    compare_sales_holidays,
    seasonal_behavior,
    correlation_analysis,
    promo_effect,
    effective_promo_deployment,
    customer_behavior_trends,
    weekday_openings,
    assortment_type_impact,
    competitor_distance_impact,
    new_competitor_effects, 
    plot_promo_distribution,
    plot_sales_during_holidays,
    plot_sales_customers_corr, 
    plot_store_corr
)
from preprocessing import load_data, merge_data, feature_engineering
from modeling import train_models, train_lstm_model
from evaluation import evaluate_model, evaluate_lstm_model, plot_rf_confidence_interval
from serialization import save_model

def main():
    # Load datasets
    train_data, test_data, store_data = load_data('../data/train.csv', '../data/test.csv', '../data/store.csv')

    print("Training Data", train_data.head())
    print("Training Data", test_data.head())
    print("Training Data", store_data.head())
    print ("Stastics summary", train_data.describe())

    # Clean datasets
    train_data = clean_data(train_data)
    test_data = clean_data(test_data)

    # Merge datasets
    merged_test_data = pd.merge(test_data, store_data, on='Store')
    merged_data = pd.merge(train_data, store_data, on='Store')
    merged_data.head()

    # Handle missing values after merging
    merged_train_data = handle_missing_values(merged_data)
    merged_test_data = handle_missing_values(merged_test_data)

    # Distribution Analysis
    plot_sales_distribution(train_data, test_data)

    # Compare promo distribution in training and test set
    plot_promo_distribution()  

    # Distribution of the stores based on StoreType
    plt.figure(figsize=(8, 6))
    sns.countplot(x='StoreType', data=merged_data)
    plt.title("Distribution of Stores by StoreType")
    plt.show()

    # Distribution of the stores based on Assortment type
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Assortment', data=merged_data)
    plt.title("Distribution of Stores by Assortment")
    plt.show()

    # Check the distribution of CompetitionDistance
    plt.figure(figsize=(10, 6))
    sns.histplot(merged_data['CompetitionDistance'], kde=True)
    plt.title('Distribution of CompetitionDistance')
    plt.show()

    # Sales Behavior During Holidays
    compare_sales_holidays(merged_data)

    # Group by holidays and calculate average sales
    holiday_sales = merged_data.groupby('StateHoliday')['Sales'].mean().reset_index()

    # Plotting sales before, during, and after holidays
    plt.figure(figsize=(10, 6))
    sns.barplot(x='StateHoliday', y='Sales', data=holiday_sales)
    plt.title('Sales During State Holidays')
    plt.show()

    # Seasonal Purchase Behavior
    seasonal_behavior(merged_data)

    # Correlation Analysis
    correlation_analysis(merged_data)

    # Correlation analysis
    plot_sales_customers_corr() 

    # Create a label encoder instance
    label_encoder = LabelEncoder()

    # Convert 'StoreType', 'Assortment', and 'PromoInterval' from strings to numeric codes
    store_data['StoreType'] = label_encoder.fit_transform(store_data['StoreType'])
    store_data['Assortment'] = label_encoder.fit_transform(store_data['Assortment'])

    # For 'PromoInterval', first replace missing values with 'None', then apply label encoding
    store_data['PromoInterval'].fillna('None', inplace=True)
    store_data['PromoInterval'] = label_encoder.fit_transform(store_data['PromoInterval'])

    # Now check the correlation
    correlation_matrix = store_data.corr()

    # Display correlation matrix
    correlation_matrix

    # Promo Effects
    promo_effect(merged_data)
    
   #merge whole data
    merged_data = merge_data(train_data, test_data, store_data)

    # Clean datasets
    cleaned_data = clean_data(merged_data)

    # Handle missing values
    cleaned_data = handle_missing_values(cleaned_data)

    # Feature Engineering
    feature_engineered_data = feature_engineering(cleaned_data)

    logging.info("Task 2: Starting Model Training and Evaluation")
    
    # Train models (Random Forest, XGBoost)
    rf_model, xgb_model, X_test, y_test = train_models(feature_engineered_data)

    # Evaluate models
    logging.info("Evaluating Random Forest Model")
    y_pred_rf = rf_model.predict(X_test)
    mse_rf, mae_rf, r2_rf = evaluate_model(y_test, y_pred_rf)
    logging.info(f"Random Forest - MSE: {mse_rf}, MAE: {mae_rf}, R2: {r2_rf}")

        # Plot confidence intervals for Random Forest predictions
    plot_rf_confidence_interval(rf_model, X_test, y_test, y_pred_rf)

    logging.info("Evaluating XGBoost Model")
    y_pred_xgb = xgb_model.predict(X_test)
    mse_xgb, mae_xgb, r2_xgb = evaluate_model(y_test, y_pred_xgb)
    logging.info(f"XGBoost - MSE: {mse_xgb}, MAE: {mae_xgb}, R2: {r2_xgb}")

    #  LSTM Modeling
    logging.info("Task 3: Starting LSTM Model Training and Evaluation")
    
    df = feature_engineered_data
    lstm_model, y_test_lstm_rescaled, y_pred_lstm_rescaled = train_lstm_model(df)

    # Evaluate LSTM model
    evaluate_lstm_model(y_test_lstm_rescaled, y_pred_lstm_rescaled)


    # Save Models
    logging.info("Task 2: Saving Models")
    save_model(rf_model, 'random_forest_model')
    save_model(xgb_model, 'xgboost_model')
    save_model(lstm_model, 'lstm_model')

    logging.info("Main pipeline completed successfully.")


if __name__ == "__main__":
    main()