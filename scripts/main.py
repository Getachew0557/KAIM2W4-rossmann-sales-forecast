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

def main():
    # Load datasets
    train_data = load_data('../data/train.csv')
    test_data = load_data('../data/test.csv')
    store_data = load_data('../data/store.csv')

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

    # Correlation analysis between features
    plt.figure(figsize=(12, 8))
    sns.heatmap(store_data.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation between features")
    plt.show()

    # Promo Effects
    promo_effect(merged_data)




