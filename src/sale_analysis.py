import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

log_dir = os.path.join(os.path.dirname(__file__), '../logs')
from data_processing import load_data, clean_data, detect_outliers
from custom_logging import info_logger, error_logger

import matplotlib.pyplot as plt
import seaborn as sns
import logging
sys.path.append(os.path.abspath('../data'))
from data_processing import load_data

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the 'plots' directory exists
plots_dir = os.path.join(os.path.dirname(__file__), '../notebook/plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Load datasets
train_data = load_data('../data/train.csv')
test_data = load_data('../data/test.csv')
store_data = load_data('../data/store.csv')

def plot_sales_distribution(data_train, data_test):
    """ Compare sales distribution between training and test sets if available. """
    
    # Check if 'Sales' column exists in the training set
    if 'Sales' not in data_train.columns:
        raise KeyError("The 'Sales' column is missing from the training data.")
    
    # Plot distribution for the training set
    plt.figure(figsize=(10, 6))
    sns.histplot(data_train['Sales'], bins=30, color='blue', label='Train', kde=True)
    
    # Test data does not have 'Sales', so only plot if available
    if 'Sales' in data_test.columns:
        sns.histplot(data_test['Sales'], bins=30, color='orange', label='Test', kde=True)
    else:
        logger.warning("The 'Sales' column is missing from the test data. Only training data will be plotted.")

    plt.title('Sales Distribution: Train vs Test')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'sales_distribution_comparison.png'))
    logger.info('Sales distribution comparison plot saved as sales_distribution_comparison.png')


def compare_sales_holidays(data):
    """ Compare sales behavior during holidays. """
    holiday_sales = data[data['StateHoliday'] != 0]['Sales'].mean()  # Non-zero indicates holidays
    non_holiday_sales = data[data['StateHoliday'] == 0]['Sales'].mean()
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=['Holidays', 'Non-Holidays'], y=[holiday_sales, non_holiday_sales])
    plt.title('Sales Before, During, and After Holidays')
    plt.ylabel('Average Sales')
    plt.savefig(os.path.join(plots_dir,'holiday_sales_comparison.png'))
    info_logger.info('Holiday sales comparison plot saved as holiday_sales_comparison.png')

def seasonal_behavior(data):
    """ Check for seasonal purchase behaviors. """
    data['Date'] = pd.to_datetime(data['Date'])  # Ensure the Date column is datetime
    data['Month'] = data['Date'].dt.month
    seasonal_sales = data.groupby('Month')['Sales'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Month', y='Sales', data=seasonal_sales, marker='o')
    plt.title('Average Sales by Month')
    plt.ylabel('Average Sales')
    plt.savefig(os.path.join(plots_dir,'seasonal_sales.png'))
    info_logger.info(os.path.join(plots_dir,'Seasonal sales plot saved as seasonal_sales.png'))

def correlation_analysis(data):
    """ Analyze correlation between sales and number of customers. """
    correlation = data['Sales'].corr(data['Customers'])
    info_logger.info(f'Correlation between Sales and Customers: {correlation}')
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Customers', y='Sales', data=data)
    plt.title('Sales vs Number of Customers')
    plt.xlabel('Number of Customers')
    plt.ylabel('Sales')
    plt.savefig(os.path.join(plots_dir,'sales_customers_correlation.png'))
    info_logger.info('Sales vs Customers correlation plot saved as sales_customers_correlation.png')

def plot_promo_distribution():
    """Promo distribution in train vs test sets."""
    plt.figure(figsize=(10, 6))
    sns.histplot(train_data['Promo'], color='blue', label='Train', kde=True, bins=30)
    sns.histplot(test_data['Promo'], color='green', label='Test', kde=True, bins=30)
    plt.title('Promo Distribution: Training vs Test Set')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'promo_distribution.png'))
    plt.show()

def plot_sales_during_holidays():
    """Analyze sales behavior during holidays."""
    holiday_sales = merged_data.groupby(['StateHoliday', 'Date']).agg({'Sales': 'mean'}).reset_index()
    sns.lineplot(x='Date', y='Sales', hue='StateHoliday', data=holiday_sales)
    plt.title('Sales Behavior During Holidays')
    plt.savefig(os.path.join(plots_dir, 'sales_during_holidays.png'))
    plt.show()

def plot_sales_customers_corr():
    """Correlation between sales and customers."""
    correlation = train_data[['Sales', 'Customers']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation between Sales and Customers')
    plt.savefig(os.path.join(plots_dir, 'sales_customers_corr.png'))
    plt.show()

def plot_store_corr():
    """Correlation matrix for store data."""
    correlation_matrix = store_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation between features")
    plt.savefig(os.path.join(plots_dir, 'store_feature_correlation.png'))
    plt.show()

def promo_effect(data):
    """ Analyze the effect of promos on sales. """
    promo_sales = data[data['Promo'] == 1]['Sales'].mean()
    non_promo_sales = data[data['Promo'] == 0]['Sales'].mean()
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=['Promotions', 'No Promotions'], y=[promo_sales, non_promo_sales])
    plt.title('Sales with and without Promotions')
    plt.ylabel('Average Sales')
    plt.savefig(os.path.join(plots_dir,'promo_effect.png'))
    info_logger.info('Promo effect plot saved as promo_effect.png')

def effective_promo_deployment(data):
    """ Analyze effective promo deployment per store. """
    store_promo_effect = data.groupby('Store')['Sales'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Store', y='Sales', data=store_promo_effect)
    plt.title('Average Sales by Store')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(plots_dir,'store_promo_effect.png'))
    info_logger.info('Store promo effect plot saved as store_promo_effect.png')

def customer_behavior_trends(data):
    """ Analyze trends of customer behavior during store hours. """
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Open', y='Sales', data=data)
    plt.title('Sales Trends Based on Store Opening Status')
    plt.ylabel('Sales')
    plt.savefig(os.path.join(plots_dir,'customer_behavior_trends.png'))
    info_logger.info('Customer behavior trends plot saved as customer_behavior_trends.png')

def weekday_openings(data):
    """ Analyze which stores are open on all weekdays. """
    weekday_open = data[data['Open'] == 1].groupby('Store').size()
    open_stores = weekday_open[weekday_open >= 5].index.tolist()  # Assuming 5 weekdays
    info_logger.info(f'Stores open on all weekdays: {open_stores}')

def assortment_type_impact(data):
    """ Analyze the impact of assortment type on sales. """
    assortment_sales = data.groupby('Assortment')['Sales'].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Assortment', y='Sales', data=assortment_sales)
    plt.title('Average Sales by Assortment Type')
    plt.ylabel('Average Sales')
    plt.savefig(os.path.join(plots_dir,'assortment_impact.png'))
    info_logger.info('Assortment impact plot saved as assortment_impact.png')

def competitor_distance_impact(data):
    """ Analyze the impact of competitor distance on sales. """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=data)
    plt.title('Sales vs Competitor Distance')
    plt.xlabel('Distance to Competitor')
    plt.ylabel('Sales')
    plt.savefig(os.path.join(plots_dir,'competitor_distance_impact.png'))
    info_logger.info('Competitor distance impact plot saved as competitor_distance_impact.png')

def new_competitor_effects(data):
    """ Analyze the effects of new competitors on sales. """
    new_competitors = data[data['CompetitionDistance'].isna()].copy()
    new_competitors['CompetitionDistance'] = new_competitors['CompetitionDistance'].fillna(method='ffill')
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Date', y='Sales', data=new_competitors)
    plt.title('Sales Trends After New Competitors')
    plt.ylabel('Sales')
    plt.savefig(os.path.join(plots_dir,'new_competitor_effects.png'))
    info_logger.info('New competitor effects plot saved as new_competitor_effects.png')