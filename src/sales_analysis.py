import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import holidays
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    logging.info("Loading data from file...")
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    logging.info(f"Data loaded with shape {df.shape}")
    return df

def plot_weekly_sales(df):
    logging.info("Plotting weekly sales...")
    weekly_sales = df['Sales'].resample('W').sum()
    plt.figure(figsize=(15, 7))
    plt.plot(weekly_sales.index, weekly_sales)
    plt.title('Weekly Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()

def plot_monthly_sales(df):
    logging.info("Plotting monthly sales...")
    monthly_sales = df['Sales'].resample('M').sum()
    plt.figure(figsize=(15, 7))
    plt.plot(monthly_sales.index, monthly_sales)
    plt.title('Monthly Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()

def seasonal_decomposition(df):
    logging.info("Performing seasonal decomposition...")
    monthly_sales = df['Sales'].resample('M').sum()
    result = seasonal_decompose(monthly_sales, model='additive')
    result.plot()
    plt.tight_layout()
    plt.show()

def plot_acf_pacf(df):
    logging.info("Plotting ACF and PACF...")
    monthly_sales = df['Sales'].resample('M').sum()
    n_lags = len(monthly_sales) // 3
    acf_values = acf(monthly_sales.dropna(), nlags=n_lags)
    pacf_values = pacf(monthly_sales.dropna(), nlags=n_lags)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    ax1.stem(range(len(acf_values)), acf_values, use_line_collection=True)
    ax1.axhline(y=0, linestyle='--', color='gray')
    ax1.axhline(y=-1.96/np.sqrt(len(monthly_sales)), linestyle='--', color='gray')
    ax1.axhline(y=1.96/np.sqrt(len(monthly_sales)), linestyle='--', color='gray')
    ax1.set_title('Autocorrelation Function')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Correlation')

    ax2.stem(range(len(pacf_values)), pacf_values, use_line_collection=True)
    ax2.axhline(y=0, linestyle='--', color='gray')
    ax2.axhline(y=-1.96/np.sqrt(len(monthly_sales)), linestyle='--', color='gray')
    ax2.axhline(y=1.96/np.sqrt(len(monthly_sales)), linestyle='--', color='gray')
    ax2.set_title('Partial Autocorrelation Function')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Correlation')

    plt.tight_layout()
    plt.show()

def plot_rolling_statistics(df):
    logging.info("Plotting rolling statistics...")
    monthly_sales = df['Sales'].resample('M').sum()
    rolling_mean = monthly_sales.rolling(window=12).mean()
    rolling_std = monthly_sales.rolling(window=12).std()

    plt.figure(figsize=(15, 7))
    plt.plot(monthly_sales.index, monthly_sales, label='Monthly Sales')
    plt.plot(rolling_mean.index, rolling_mean, label='12-month Rolling Mean')
    plt.plot(rolling_std.index, rolling_std, label='12-month Rolling Std')
    plt.legend()
    plt.title('Monthly Sales - Rolling Mean & Standard Deviation')
    plt.show()

def plot_day_of_week_sales(df):
    logging.info("Plotting average sales by day of week...")
    df['DayOfWeek'] = df.index.dayofweek
    day_of_week_sales = df.groupby('DayOfWeek')['Sales'].mean()

    plt.figure(figsize=(10, 6))
    day_of_week_sales.plot(kind='bar')
    plt.title('Average Sales by Day of Week')
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Average Sales')
    plt.show()
def add_holiday_column(df):
    logging.info("Adding holiday column...")
    us_holidays = holidays.US()
    df['is_holiday'] = df.index.to_series().apply(lambda date: date in us_holidays).astype(int)
    return df

def plot_holiday_sales_distribution(df):
    logging.info("Plotting sales distribution: Holiday vs Non-Holiday...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='is_holiday', y='Sales', data=df)
    plt.title('Sales Distribution: Holiday vs Non-Holiday')
    plt.xticks([0, 1], ['Non-Holiday', 'Holiday'])
    plt.show()

def print_statistics(df):
    logging.info("Printing summary statistics...")
    print(df.groupby('DayOfWeek')['Sales'].describe())
    print("\nHoliday vs Non-Holiday Sales:")
    print(df.groupby('is_holiday')['Sales'].describe())

def plot_holiday_effect(df):
    logging.info("Plotting holiday effect...")
    df['IsHoliday'] = df['is_holiday'] | (df.index.month == 12)
    holiday_effect = df.groupby('IsHoliday')['Sales'].mean()
    holiday_effect.plot(kind='bar', figsize=(10, 6))
    plt.title('Average Sales: Holiday vs Non-Holiday')
    plt.ylabel('Average Sales')
    plt.show()

def plot_promo_effect(df):
    logging.info("Plotting promo effect over time...")
    monthly_promo_sales = df.groupby([df.index.to_period('M'), 'Promo'])['Sales'].mean().unstack()
    monthly_promo_sales.columns = ['No Promo', 'Promo']

    monthly_promo_sales[['No Promo', 'Promo']].plot(figsize=(15, 7))
    plt.title('Monthly Average Sales: Promo vs No Promo')
    plt.xlabel('Date')
    plt.ylabel('Average Sales')
    plt.legend(['No Promo', 'Promo'])
    plt.show()

def plot_store_type_performance(df):
    logging.info("Plotting store type performance over time...")
    store_type_sales = df.groupby([df.index.to_period('M'), 'Store_Type'])['Sales'].mean().unstack()
    store_type_sales.plot(figsize=(15, 7))
    plt.title('Monthly Average Sales by Store Type')
    plt.xlabel('Date')
    plt.ylabel('Average Sales')
    plt.legend(title='Store Type')
    plt.show()

def plot_sales_vs_customers(df):
    logging.info("Plotting sales vs customers scatter plot...")
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['Customers'], df['Sales'], c=df.index, cmap='viridis')
    plt.colorbar(scatter, label='Date')
    plt.title('Sales vs Customers Over Time')
    plt.xlabel('Number of Customers')
    plt.ylabel('Sales')
    plt.show()

def plot_sales_heatmap(df):
    logging.info("Plotting sales heatmap by day of week and month...")
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    sales_heatmap = df.pivot_table(values='Sales', index='DayOfWeek', columns='Month', aggfunc='mean')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(sales_heatmap, cmap='YlOrRd', annot=True, fmt='.0f')
    plt.title('Average Sales by Day of Week and Month')
    plt.xlabel('Month')
    plt.ylabel('Day of Week (0=Monday, 6=Sunday)')
    plt.show()

def plot_cumulative_sales(df):
    logging.info("Plotting cumulative sales over time...")
    df['CumulativeSales'] = df['Sales'].cumsum()
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['CumulativeSales'])
    plt.title('Cumulative Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Sales')
    plt.show()

def plot_sales_growth_rate(df):
    logging.info("Plotting daily sales growth rate...")
    df['SalesGrowthRate'] = df['Sales'].pct_change()
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['SalesGrowthRate'])
    plt.title('Daily Sales Growth Rate')
    plt.xlabel('Date')
    plt.ylabel('Growth Rate')
    plt.show()

def clean_data(df):
    
    logging.info("Cleaning data by removing specific features...")
    features_to_remove = ['MA30', 'SD30', 'SalesGrowthRate', 'IsHoliday', 'CumulativeSales']
    df_cleaned = df.drop(columns=features_to_remove, errors='ignore')
    logging.info(f"Data cleaned. Remaining columns: {df_cleaned.columns.tolist()}")
    return df_cleaned

def correlation_analysis(df):
    logging.info("Performing correlation analysis...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    correlations = df[numeric_columns].corr()['Sales'].abs().sort_values(ascending=False)

    top_features = correlations[1:11].index.tolist()
    f_correlation = df[top_features].corr()

    f_mask = np.triu(np.ones_like(f_correlation, dtype=bool))

    plt.figure(figsize=(12, 10))
    sns.heatmap(f_correlation, mask=f_mask, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title('Top 10 Features Correlated with Sales', fontsize=16)
    plt.tight_layout()
    plt.show()

    logging.info("Correlations with Sales:")
    logging.info(correlations[top_features])

