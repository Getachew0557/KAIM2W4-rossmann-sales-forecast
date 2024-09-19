import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import holidays


# Define the path to the logs directory one level up
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')

# Create the logs directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Define file paths
log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

# Create handlers
info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)

# Create a formatter and set it for the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

# Create a logger and set its level
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Capture all info and above
logger.addHandler(info_handler)
logger.addHandler(error_handler)

# Optional: Add a console handler if you want to see logs in the console
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)  # Or ERROR if you want only error messages in the console
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)


# Functions

def load_data(file_path):
    logger.info("Loading data from file...")
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        logger.info(f"Data loaded with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def plot_weekly_sales(df):
    logger.info("Plotting weekly sales...")
    try:
        weekly_sales = df['Sales'].resample('W').sum()
        plt.figure(figsize=(15, 7))
        plt.plot(weekly_sales.index, weekly_sales)
        plt.title('Weekly Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting weekly sales: {e}")

def plot_monthly_sales(df):
    logger.info("Plotting monthly sales...")
    try:
        monthly_sales = df['Sales'].resample('M').sum()
        plt.figure(figsize=(15, 7))
        plt.plot(monthly_sales.index, monthly_sales)
        plt.title('Monthly Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting monthly sales: {e}")

def seasonal_decomposition(df):
    logger.info("Performing seasonal decomposition...")
    try:
        monthly_sales = df['Sales'].resample('M').sum()
        result = seasonal_decompose(monthly_sales, model='additive')
        result.plot()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Error during seasonal decomposition: {e}")

def plot_acf_pacf(df):
    logger.info("Plotting ACF and PACF...")
    try:
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
    except Exception as e:
        logger.error(f"Error in plotting ACF and PACF: {e}")

def plot_rolling_statistics(df):
    logger.info("Plotting rolling statistics...")
    try:
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
    except Exception as e:
        logger.error(f"Error in plotting rolling statistics: {e}")

def plot_day_of_week_sales(df):
    logger.info("Plotting average sales by day of week...")
    try:
        df['DayOfWeek'] = df.index.dayofweek
        day_of_week_sales = df.groupby('DayOfWeek')['Sales'].mean()

        plt.figure(figsize=(10, 6))
        day_of_week_sales.plot(kind='bar')
        plt.title('Average Sales by Day of Week')
        plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
        plt.ylabel('Average Sales')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting day of week sales: {e}")

def add_holiday_column(df):
    logger.info("Adding holiday column...")
    try:
        us_holidays = holidays.US()
        df['is_holiday'] = df.index.to_series().apply(lambda date: date in us_holidays).astype(int)
        return df
    except Exception as e:
        logger.error(f"Error in adding holiday column: {e}")
        return df

def plot_holiday_sales_distribution(df):
    logger.info("Plotting sales distribution: Holiday vs Non-Holiday...")
    try:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='is_holiday', y='Sales', data=df)
        plt.title('Sales Distribution: Holiday vs Non-Holiday')
        plt.xticks([0, 1], ['Non-Holiday', 'Holiday'])
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting holiday sales distribution: {e}")

def print_statistics(df):
    logger.info("Printing summary statistics...")
    try:
        print(df.groupby('DayOfWeek')['Sales'].describe())
        print("\nHoliday vs Non-Holiday Sales:")
        print(df.groupby('is_holiday')['Sales'].describe())
    except Exception as e:
        logger.error(f"Error in printing statistics: {e}")

def plot_holiday_effect(df):
    logger.info("Plotting holiday effect...")
    try:
        df['IsHoliday'] = df['is_holiday'] | (df.index.month == 12)
        holiday_effect = df.groupby('IsHoliday')['Sales'].mean()
        holiday_effect.plot(kind='bar', figsize=(10, 6))
        plt.title('Average Sales: Holiday vs Non-Holiday')
        plt.ylabel('Average Sales')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting holiday effect: {e}")

def plot_promo_effect(df):
    logger.info("Plotting promo effect over time...")
    try:
        monthly_promo_sales = df.groupby([df.index.to_period('M'), 'Promo'])['Sales'].mean().unstack()
        monthly_promo_sales.columns = ['No Promo', 'Promo']

        monthly_promo_sales[['No Promo', 'Promo']].plot(figsize=(15, 7))
        plt.title('Monthly Average Sales: Promo vs No Promo')
        plt.xlabel('Date')
        plt.ylabel('Average Sales')
        plt.legend(['No Promo', 'Promo'])
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting promo effect: {e}")

def plot_store_type_performance(df):
    logger.info("Plotting store type performance over time...")
    try:
        store_type_sales = df.groupby([df.index.to_period('M'), 'Store_Type'])['Sales'].mean().unstack()
        store_type_sales.plot(figsize=(15, 7))
        plt.title('Monthly Average Sales by Store Type')
        plt.xlabel('Date')
        plt.ylabel('Average Sales')
        plt.legend(title='Store Type')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting store type performance: {e}")

def plot_sales_vs_customers(df):
    logger.info("Plotting sales vs customers scatter plot...")
    try:
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(df['Customers'], df['Sales'], c=df.index, cmap='viridis')
        plt.colorbar(scatter, label='Date')
        plt.title('Sales vs Customers Over Time')
        plt.xlabel('Number of Customers')
        plt.ylabel('Sales')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting sales vs customers: {e}")

def plot_sales_correlation(df):
    logger.info("Plotting correlation between sales and customers...")
    try:
        correlation = df[['Sales', 'Customers']].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation between Sales and Customers')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting sales correlation: {e}")

def plot_store_sales(df):
    logger.info("Plotting total sales per store...")
    try:
        store_sales = df.groupby('Store')['Sales'].sum()
        plt.figure(figsize=(15, 7))
        store_sales.plot(kind='bar')
        plt.title('Total Sales per Store')
        plt.xlabel('Store')
        plt.ylabel('Total Sales')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting store sales: {e}")

def plot_sales_heatmap(df):
    logger.info("Plotting sales heatmap by day and month...")
    try:
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        sales_pivot = df.pivot_table(values='Sales', index='Day', columns='Month', aggfunc='sum')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(sales_pivot, cmap='YlGnBu', annot=False)
        plt.title('Sales Heatmap by Day and Month')
        plt.xlabel('Month')
        plt.ylabel('Day')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting sales heatmap: {e}")
def plot_sales_vs_promo(df):
    logger.info("Plotting sales vs promotions...")
    try:
        monthly_promo_sales = df.groupby([df.index.to_period('M'), 'Promo'])['Sales'].mean().unstack()
        monthly_promo_sales.columns = ['No Promo', 'Promo']
        
        plt.figure(figsize=(15, 7))
        monthly_promo_sales[['No Promo', 'Promo']].plot()
        plt.title('Monthly Average Sales: Promo vs No Promo')
        plt.xlabel('Date')
        plt.ylabel('Average Sales')
        plt.legend(['No Promo', 'Promo'])
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting sales vs promotions: {e}")

def plot_store_type_performance(df):
    logger.info("Plotting store type performance over time...")
    try:
        store_type_sales = df.groupby([df.index.to_period('M'), 'Store_Type'])['Sales'].mean().unstack()
        plt.figure(figsize=(15, 7))
        store_type_sales.plot()
        plt.title('Monthly Average Sales by Store Type')
        plt.xlabel('Date')
        plt.ylabel('Average Sales')
        plt.legend(title='Store Type')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting store type performance: {e}")

def plot_cumulative_sales(df):
    logger.info("Plotting cumulative sales over time...")
    try:
        df['CumulativeSales'] = df['Sales'].cumsum()
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['CumulativeSales'])
        plt.title('Cumulative Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Sales')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting cumulative sales: {e}")

def plot_sales_growth_rate(df):
    logger.info("Plotting daily sales growth rate...")
    try:
        df['SalesGrowthRate'] = df['Sales'].pct_change()
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['SalesGrowthRate'])
        plt.title('Daily Sales Growth Rate')
        plt.xlabel('Date')
        plt.ylabel('Growth Rate')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting sales growth rate: {e}")

def clean_data(df):
    logger.info("Cleaning data by removing specific features...")
    try:
        features_to_remove = ['MA30', 'SD30', 'SalesGrowthRate', 'IsHoliday', 'CumulativeSales']
        df_cleaned = df.drop(columns=features_to_remove, errors='ignore')
        logger.info(f"Data cleaned. Remaining columns: {df_cleaned.columns.tolist()}")
        return df_cleaned
    except Exception as e:
        logger.error(f"Error in cleaning data: {e}")
        return df

def correlation_analysis(df):
    logger.info("Performing correlation analysis...")
    try:
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

        logger.info("Correlations with Sales:")
        logger.info(correlations[top_features])
    except Exception as e:
        logger.error(f"Error in performing correlation analysis: {e}")


# # Example of using the functions
# if __name__ == "__main__":
#     data_file = "sales_data.csv"  # Example file path
#     df = load_data(data_file)

#     if df is not None:
#         df = add_holiday_column(df)
#         plot_weekly_sales(df)
#         plot_monthly_sales(df)
#         seasonal_decomposition(df)
#         plot_acf_pacf(df)
#         plot_rolling_statistics(df)
#         plot_day_of_week_sales(df)
#         plot_holiday_sales_distribution(df)
#         print_statistics(df)
#         plot_holiday_effect(df)
#         plot_promo_effect(df)
#         plot_store_type_performance(df)
#         plot_sales_vs_customers(df)
#         plot_sales_correlation(df)
#         plot_store_sales(df)
        # plot_sales_heatmap(df)