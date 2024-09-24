import os
import logging
import pandas as pd
import numpy as np
<<<<<<< HEAD
=======
from sklearn.preprocessing import StandardScaler
>>>>>>> 2bb416d (Re-initialize repository with current changes)

# Ensure the 'logs' directory exists
log_dir = os.path.join(os.path.dirname(__file__), '../logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

from custom_logging import info_logger, error_logger

def load_data(file_path):
    """ Load dataset from a given file path """
    try:
        data = pd.read_csv(file_path)
        info_logger.info(f'Data loaded successfully from {file_path}')
        return data
    except FileNotFoundError as e:
        error_logger.error(f"File not found: {file_path}")
        raise e

def clean_data(data):
    """ Perform data cleaning """
    data = data.dropna()
    info_logger.info('Data cleaned successfully')
    return data

def detect_outliers(data, column):
    """ Detect outliers using the IQR method """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    info_logger.info(f'Detected {len(outliers)} outliers in {column}')
    return outliers

def handle_missing_values(merged_data):
    """ Handle missing values in the merged dataset """
    try:
        merged_data['CompetitionDistance'].fillna(merged_data['CompetitionDistance'].median(), inplace=True)
        merged_data['CompetitionOpenSinceMonth'].fillna(merged_data['CompetitionOpenSinceMonth'].mode()[0], inplace=True)
        merged_data['CompetitionOpenSinceYear'].fillna(merged_data['CompetitionOpenSinceYear'].mode()[0], inplace=True)
        merged_data['Promo2SinceWeek'].fillna(merged_data['Promo2SinceWeek'].mode()[0], inplace=True)
        merged_data['Promo2SinceYear'].fillna(merged_data['Promo2SinceYear'].mode()[0], inplace=True)
        merged_data['PromoInterval'].fillna('None', inplace=True)
        logging.info('Missing values handled successfully')
    except Exception as e:
        logging.error(f"Error handling missing values: {str(e)}")
        raise e
<<<<<<< HEAD
    return merged_data
=======
    return merged_data

def extract_features(data):
    """ Extract useful features from the dataset. """
    data['Date'] = pd.to_datetime(data['Date'])
    data['Weekday'] = data['Date'].dt.weekday
    data['Weekend'] = (data['Weekday'] >= 5).astype(int)
    data['Days_to_Holiday'] = (data['Date'].dt.date - data['Date'].dt.date.apply(lambda x: pd.offsets.Day(x.day).rollback(x))).dt.days
    data['Days_After_Holiday'] = (data['Date'].dt.date - data['Date'].dt.date.apply(lambda x: pd.offsets.Day(x.day).rollforward(x))).dt.days
    data['Beginning_of_Month'] = (data['Date'].dt.day <= 7).astype(int)
    data['Mid_Month'] = ((data['Date'].dt.day > 7) & (data['Date'].dt.day <= 14)).astype(int)
    data['End_of_Month'] = (data['Date'].dt.day > 14).astype(int)

    info_logger.info('Features extracted successfully.')
    return data

def scale_data(data):
    """ Scale the dataset using Standard Scaler. """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.select_dtypes(include=[np.number]))
    scaled_df = pd.DataFrame(scaled_data, columns=data.select_dtypes(include=[np.number]).columns)
    scaled_df = pd.concat([data.drop(columns=data.select_dtypes(include=[np.number]).columns), scaled_df], axis=1)

    info_logger.info('Data scaled successfully.')
    return scaled_df
>>>>>>> 2bb416d (Re-initialize repository with current changes)
