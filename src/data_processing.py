import os
import logging
import pandas as pd
import numpy as np

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
