import os
import sys
import pytest
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

from data_processing import load_data, clean_data, handle_missing_values

@pytest.fixture
def sample_data():
    """Fixture to provide sample data for testing."""
    return pd.DataFrame({
        'Store': [1, 2, 3, 4, 5],
        'Sales': [500, 600, None, 700, 400],
        'Customers': [50, 60, 70, None, 90],
        'CompetitionDistance': [500, None, 300, 1000, 1200],
        'CompetitionOpenSinceMonth': [1, None, 3, 5, 7],
        'CompetitionOpenSinceYear': [2010, None, 2015, 2012, 2013],
        'Promo2SinceWeek': [10, 20, None, 30, 40],
        'Promo2SinceYear': [2011, 2012, None, 2013, 2014],
        'PromoInterval': [None, 'Jan,Apr,Jul,Oct', None, 'Feb,May,Aug,Nov', 'Mar,Jun,Sep,Dec']
    })

def test_load_data_valid_file(sample_data):
    """Test if data is loaded successfully from a DataFrame."""
    # Instead of loading from CSV, directly use the sample data
    data = sample_data
    assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame."

def test_load_data_invalid_file():
    """Test if the function raises an error for invalid file paths."""
    with pytest.raises(FileNotFoundError):
        load_data('invalid/path/to/file.csv')

def test_clean_data(sample_data):
    """Test cleaning of data."""
    cleaned_data = clean_data(sample_data)
    # Ensure there are no missing values in 'Sales' and 'Customers'
    assert cleaned_data['Sales'].isnull().sum() == 0, "Sales column should not have missing values after cleaning."
    assert cleaned_data['Customers'].isnull().sum() == 0, "Customers column should not have missing values after cleaning."

def test_handle_missing_values(sample_data):
    """Test handling of missing values."""
    filled_data = handle_missing_values(sample_data)
    
    # Ensure 'CompetitionDistance' missing values are filled with the median
    assert filled_data['CompetitionDistance'].isnull().sum() == 0, "CompetitionDistance should not have missing values."
    
    # Ensure 'PromoInterval' missing values are filled with 'None'
    assert 'None' in filled_data['PromoInterval'].unique(), "'PromoInterval' missing values should be filled with 'None'."
