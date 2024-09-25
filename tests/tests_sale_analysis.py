import os
import sys
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from sale_analysis import (
    plot_sales_distribution,
    compare_sales_holidays,
    seasonal_behavior,
    correlation_analysis,
    plot_promo_distribution,
)

@pytest.fixture
def sample_train_data():
    """Fixture to provide sample training data for testing."""
    return pd.DataFrame({
        'Sales': [500, 600, 700, 800, 900],
        'Customers': [50, 60, 70, 80, 90],
        'StateHoliday': [0, 1, 1, 0, 1],
        'Date': pd.date_range(start='2024-01-01', periods=5),
        'Promo': [1, 0, 1, 0, 1],
    })

@pytest.fixture
def sample_test_data():
    """Fixture to provide sample testing data for testing."""
    return pd.DataFrame({
        'Sales': [550, 650, 750, None, 950],
        'Customers': [55, 65, 75, None, 95],
        'StateHoliday': [0, 1, 1, 0, 1],
        'Date': pd.date_range(start='2024-01-06', periods=5),
        'Promo': [1, 1, 0, 0, 1],
    })

# Mock the load_data function to return sample data instead of reading a CSV
@patch('src.data_processing.load_data')
def test_plot_sales_distribution(mock_load_data, sample_train_data, sample_test_data):
    """Test sales distribution plotting."""
    mock_load_data.side_effect = lambda file_path: sample_train_data if 'train.csv' in file_path else sample_test_data
    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        plot_sales_distribution(sample_train_data, sample_test_data)
        mock_savefig.assert_called_once_with(os.path.join('notebook/plots', 'sales_distribution_comparison.png'))

@patch('src.data_processing.load_data')
def test_compare_sales_holidays(mock_load_data, sample_train_data):
    """Test comparison of sales during holidays."""
    mock_load_data.return_value = sample_train_data
    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        compare_sales_holidays(sample_train_data)
        mock_savefig.assert_called_once_with(os.path.join('notebook/plots', 'holiday_sales_comparison.png'))

@patch('src.data_processing.load_data')
def test_seasonal_behavior(mock_load_data, sample_train_data):
    """Test seasonal behavior plotting."""
    mock_load_data.return_value = sample_train_data
    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        seasonal_behavior(sample_train_data)
        mock_savefig.assert_called_once_with(os.path.join('notebook/plots', 'seasonal_sales.png'))

@patch('src.data_processing.load_data')
def test_correlation_analysis(mock_load_data, sample_train_data):
    """Test correlation analysis plotting."""
    mock_load_data.return_value = sample_train_data
    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        correlation_analysis(sample_train_data)
        mock_savefig.assert_called_once_with(os.path.join('notebook/plots', 'sales_customers_correlation.png'))

@patch('src.data_processing.load_data')
def test_plot_promo_distribution(mock_load_data, sample_train_data, sample_test_data):
    """Test promo distribution plotting."""
    mock_load_data.side_effect = lambda file_path: sample_train_data if 'train.csv' in file_path else sample_test_data
    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        plot_promo_distribution(sample_train_data, sample_test_data)
        mock_savefig.assert_called_once_with(os.path.join('notebook/plots', 'promo_distribution.png'))

if __name__ == "__main__":
    pytest.main()
