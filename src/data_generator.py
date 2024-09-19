import pandas as pd
import numpy as np
import random

# Initialize parameters for synthetic data
start_date = '2021-01-01'
end_date = '2023-12-31'
num_stores = 10  # Number of stores to simulate
np.random.seed(42)

# Generate date range
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Initialize an empty DataFrame to hold the generated data
data = pd.DataFrame({
    'Date': np.tile(date_range, num_stores),
    'Store': np.repeat(np.arange(1, num_stores + 1), len(date_range))
})

# Simulate store type (e.g., pharmacy, convenience, supermarket)
store_types = ['pharmacy', 'convenience', 'supermarket']
data['Store_Type'] = np.random.choice(store_types, size=len(data))

# Simulate store open/closed status
data['Store_Status'] = np.random.choice([0, 1], p=[0.1, 0.9], size=len(data))

# Simulate promotions (binary: active or not active)
data['Promo'] = np.random.choice([0, 1], p=[0.7, 0.3], size=len(data))

# Simulate secondary promotion (Promo2)
data['Promo2'] = np.random.choice([0, 1], p=[0.85, 0.15], size=len(data))

# Simulate School Holidays (assuming these are more frequent in summer or near holiday periods)
school_holiday_dates = pd.date_range(start='2021-06-01', end='2021-08-31').union(
    pd.date_range(start='2022-06-01', end='2022-08-31')).union(
    pd.date_range(start='2023-06-01', end='2023-08-31'))

data['School_Holiday'] = data['Date'].isin(school_holiday_dates).astype(int)

# Simulate customer numbers (higher during promo, lower during school holidays, some noise added)
data['Customers'] = np.where(data['Store_Status'] == 0, 0, 
                             (np.random.poisson(lam=100, size=len(data)) + 
                             np.random.normal(0, 10, len(data)) * data['Promo'] - 
                             data['School_Holiday'] * 10).astype(int))

# Simulate sales (higher with more customers, promo, store type effects)
store_type_effect = {'pharmacy': 1.1, 'convenience': 0.9, 'supermarket': 1.2}
data['Sales'] = np.where(data['Store_Status'] == 0, 0, 
                         (data['Customers'] * 10 * data['Promo'] * 
                          data['Store_Type'].map(store_type_effect) + 
                          np.random.normal(0, 100, len(data))).astype(int))

# Clip sales to ensure no negative values
data['Sales'] = data['Sales'].clip(lower=0)

# Preview the first few rows of the generated data
print(data.head())

# Save to CSV if needed
data.to_csv('synthetic_sales_data.csv', index=False)
