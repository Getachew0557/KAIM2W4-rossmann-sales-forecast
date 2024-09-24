import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(train_file, test_file, store_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    store_data = pd.read_csv(store_file)
    return train_data, test_data, store_data

def merge_data(train_data, test_data, store_data):
    train_merged = pd.merge(train_data, store_data, on='Store', how='left')
    test_merged = pd.merge(test_data, store_data, on='Store', how='left')
    df = pd.concat([train_merged, test_merged], axis=0)
    
    # Fill missing values
    df['Sales'].fillna(0, inplace=True)
    df['Customers'].fillna(0, inplace=True)
    df['Open'].fillna(0, inplace=True)
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].max() + 1, inplace=True)
    df['CompetitionOpenSinceMonth'].fillna(1, inplace=True)
    df['CompetitionOpenSinceYear'].fillna(2000, inplace=True)
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)
    df['PromoInterval'].fillna('No Promo', inplace=True)
    
    # Convert datetime column
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract features
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    df['IsBeginningOfMonth'] = df['Day'].apply(lambda x: 1 if x <= 7 else 0)
    df['IsMidMonth'] = df['Day'].apply(lambda x: 1 if 8 <= x <= 21 else 0)
    df['IsEndOfMonth'] = df['Day'].apply(lambda x: 1 if x > 21 else 0)

    # Label encoding
    categorical_features = ['StoreType', 'Assortment', 'PromoInterval']
    label_encoder = LabelEncoder()
    for col in categorical_features:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

    # Scale features
    scaler = StandardScaler()
    scaled_columns = ['CompetitionDistance', 'Day', 'WeekOfYear', 'Month', 'Year']
    df[scaled_columns] = scaler.fit_transform(df[scaled_columns])
    
    return df
