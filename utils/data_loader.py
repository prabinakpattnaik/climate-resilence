import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Constants for data paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

RAINFALL_PATH = os.path.join(DATA_DIR, "hyderabad_rainfall_data.csv")
TEMP_PATH = os.path.join(DATA_DIR, "hyderabad_temperature.csv")
CROP_PATH = os.path.join(DATA_DIR, "crop_yield_india.csv")

def load_rainfall_data():
    """
    Load and preprocess rainfall data.
    Returns:
        X (DataFrame): Features
        y (Series): Target (rainfall)
    """
    if not os.path.exists(RAINFALL_PATH):
        raise FileNotFoundError(f"Rainfall data not found at {RAINFALL_PATH}")
    
    df = pd.read_csv(RAINFALL_PATH)
    
    # Melt to monthly format
    months = ['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 
              'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    
    records = []
    for _, row in df.iterrows():
        for i, month in enumerate(months):
            if month in row:
                records.append({
                    'year': row['Year'],
                    'month': i + 1,
                    'rainfall': row[month]
                })
    
    monthly_df = pd.DataFrame(records)
    monthly_df = monthly_df.sort_values(['year', 'month']).reset_index(drop=True)
    
    # Feature Engineering
    # Lagged features
    monthly_df['lag_1'] = monthly_df['rainfall'].shift(1)
    monthly_df['lag_2'] = monthly_df['rainfall'].shift(2)
    monthly_df['lag_3'] = monthly_df['rainfall'].shift(3)
    monthly_df['lag_12'] = monthly_df['rainfall'].shift(12)
    
    # Cyclical month encoding
    monthly_df['month_sin'] = np.sin(2 * np.pi * monthly_df['month'] / 12)
    monthly_df['month_cos'] = np.cos(2 * np.pi * monthly_df['month'] / 12)
    
    # Rolling averages
    # Fix Leakage: Must use shifted data, otherwise it includes current month!
    monthly_df['rolling_3'] = monthly_df['rainfall'].shift(1).rolling(3).mean()
    
    monthly_df = monthly_df.dropna()
    
    features = ['lag_1', 'lag_2', 'lag_3', 'lag_12', 'month_sin', 'month_cos', 'rolling_3']
    target = 'rainfall'
    
    return monthly_df[features], monthly_df[target]

def load_drought_data():
    """
    Load and preprocess drought data (using rainfall data source).
    Returns:
        X (DataFrame): Features
        y (Series): Target (drought_score)
    """
    if not os.path.exists(RAINFALL_PATH):
        raise FileNotFoundError("Rainfall data not found for drought model")
        
    df = pd.read_csv(RAINFALL_PATH)
    
    # Melt to monthly format (Same as rainfall)
    months = ['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 
              'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    
    records = []
    for _, row in df.iterrows():
        for i, month in enumerate(months):
            if month in row:
                records.append({
                    'Year': row['Year'],
                    'Month': i + 1,
                    'Rainfall': row[month]
                })
    
    df_melted = pd.DataFrame(records)
    df_melted = df_melted.sort_values(['Year', 'Month']).reset_index(drop=True)
    
    # --- Feature Engineering specific to Drought ---
    
    # Rolling averages
    df_melted['rolling_3mo'] = df_melted['Rainfall'].rolling(window=3, min_periods=1).mean()
    df_melted['rolling_6mo'] = df_melted['Rainfall'].rolling(window=6, min_periods=1).mean()
    
    # Calculate normal rainfall per month (long-term average)
    monthly_normals = df_melted.groupby('Month')['Rainfall'].transform('mean')
    df_melted['normal_rainfall'] = monthly_normals
    
    # Deficit percentage
    df_melted['deficit_pct'] = (df_melted['normal_rainfall'] - df_melted['Rainfall']) / df_melted['normal_rainfall'] * 100
    df_melted['deficit_pct'] = df_melted['deficit_pct'].clip(lower=0, upper=100)
    
    # Previous year's drought indicator (lag 12)
    df_melted['prev_year_drought'] = df_melted['deficit_pct'].shift(12)
    
    # Monsoon strength (approximate)
    df_melted['rolling_4mo_sum'] = df_melted['Rainfall'].rolling(window=4, min_periods=1).sum()
    monsoon_normal = df_melted[df_melted['Month'].isin([6,7,8,9])]['Rainfall'].mean() * 4
    df_melted['monsoon_strength'] = df_melted['rolling_4mo_sum'] / monsoon_normal
    df_melted['monsoon_strength'] = df_melted['monsoon_strength'].clip(upper=2.0)
    
    # Target: Drought Score
    # Logic: If rainfall is 0 when normal is 0, score is 0. Else use deficit.
    def calculate_score(row):
        if row['normal_rainfall'] == 0: return 0
        deficit = (row['normal_rainfall'] - row['Rainfall']) / row['normal_rainfall'] * 100
        return max(0, min(100, deficit))

    df_melted['drought_score'] = df_melted.apply(calculate_score, axis=1)
    
    df_melted = df_melted.dropna()
    
    features = ['rolling_3mo', 'rolling_6mo', 'deficit_pct', 'prev_year_drought', 'monsoon_strength']
    target = 'drought_score'
    
    return df_melted[features], df_melted[target]

def load_heatwave_data():
    """
    Load and preprocess temperature data for heatwave prediction.
    Returns:
        X (DataFrame): Features
        y (Series): Target (is_heatwave)
    """
    if not os.path.exists(TEMP_PATH):
        raise FileNotFoundError(f"Temperature data not found at {TEMP_PATH}")
        
    df = pd.read_csv(TEMP_PATH, parse_dates=["date"])
    
    # Feature Engineering
    df['month'] = df['date'].dt.month
    
    # Lags
    df['temp_max_lag1'] = df['temp_max'].shift(1)
    df['temp_max_lag2'] = df['temp_max'].shift(2)
    df['temp_max_lag3'] = df['temp_max'].shift(3)
    
    # Rolling
    df['temp_max_7day_avg'] = df['temp_max'].rolling(window=7).mean()
    
    # Cyclical
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Anomaly
    monthly_normals = df.groupby('month')['temp_max'].transform('mean')
    df['temp_anomaly'] = df['temp_max'] - monthly_normals
    
    # Target: Heatwave Definition
    # Max temp >= 40 AND anomaly >= 4.5  (Strict definition) OR just > 40 for simplicity in some contexts, 
    # but sticking to previous definition: >= 40 OR >= 4.5 above normal
    df['is_heatwave'] = ((df['temp_max'] >= 40) | (df['temp_anomaly'] >= 4.5)).astype(int)
    
    # Handle humidity
    df['humidity'] = df['humidity'].fillna(df['humidity'].median())
    
    df = df.dropna()
    
    features = ['temp_max_lag1', 'temp_max_lag2', 'temp_max_lag3', 
                'temp_max_7day_avg', 'humidity', 'month_sin', 'month_cos', 'month']
    target = 'is_heatwave'
    
    return df[features], df[target]

def load_crop_data():
    """
    Load and preprocess crop data.
    Returns:
        X (DataFrame): Features (including encoded ones)
        y (Series): Target (yield_deviation_pct)
        encoders (dict): Dictionary of fitted label encoders
    """
    if not os.path.exists(CROP_PATH):
        raise FileNotFoundError(f"Crop data not found at {CROP_PATH}")
        
    crop_df = pd.read_csv(CROP_PATH)
    
    # Filter targets
    target_states = ['Andhra Pradesh', 'Telangana', 'Karnataka']
    crop_df = crop_df[crop_df['State'].isin(target_states)]
    major_crops = ['Rice', 'Cotton(lint)', 'Maize', 'Groundnut', 'Sugarcane', 'Jowar']
    crop_df = crop_df[crop_df['Crop'].isin(major_crops)]
    
    # Target Calculation
    avg_yield_per_crop = crop_df.groupby('Crop')['Yield'].transform('mean')
    crop_df['yield_deviation_pct'] = ((avg_yield_per_crop - crop_df['Yield']) / avg_yield_per_crop * 100).clip(0, 80)
    
    # Features
    crop_df['rainfall'] = crop_df['Annual_Rainfall']
    crop_df['fertilizer_per_area'] = crop_df['Fertilizer'] / (crop_df['Area'] + 1)
    crop_df['pesticide_per_area'] = crop_df['Pesticide'] / (crop_df['Area'] + 1)
    
    avg_rainfall = crop_df['rainfall'].mean()
    crop_df['rainfall_anomaly'] = ((crop_df['rainfall'] - avg_rainfall) / avg_rainfall * 100)
    
    # Encoding
    from sklearn.preprocessing import LabelEncoder
    le_crop = LabelEncoder()
    le_state = LabelEncoder()
    le_season = LabelEncoder()
    
    crop_df['crop_encoded'] = le_crop.fit_transform(crop_df['Crop'])
    crop_df['state_encoded'] = le_state.fit_transform(crop_df['State'])
    crop_df['season_encoded'] = le_season.fit_transform(crop_df['Season'].str.strip())
    
    # Clean
    crop_df = crop_df.dropna(subset=['Yield', 'rainfall', 'yield_deviation_pct'])
    crop_df = crop_df[crop_df['Yield'] > 0]
    
    features = ['rainfall', 'rainfall_anomaly', 'fertilizer_per_area', 
                'pesticide_per_area', 'crop_encoded', 'state_encoded', 'season_encoded']
    target = 'yield_deviation_pct'
    
    encoders = {
        'crop': le_crop,
        'state': le_state,
        'season': le_season
    }
    
    return crop_df[features], crop_df[target], encoders
