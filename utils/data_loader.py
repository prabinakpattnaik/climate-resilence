import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import gamma as gamma_dist
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

def _compute_spi(rainfall_series, window=3):
    """
    Compute Standardized Precipitation Index (SPI) from a rainfall series.

    SPI uses gamma distribution fitting on rolling-window accumulated rainfall,
    then transforms to standard normal space. This is the meteorological standard
    for drought quantification (McKee et al. 1993).

    Args:
        rainfall_series: pandas Series of monthly rainfall values
        window: accumulation window in months (commonly 3 or 6)

    Returns:
        pandas Series of SPI values
    """
    # Accumulate rainfall over the window
    accumulated = rainfall_series.rolling(window=window, min_periods=window).sum()

    spi_values = pd.Series(index=rainfall_series.index, dtype=float)

    for month in range(1, 13):
        # Get accumulated values for this calendar month across all years
        # (SPI is fitted per calendar month to account for seasonality)
        month_mask = (rainfall_series.index % 12) == (month - 1) % 12
        month_vals = accumulated[month_mask].dropna()

        if len(month_vals) < 10:
            # Not enough data to fit; use z-score fallback
            if month_vals.std() > 0:
                spi_values[month_mask] = (accumulated[month_mask] - month_vals.mean()) / month_vals.std()
            else:
                spi_values[month_mask] = 0.0
            continue

        # Filter out zeros for gamma fitting (gamma requires positive values)
        positive_vals = month_vals[month_vals > 0]
        q_zero = (month_vals <= 0).sum() / len(month_vals)  # probability of zero

        if len(positive_vals) < 5:
            spi_values[month_mask] = 0.0
            continue

        try:
            # Fit gamma distribution to positive values
            alpha, loc, beta = gamma_dist.fit(positive_vals, floc=0)

            # Transform each value to SPI
            for idx in accumulated[month_mask].dropna().index:
                val = accumulated[idx]
                if val <= 0:
                    # Map zero rainfall to cumulative probability
                    cum_prob = q_zero / 2  # midpoint of zero probability
                else:
                    # Gamma CDF + zero adjustment
                    cum_prob = q_zero + (1 - q_zero) * gamma_dist.cdf(val, alpha, loc=0, scale=beta)

                # Clamp to avoid infinity
                cum_prob = max(0.001, min(0.999, cum_prob))

                # Convert to standard normal (inverse CDF)
                from scipy.stats import norm
                spi_values[idx] = norm.ppf(cum_prob)
        except Exception:
            # Fallback to z-score if gamma fit fails
            if month_vals.std() > 0:
                spi_values[month_mask] = (accumulated[month_mask] - month_vals.mean()) / month_vals.std()
            else:
                spi_values[month_mask] = 0.0

    return spi_values


def load_drought_data():
    """
    Load and preprocess drought data (using rainfall data source).

    IMPORTANT: Features use LAGGED (shifted) values only to prevent data leakage.
    The target (drought_score) is derived from current-month rainfall deficit,
    so features must not include current-month deficit or unshifted rolling averages.

    NEW in v2: Adds SPI-3 and SPI-6 (Standardized Precipitation Index) computed
    from gamma-fitted rainfall distributions. SPI is the meteorological standard
    for drought quantification.

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
    # FIX: All rolling averages SHIFTED by 1 to avoid leaking current month's data

    # Rolling averages using ONLY past data (shift(1) excludes current month)
    df_melted['rolling_3mo'] = df_melted['Rainfall'].shift(1).rolling(window=3, min_periods=1).mean()
    df_melted['rolling_6mo'] = df_melted['Rainfall'].shift(1).rolling(window=6, min_periods=1).mean()

    # Calculate normal rainfall per month (long-term average)
    monthly_normals = df_melted.groupby('Month')['Rainfall'].transform('mean')
    df_melted['normal_rainfall'] = monthly_normals

    # FIX: Use LAGGED deficit (previous month's deficit), not current month's
    df_melted['_raw_deficit'] = (df_melted['normal_rainfall'] - df_melted['Rainfall']) / df_melted['normal_rainfall'].replace(0, 1) * 100
    df_melted['_raw_deficit'] = df_melted['_raw_deficit'].clip(lower=0, upper=100)
    df_melted['deficit_pct'] = df_melted['_raw_deficit'].shift(1)  # Previous month's deficit as feature

    # Previous year's drought indicator (lag 12)
    df_melted['prev_year_drought'] = df_melted['_raw_deficit'].shift(12)

    # Monsoon strength (approximate) - also lagged
    df_melted['rolling_4mo_sum'] = df_melted['Rainfall'].shift(1).rolling(window=4, min_periods=1).sum()
    monsoon_normal = df_melted[df_melted['Month'].isin([6,7,8,9])]['Rainfall'].mean() * 4
    df_melted['monsoon_strength'] = df_melted['rolling_4mo_sum'] / monsoon_normal
    df_melted['monsoon_strength'] = df_melted['monsoon_strength'].clip(upper=2.0)

    # NEW: SPI-3 and SPI-6 (Standardized Precipitation Index)
    # Computed from LAGGED rainfall (shift by 1) to prevent data leakage
    lagged_rainfall = df_melted['Rainfall'].shift(1)
    df_melted['spi_3'] = _compute_spi(lagged_rainfall, window=3)
    df_melted['spi_6'] = _compute_spi(lagged_rainfall, window=6)

    # NEW: Consecutive dry months (count of consecutive months with below-normal rainfall)
    below_normal = (df_melted['Rainfall'].shift(1) < df_melted['normal_rainfall'] * 0.75).astype(int)
    # Count consecutive 1s using cumsum trick
    cumsum = below_normal.cumsum()
    reset = cumsum.where(below_normal == 0).ffill().fillna(0)
    df_melted['consecutive_dry_months'] = (cumsum - reset).clip(upper=12)

    # Target: Drought Score (uses CURRENT month data - this is what we predict)
    def calculate_score(row):
        if row['normal_rainfall'] == 0: return 0
        deficit = (row['normal_rainfall'] - row['Rainfall']) / row['normal_rainfall'] * 100
        return max(0, min(100, deficit))

    df_melted['drought_score'] = df_melted.apply(calculate_score, axis=1)

    df_melted = df_melted.dropna()

    features = ['rolling_3mo', 'rolling_6mo', 'deficit_pct', 'prev_year_drought',
                'monsoon_strength', 'spi_3', 'spi_6', 'consecutive_dry_months']
    target = 'drought_score'

    return df_melted[features], df_melted[target]

def load_heatwave_data():
    """
    Load and preprocess temperature data for heatwave prediction.

    NEW in v2: Added temp_min, diurnal_range, precipitation, heat_streak,
    and temp_min lags for better signal extraction. These use data already
    in the CSV (no external download needed).

    Returns:
        X (DataFrame): Features
        y (Series): Target (is_heatwave)
    """
    if not os.path.exists(TEMP_PATH):
        raise FileNotFoundError(f"Temperature data not found at {TEMP_PATH}")

    df = pd.read_csv(TEMP_PATH, parse_dates=["date"])

    # Feature Engineering
    df['month'] = df['date'].dt.month

    # Lags - max temp
    df['temp_max_lag1'] = df['temp_max'].shift(1)
    df['temp_max_lag2'] = df['temp_max'].shift(2)
    df['temp_max_lag3'] = df['temp_max'].shift(3)

    # Rolling - max temp
    df['temp_max_7day_avg'] = df['temp_max'].rolling(window=7).mean()

    # Cyclical
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Anomaly
    monthly_normals = df.groupby('month')['temp_max'].transform('mean')
    df['temp_anomaly'] = df['temp_max'] - monthly_normals

    # NEW: Diurnal range (large range = dry air = more heatwave risk)
    df['diurnal_range'] = df['temp_max'] - df['temp_min']
    df['diurnal_range_lag1'] = df['diurnal_range'].shift(1)

    # NEW: Min temp features (warm nights = sustained heat stress)
    df['temp_min_lag1'] = df['temp_min'].shift(1)
    df['temp_min_7day_avg'] = df['temp_min'].rolling(window=7).mean()

    # NEW: Precipitation (lack of rain intensifies heat)
    df['precipitation'] = df['precipitation'].fillna(0.0)
    df['precip_7day_sum'] = df['precipitation'].rolling(window=7).sum()

    # NEW: Heat streak - consecutive days above 38°C (shifted to avoid leakage)
    hot_day = (df['temp_max'].shift(1) >= 38).astype(int)
    cumsum_hot = hot_day.cumsum()
    reset_hot = cumsum_hot.where(hot_day == 0).ffill().fillna(0)
    df['heat_streak'] = (cumsum_hot - reset_hot).clip(upper=14)

    # Target: Heatwave Definition
    # Max temp >= 40 AND anomaly >= 4.5  (Strict definition) OR just > 40 for simplicity in some contexts,
    # but sticking to previous definition: >= 40 OR >= 4.5 above normal
    df['is_heatwave'] = ((df['temp_max'] >= 40) | (df['temp_anomaly'] >= 4.5)).astype(int)

    # Handle humidity
    df['humidity'] = df['humidity'].fillna(df['humidity'].median())

    df = df.dropna()

    features = ['temp_max_lag1', 'temp_max_lag2', 'temp_max_lag3',
                'temp_max_7day_avg', 'humidity', 'month_sin', 'month_cos', 'month',
                'diurnal_range_lag1', 'temp_min_lag1', 'temp_min_7day_avg',
                'precip_7day_sum', 'heat_streak']
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
