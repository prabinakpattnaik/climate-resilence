"""
SmartFeatureEngine: Auto-computes ML features from CSV historical data + live API.
Enables citizen-friendly predictions where users only provide minimal input (month, crop type).
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

RAINFALL_PATH = os.path.join(DATA_DIR, "hyderabad_rainfall_data.csv")
TEMP_PATH = os.path.join(DATA_DIR, "hyderabad_temperature.csv")
CROP_PATH = os.path.join(DATA_DIR, "crop_yield_india.csv")


class SmartFeatureEngine:
    """Auto-computes ML features from CSV data + live API for citizen-friendly predictions."""

    def __init__(self):
        self._rainfall_monthly = None
        self._temp_df = None
        self._crop_df = None

    # --- Lazy loaders (cached after first call) ---

    def _load_rainfall_monthly(self):
        if self._rainfall_monthly is not None:
            return self._rainfall_monthly

        df = pd.read_csv(RAINFALL_PATH)
        months_cols = ['Jan', 'Feb', 'Mar', 'April', 'May', 'June',
                       'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
        records = []
        for _, row in df.iterrows():
            for i, m in enumerate(months_cols):
                if m in row:
                    records.append({
                        'year': int(row['Year']),
                        'month': i + 1,
                        'rainfall': float(row[m])
                    })
        self._rainfall_monthly = pd.DataFrame(records).sort_values(
            ['year', 'month']).reset_index(drop=True)
        return self._rainfall_monthly

    def _load_temp_df(self):
        if self._temp_df is not None:
            return self._temp_df
        self._temp_df = pd.read_csv(TEMP_PATH, parse_dates=['date'])
        return self._temp_df

    def _load_crop_df(self):
        if self._crop_df is not None:
            return self._crop_df
        self._crop_df = pd.read_csv(CROP_PATH)
        return self._crop_df

    # --- RAINFALL ---

    def compute_rainfall_features(self, target_month: int) -> dict:
        """
        Given a target month (1-12), auto-compute all features the rainfall model needs
        from the most recent year of historical data.

        Returns dict with: lag_1, lag_2, lag_3, lag_12, month_sin, month_cos, rolling_3, month
        """
        mdf = self._load_rainfall_monthly()
        last_year = int(mdf['year'].max())
        monthly_avg = mdf.groupby('month')['rainfall'].mean()

        # Build recent window (last 2 years)
        recent = mdf[mdf['year'] >= last_year - 1].copy()
        recent = recent.sort_values(['year', 'month']).reset_index(drop=True)

        target_rows = recent[
            (recent['year'] == last_year) & (recent['month'] == target_month)
        ]

        if len(target_rows) > 0:
            idx = target_rows.index[0]
            lag_1 = float(recent.iloc[idx - 1]['rainfall']) if idx >= 1 else float(monthly_avg.get((target_month - 1) or 12, 50.0))
            lag_2 = float(recent.iloc[idx - 2]['rainfall']) if idx >= 2 else float(monthly_avg.get((target_month - 2) or 12, 50.0))
            lag_3 = float(recent.iloc[idx - 3]['rainfall']) if idx >= 3 else float(monthly_avg.get((target_month - 3) or 12, 50.0))

            lag_12_rows = recent[
                (recent['year'] == last_year - 1) & (recent['month'] == target_month)
            ]
            lag_12 = float(lag_12_rows.iloc[0]['rainfall']) if len(lag_12_rows) > 0 else float(monthly_avg.get(target_month, 100.0))
        else:
            # Fallback: use long-term monthly averages
            prev = [(target_month - i - 1) % 12 or 12 for i in range(3)]
            lag_1 = float(monthly_avg.get(prev[0], 50.0))
            lag_2 = float(monthly_avg.get(prev[1], 50.0))
            lag_3 = float(monthly_avg.get(prev[2], 50.0))
            lag_12 = float(monthly_avg.get(target_month, 100.0))

        month_sin = float(np.sin(2 * np.pi * target_month / 12))
        month_cos = float(np.cos(2 * np.pi * target_month / 12))
        rolling_3 = (lag_1 + lag_2 + lag_3) / 3.0

        return {
            'lag_1': round(lag_1, 2),
            'lag_2': round(lag_2, 2),
            'lag_3': round(lag_3, 2),
            'lag_12': round(lag_12, 2),
            'month_sin': round(month_sin, 6),
            'month_cos': round(month_cos, 6),
            'rolling_3': round(rolling_3, 2),
            'month': target_month
        }

    # --- DROUGHT ---

    def compute_drought_features(self, target_month: int) -> dict:
        """
        Auto-compute drought features from rainfall CSV.
        All features use LAGGED data (consistent with how the model was trained).

        Returns dict with: rolling_3mo_avg, rolling_6mo_avg, deficit_pct, prev_year_drought, monsoon_strength
        """
        mdf = self._load_rainfall_monthly()
        last_year = int(mdf['year'].max())
        monthly_normals = mdf.groupby('month')['rainfall'].mean()

        recent = mdf.sort_values(['year', 'month']).reset_index(drop=True)
        target_rows = recent[
            (recent['year'] == last_year) & (recent['month'] == target_month)
        ]

        if len(target_rows) == 0:
            return self._drought_fallback(monthly_normals)

        idx = target_rows.index[0]

        # rolling_3mo: avg of 3 months BEFORE target (shifted, matching data_loader)
        prev_3 = recent.iloc[max(0, idx - 3):idx]['rainfall']
        rolling_3mo = float(prev_3.mean()) if len(prev_3) > 0 else 50.0

        # rolling_6mo: avg of 6 months BEFORE target (shifted)
        prev_6 = recent.iloc[max(0, idx - 6):idx]['rainfall']
        rolling_6mo = float(prev_6.mean()) if len(prev_6) > 0 else 50.0

        # deficit_pct: PREVIOUS month's deficit from long-term normal (shifted)
        prev_month = (target_month - 1) or 12
        prev_rain = float(recent.iloc[idx - 1]['rainfall']) if idx >= 1 else 0
        normal_prev = float(monthly_normals.get(prev_month, 1))
        deficit_pct = max(0.0, min(100.0,
            (normal_prev - prev_rain) / max(normal_prev, 1) * 100))

        # prev_year_drought: deficit for same month one year ago
        prev_year_rows = recent[
            (recent['year'] == last_year - 1) & (recent['month'] == target_month)
        ]
        if len(prev_year_rows) > 0:
            py_rain = float(prev_year_rows.iloc[0]['rainfall'])
            normal_cur = float(monthly_normals.get(target_month, 1))
            prev_year_drought = max(0.0, min(100.0,
                (normal_cur - py_rain) / max(normal_cur, 1) * 100))
        else:
            prev_year_drought = 25.0

        # monsoon_strength: ratio of recent 4-month sum to historical monsoon average (shifted)
        prev_4 = recent.iloc[max(0, idx - 4):idx]['rainfall']
        rolling_4mo_sum = float(prev_4.sum())
        monsoon_avg = float(mdf[mdf['month'].isin([6, 7, 8, 9])]['rainfall'].mean()) * 4
        monsoon_strength = min(2.0, rolling_4mo_sum / max(monsoon_avg, 1))

        return {
            'rolling_3mo_avg': round(rolling_3mo, 2),
            'rolling_6mo_avg': round(rolling_6mo, 2),
            'deficit_pct': round(deficit_pct, 2),
            'prev_year_drought': round(prev_year_drought, 2),
            'monsoon_strength': round(monsoon_strength, 3)
        }

    def _drought_fallback(self, monthly_avg):
        return {
            'rolling_3mo_avg': round(float(monthly_avg.mean()), 2),
            'rolling_6mo_avg': round(float(monthly_avg.mean()), 2),
            'deficit_pct': 25.0,
            'prev_year_drought': 25.0,
            'monsoon_strength': 0.8
        }

    # --- HEATWAVE ---

    def compute_heatwave_features(self, target_month: int) -> dict:
        """
        Auto-compute heatwave features from temperature CSV + live API humidity.

        Returns dict with: max_temp_lag1/2/3, temp_max_7day_avg, humidity, month_sin, month_cos, month
        """
        tdf = self._load_temp_df()
        last_rows = tdf.tail(7)

        temp_max_lag1 = float(last_rows.iloc[-1]['temp_max'])
        temp_max_lag2 = float(last_rows.iloc[-2]['temp_max'])
        temp_max_lag3 = float(last_rows.iloc[-3]['temp_max'])
        temp_max_7day_avg = float(last_rows['temp_max'].mean())

        # Try live humidity from Open-Meteo; fall back to CSV
        humidity = float(last_rows.iloc[-1].get('humidity', 50.0))
        try:
            from utils.weather_service import WeatherService
            ws = WeatherService()
            live = ws.get_live_conditions()
            if live.get('status') == 'success':
                humidity = float(live['humidity'])
        except Exception:
            pass

        month_sin = float(np.sin(2 * np.pi * target_month / 12))
        month_cos = float(np.cos(2 * np.pi * target_month / 12))

        return {
            'max_temp_lag1': round(temp_max_lag1, 1),
            'max_temp_lag2': round(temp_max_lag2, 1),
            'max_temp_lag3': round(temp_max_lag3, 1),
            'temp_max_7day_avg': round(temp_max_7day_avg, 1),
            'humidity': round(humidity, 1),
            'month_sin': round(month_sin, 6),
            'month_cos': round(month_cos, 6),
            'month': target_month
        }

    # --- CROP ---

    def compute_crop_features(self, crop_type: str, state: str, season: str) -> dict:
        """
        Auto-compute crop features from historical averages in the crop CSV.

        Returns dict with: rainfall, rainfall_anomaly, fertilizer_per_area, pesticide_per_area
        """
        cdf = self._load_crop_df()
        target_states = ['Andhra Pradesh', 'Telangana', 'Karnataka']

        filtered = cdf[
            (cdf['State'].isin(target_states)) &
            (cdf['Crop'] == crop_type)
        ]

        if state in target_states:
            state_filtered = filtered[filtered['State'] == state]
            if len(state_filtered) > 0:
                filtered = state_filtered

        if season.strip():
            season_filtered = filtered[filtered['Season'].str.strip() == season.strip()]
            if len(season_filtered) > 0:
                filtered = season_filtered

        if len(filtered) == 0:
            filtered = cdf[cdf['Crop'] == crop_type]

        if len(filtered) == 0:
            return {
                'rainfall': 1000.0,
                'rainfall_anomaly': 0.0,
                'fertilizer_per_area': 80.0,
                'pesticide_per_area': 2.5
            }

        avg_rainfall = float(filtered['Annual_Rainfall'].mean())
        global_avg = float(cdf['Annual_Rainfall'].mean())
        rainfall_anomaly = (avg_rainfall - global_avg) / max(global_avg, 1) * 100

        avg_fert = float((filtered['Fertilizer'] / (filtered['Area'] + 1)).mean())
        avg_pest = float((filtered['Pesticide'] / (filtered['Area'] + 1)).mean())

        return {
            'rainfall': round(avg_rainfall, 2),
            'rainfall_anomaly': round(rainfall_anomaly, 2),
            'fertilizer_per_area': round(avg_fert, 2),
            'pesticide_per_area': round(avg_pest, 2)
        }
