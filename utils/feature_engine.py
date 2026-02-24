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
ENSO_PATH = os.path.join(DATA_DIR, "enso_mei.csv")


class SmartFeatureEngine:
    """Auto-computes ML features from CSV data + live API for citizen-friendly predictions."""

    def __init__(self):
        self._rainfall_monthly = None
        self._temp_df = None
        self._crop_df = None
        self._enso_df = None

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

    def _load_enso(self):
        """Lazy-load ENSO MEI v2 data."""
        if self._enso_df is not None:
            return self._enso_df
        if os.path.exists(ENSO_PATH):
            self._enso_df = pd.read_csv(ENSO_PATH)
        else:
            self._enso_df = pd.DataFrame(columns=['year', 'month', 'mei_value'])
        return self._enso_df

    def _get_enso_values(self, target_year: int, target_month: int) -> tuple:
        """
        Look up ENSO MEI value for target month and 3-month lagged value.
        Returns (mei_value, mei_lag3).
        """
        enso = self._load_enso()
        if len(enso) == 0:
            return 0.0, 0.0

        # Current MEI
        row = enso[(enso['year'] == target_year) & (enso['month'] == target_month)]
        mei_value = float(row.iloc[0]['mei_value']) if len(row) > 0 else 0.0

        # 3-month lagged MEI
        lag_month = target_month - 3
        lag_year = target_year
        if lag_month <= 0:
            lag_month += 12
            lag_year -= 1
        lag_row = enso[(enso['year'] == lag_year) & (enso['month'] == lag_month)]
        mei_lag3 = float(lag_row.iloc[0]['mei_value']) if len(lag_row) > 0 else 0.0

        return mei_value, mei_lag3

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

        # NEW v3: ENSO index
        mei_value, mei_lag3 = self._get_enso_values(last_year, target_month)

        return {
            'lag_1': round(lag_1, 2),
            'lag_2': round(lag_2, 2),
            'lag_3': round(lag_3, 2),
            'lag_12': round(lag_12, 2),
            'month_sin': round(month_sin, 6),
            'month_cos': round(month_cos, 6),
            'rolling_3': round(rolling_3, 2),
            'mei_value': round(mei_value, 3),
            'mei_lag3': round(mei_lag3, 3),
            'month': target_month
        }

    # --- DROUGHT ---

    def compute_drought_features(self, target_month: int) -> dict:
        """
        Auto-compute drought features from rainfall CSV.
        All features use LAGGED data (consistent with how the model was trained).

        v2: Now includes SPI-3, SPI-6 (z-score approximation), and consecutive_dry_months.

        Returns dict with: rolling_3mo_avg, rolling_6mo_avg, deficit_pct, prev_year_drought,
                          monsoon_strength, spi_3, spi_6, consecutive_dry_months
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

        # NEW: SPI-3 (z-score of 3-month accumulated rainfall for this calendar month)
        spi_3 = self._compute_spi_for_month(mdf, recent, idx, target_month, window=3)
        spi_6 = self._compute_spi_for_month(mdf, recent, idx, target_month, window=6)

        # NEW: Consecutive dry months
        consecutive_dry = 0
        for k in range(1, 13):
            check_idx = idx - k
            if check_idx < 0:
                break
            check_month = int(recent.iloc[check_idx]['month'])
            rain_val = float(recent.iloc[check_idx]['rainfall'])
            normal_val = float(monthly_normals.get(check_month, 1))
            if rain_val < normal_val * 0.75:
                consecutive_dry += 1
            else:
                break

        # NEW v3: ENSO index for drought prediction
        mei_value, mei_lag3 = self._get_enso_values(last_year, target_month)

        return {
            'rolling_3mo_avg': round(rolling_3mo, 2),
            'rolling_6mo_avg': round(rolling_6mo, 2),
            'deficit_pct': round(deficit_pct, 2),
            'prev_year_drought': round(prev_year_drought, 2),
            'monsoon_strength': round(monsoon_strength, 3),
            'spi_3': round(spi_3, 3),
            'spi_6': round(spi_6, 3),
            'consecutive_dry_months': min(12, consecutive_dry),
            'mei_value': round(mei_value, 3),
            'mei_lag3': round(mei_lag3, 3)
        }

    def _compute_spi_for_month(self, full_df, recent, idx, target_month, window=3):
        """
        Approximate SPI for a specific month using z-score of accumulated rainfall.
        Uses all historical data for the same calendar month to compute mean/std.
        """
        # Accumulate rainfall over window months BEFORE target (lagged)
        prev_window = recent.iloc[max(0, idx - window):idx]['rainfall']
        if len(prev_window) < window:
            return 0.0
        accumulated = float(prev_window.sum())

        # Get historical accumulated values for same calendar month
        all_accumulated = []
        for _, grp in full_df.groupby('year'):
            grp = grp.sort_values('month').reset_index(drop=True)
            month_idx = grp[grp['month'] == target_month].index
            if len(month_idx) == 0:
                continue
            mi = month_idx[0]
            if mi >= window:
                prev_vals = grp.iloc[mi - window:mi]['rainfall']
                all_accumulated.append(float(prev_vals.sum()))

        if len(all_accumulated) < 5:
            return 0.0

        hist_mean = np.mean(all_accumulated)
        hist_std = np.std(all_accumulated)
        if hist_std < 0.01:
            return 0.0

        return float((accumulated - hist_mean) / hist_std)

    def _drought_fallback(self, monthly_avg):
        return {
            'rolling_3mo_avg': round(float(monthly_avg.mean()), 2),
            'rolling_6mo_avg': round(float(monthly_avg.mean()), 2),
            'deficit_pct': 25.0,
            'prev_year_drought': 25.0,
            'monsoon_strength': 0.8,
            'spi_3': 0.0,
            'spi_6': 0.0,
            'consecutive_dry_months': 2,
            'mei_value': 0.0,
            'mei_lag3': 0.0
        }

    # --- HEATWAVE ---

    def compute_heatwave_features(self, target_month: int) -> dict:
        """
        Auto-compute heatwave features from temperature CSV + live API humidity.

        v2: Now includes diurnal_range, temp_min, precipitation, and heat_streak features
        matching the upgraded heatwave model (13 features total).

        Returns dict with: max_temp_lag1/2/3, temp_max_7day_avg, humidity, month_sin, month_cos,
                          month, diurnal_range_lag1, temp_min_lag1, temp_min_7day_avg,
                          precip_7day_sum, heat_streak
        """
        tdf = self._load_temp_df()
        last_rows = tdf.tail(14)  # Need 14 days for heat streak computation

        temp_max_lag1 = float(last_rows.iloc[-1]['temp_max'])
        temp_max_lag2 = float(last_rows.iloc[-2]['temp_max'])
        temp_max_lag3 = float(last_rows.iloc[-3]['temp_max'])
        temp_max_7day_avg = float(last_rows.tail(7)['temp_max'].mean())

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

        # NEW: Diurnal range (temp_max - temp_min) for previous day
        diurnal_range_lag1 = float(last_rows.iloc[-1]['temp_max'] - last_rows.iloc[-1]['temp_min'])

        # NEW: Min temperature features
        temp_min_lag1 = float(last_rows.iloc[-1]['temp_min'])
        temp_min_7day_avg = float(last_rows.tail(7)['temp_min'].mean())

        # NEW: 7-day precipitation sum
        precip_vals = last_rows.tail(7)['precipitation'].fillna(0.0)
        precip_7day_sum = float(precip_vals.sum())

        # NEW: Heat streak (consecutive days >= 38°C in recent data, shifted by 1)
        heat_streak = 0
        for i in range(len(last_rows) - 1, 0, -1):  # skip the most recent (lag)
            if float(last_rows.iloc[i - 1]['temp_max']) >= 38:
                heat_streak += 1
            else:
                break
        heat_streak = min(14, heat_streak)

        # NEW v3: Weather-enriched features from extended CSV
        wind_speed_lag1 = 0.0
        wind_gusts_lag1 = 0.0
        solar_rad_lag1 = 0.0
        solar_rad_3day_avg = 0.0
        et0_lag1 = 0.0
        et0_7day_avg = 0.0
        soil_moisture_lag1 = 0.2
        pressure_lag1 = 1013.0
        pressure_change_1d = 0.0

        if 'wind_speed_max' in last_rows.columns:
            wind_speed_lag1 = float(last_rows.iloc[-1]['wind_speed_max'])
            wind_gusts_lag1 = float(last_rows.iloc[-1].get('wind_gusts_max', 0.0))

        if 'solar_radiation' in last_rows.columns:
            solar_rad_lag1 = float(last_rows.iloc[-1]['solar_radiation'])
            solar_rad_3day_avg = float(last_rows.tail(3)['solar_radiation'].mean())

        if 'et0_evapotranspiration' in last_rows.columns:
            et0_lag1 = float(last_rows.iloc[-1]['et0_evapotranspiration'])
            et0_7day_avg = float(last_rows.tail(7)['et0_evapotranspiration'].mean())

        if 'soil_moisture_0_10cm' in last_rows.columns:
            soil_moisture_lag1 = float(last_rows.iloc[-1]['soil_moisture_0_10cm'])

        if 'pressure_msl' in last_rows.columns:
            pressure_lag1 = float(last_rows.iloc[-1]['pressure_msl'])
            if len(last_rows) >= 2:
                pressure_change_1d = float(
                    last_rows.iloc[-1]['pressure_msl'] - last_rows.iloc[-2]['pressure_msl']
                )

        # Try live data for wind/pressure if CSV doesn't have them
        if wind_speed_lag1 == 0.0:
            try:
                from utils.weather_service import WeatherService
                ws = WeatherService()
                live = ws.get_live_conditions()
                if live.get('status') == 'success':
                    wind_speed_lag1 = float(live.get('wind_speed', 0.0))
                    pressure_lag1 = float(live.get('pressure', 1013.0))
            except Exception:
                pass

        return {
            'max_temp_lag1': round(temp_max_lag1, 1),
            'max_temp_lag2': round(temp_max_lag2, 1),
            'max_temp_lag3': round(temp_max_lag3, 1),
            'temp_max_7day_avg': round(temp_max_7day_avg, 1),
            'humidity': round(humidity, 1),
            'month_sin': round(month_sin, 6),
            'month_cos': round(month_cos, 6),
            'month': target_month,
            'diurnal_range_lag1': round(diurnal_range_lag1, 1),
            'temp_min_lag1': round(temp_min_lag1, 1),
            'temp_min_7day_avg': round(temp_min_7day_avg, 1),
            'precip_7day_sum': round(precip_7day_sum, 1),
            'heat_streak': heat_streak,
            # NEW v3: Weather-enriched features
            'wind_speed_lag1': round(wind_speed_lag1, 1),
            'wind_gusts_lag1': round(wind_gusts_lag1, 1),
            'solar_rad_lag1': round(solar_rad_lag1, 1),
            'solar_rad_3day_avg': round(solar_rad_3day_avg, 1),
            'et0_lag1': round(et0_lag1, 2),
            'et0_7day_avg': round(et0_7day_avg, 2),
            'soil_moisture_lag1': round(soil_moisture_lag1, 3),
            'pressure_lag1': round(pressure_lag1, 1),
            'pressure_change_1d': round(pressure_change_1d, 1)
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
