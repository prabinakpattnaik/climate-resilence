"""
Download real temperature + weather data from Open-Meteo API for Hyderabad.
This data is free, no API key required.

v2: Extended to 2025 + wind, solar, ET0, soil moisture, pressure columns.
"""

import requests
import pandas as pd
import os
from datetime import datetime


def download_hyderabad_temperature():
    """Download historical temperature & weather data for Hyderabad from Open-Meteo API."""

    print("=" * 60)
    print("DOWNLOADING HYDERABAD WEATHER DATA (Extended v2)")
    print("Source: Open-Meteo Historical Weather API (FREE)")
    print("=" * 60)

    # Hyderabad coordinates
    LAT = 17.385
    LON = 78.4867

    # Date range (2015-2025 for 11 years of coverage)
    START_DATE = "2015-01-01"
    END_DATE = "2025-12-31"

    # Open-Meteo Historical API
    url = "https://archive-api.open-meteo.com/v1/archive"

    # Extended daily variables
    daily_vars = ",".join([
        # Original 4 variables
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "relative_humidity_2m_mean",
        # NEW v2: Wind
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        # NEW v2: Solar radiation
        "shortwave_radiation_sum",
        # NEW v2: Evapotranspiration (FAO reference)
        "et0_fao_evapotranspiration",
        # NEW v2: Soil moisture (top 10cm)
        "soil_moisture_0_to_10cm_mean",
        # NEW v2: Sea-level pressure
        "pressure_msl_mean",
    ])

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": daily_vars,
        "timezone": "Asia/Kolkata"
    }

    print(f"\n1. Fetching data from Open-Meteo API...")
    print(f"   Location: Hyderabad ({LAT}, {LON})")
    print(f"   Period: {START_DATE} to {END_DATE}")
    print(f"   Variables: 10 daily weather variables")

    try:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()

        # Extract daily data
        daily = data["daily"]
        n = len(daily["time"])

        df = pd.DataFrame({
            # Original columns (backward compatible)
            "date": pd.to_datetime(daily["time"]),
            "temp_max": daily["temperature_2m_max"],
            "temp_min": daily["temperature_2m_min"],
            "precipitation": daily.get("precipitation_sum", [0] * n),
            "humidity": daily.get("relative_humidity_2m_mean", [50] * n),
            # NEW v2 columns
            "wind_speed_max": daily.get("wind_speed_10m_max", [0] * n),
            "wind_gusts_max": daily.get("wind_gusts_10m_max", [0] * n),
            "solar_radiation": daily.get("shortwave_radiation_sum", [0] * n),
            "et0_evapotranspiration": daily.get("et0_fao_evapotranspiration", [0] * n),
            "soil_moisture_0_10cm": daily.get("soil_moisture_0_to_10cm_mean", [0] * n),
            "pressure_msl": daily.get("pressure_msl_mean", [1013] * n),
        })

        # Fill any None/NaN with sensible defaults
        defaults = {
            "temp_max": 30.0, "temp_min": 20.0, "precipitation": 0.0,
            "humidity": 50.0, "wind_speed_max": 0.0, "wind_gusts_max": 0.0,
            "solar_radiation": 0.0, "et0_evapotranspiration": 0.0,
            "soil_moisture_0_10cm": 0.2, "pressure_msl": 1013.0,
        }
        df = df.fillna(defaults)

        print(f"\n2. Data retrieved successfully!")
        print(f"   Total records: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Max temp range: {df['temp_max'].min():.1f} to {df['temp_max'].max():.1f} C")
        print(f"   Wind speed range: {df['wind_speed_max'].min():.1f} to {df['wind_speed_max'].max():.1f} km/h")
        print(f"   Pressure range: {df['pressure_msl'].min():.1f} to {df['pressure_msl'].max():.1f} hPa")
        print(f"   Solar radiation range: {df['solar_radiation'].min():.1f} to {df['solar_radiation'].max():.1f} MJ/m2")

        # Save to data folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "..", "data")
        os.makedirs(data_dir, exist_ok=True)

        output_path = os.path.join(data_dir, "hyderabad_temperature.csv")
        df.to_csv(output_path, index=False)

        print(f"\n3. Data saved to: {output_path}")
        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETE - Ready for model training")
        print(f"  Original columns: date, temp_max, temp_min, precipitation, humidity")
        print(f"  NEW columns: wind_speed_max, wind_gusts_max, solar_radiation,")
        print(f"               et0_evapotranspiration, soil_moisture_0_10cm, pressure_msl")
        print("=" * 60)

        return df

    except requests.exceptions.RequestException as e:
        print(f"\nERROR: Failed to fetch data - {e}")
        return None


if __name__ == "__main__":
    download_hyderabad_temperature()
