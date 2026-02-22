"""
Download real temperature data from Open-Meteo API for Hyderabad.
This data is free, no API key required.
"""

import requests
import pandas as pd
import os
from datetime import datetime

def download_hyderabad_temperature():
    """Download historical temperature data for Hyderabad from Open-Meteo API."""
    
    print("=" * 60)
    print("DOWNLOADING HYDERABAD TEMPERATURE DATA")
    print("Source: Open-Meteo Historical Weather API (FREE)")
    print("=" * 60)
    
    # Hyderabad coordinates
    LAT = 17.385
    LON = 78.4867
    
    # Date range (2015-2024 for good coverage)
    START_DATE = "2015-01-01"
    END_DATE = "2024-12-31"
    
    # Open-Meteo Historical API
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean",
        "timezone": "Asia/Kolkata"
    }
    
    print(f"\n1. Fetching data from Open-Meteo API...")
    print(f"   Location: Hyderabad ({LAT}, {LON})")
    print(f"   Period: {START_DATE} to {END_DATE}")
    
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # Extract daily data
        daily = data["daily"]
        
        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "temp_max": daily["temperature_2m_max"],
            "temp_min": daily["temperature_2m_min"],
            "precipitation": daily.get("precipitation_sum", [0] * len(daily["time"])),
            "humidity": daily.get("relative_humidity_2m_mean", [50] * len(daily["time"]))
        })
        
        print(f"\n2. Data retrieved successfully!")
        print(f"   Total records: {len(df)}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Max temp range: {df['temp_max'].min():.1f}°C to {df['temp_max'].max():.1f}°C")
        
        # Save to data folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        
        output_path = os.path.join(data_dir, "hyderabad_temperature.csv")
        df.to_csv(output_path, index=False)
        
        print(f"\n3. Data saved to: {output_path}")
        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETE - Ready for model training")
        print("=" * 60)
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"\nERROR: Failed to fetch data - {e}")
        return None

if __name__ == "__main__":
    download_hyderabad_temperature()
