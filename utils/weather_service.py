import requests
import datetime

class WeatherService:
    def __init__(self, lat=17.3850, lon=78.4867):
        """Default target: Hyderabad city center."""
        self.lat = lat
        self.lon = lon
        self.base_url = "https://api.open-meteo.com/v1/forecast"

    def get_live_rainfall(self):
        """
        Fetches current weather data from Open-Meteo.
        Returns a dict: {'current': mm, 'prediction_1h': mm, 'is_raining': bool}
        """
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "current": ["precipitation", "rain", "showers"],
            "hourly": ["precipitation", "precipitation_probability"],
            "timezone": "auto",
            "forecast_days": 1
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            current_precip = data.get('current', {}).get('precipitation', 0)
            
            # Predict for the next hour (index 0 of hourly data usually corresponds to the current hour)
            hourly_precip = data.get('hourly', {}).get('precipitation', [0, 0])
            pred_1h = hourly_precip[1] if len(hourly_precip) > 1 else 0
            
            return {
                "current_rainfall_mm": current_precip,
                "predicted_next_1h_mm": pred_1h,
                "is_raining": current_precip > 0,
                "status": "success",
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            }
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return {
                "current_rainfall_mm": 0,
                "predicted_next_1h_mm": 0,
                "is_raining": False,
                "status": "error",
                "error": str(e)
            }

if __name__ == "__main__":
    ws = WeatherService()
    print(ws.get_live_rainfall())
