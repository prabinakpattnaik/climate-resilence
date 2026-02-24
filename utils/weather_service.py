import requests
import datetime
import time
import threading

class WeatherService:
    """Cached singleton weather service. Reuses data within TTL to avoid API rate limits."""
    _instance = None
    _lock = threading.Lock()
    _cache = {}
    _CACHE_TTL = 60  # seconds

    def __new__(cls, lat=17.3850, lon=78.4867):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.lat = lat
                cls._instance.lon = lon
                cls._instance.base_url = "https://api.open-meteo.com/v1/forecast"
            return cls._instance

    def _get_cached(self, key):
        entry = self._cache.get(key)
        if entry and (time.time() - entry['ts']) < self._CACHE_TTL:
            return entry['data']
        return None

    def _set_cached(self, key, data):
        self._cache[key] = {'data': data, 'ts': time.time()}

    def get_live_rainfall(self):
        """
        Fetches current weather data from Open-Meteo with 60s cache.
        Returns a dict: {'current': mm, 'prediction_1h': mm, 'is_raining': bool}
        """
        cached = self._get_cached('rainfall')
        if cached:
            return cached

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

            result = {
                "current_rainfall_mm": current_precip,
                "predicted_next_1h_mm": pred_1h,
                "is_raining": current_precip > 0,
                "status": "success",
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            }
            self._set_cached('rainfall', result)
            return result
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return {
                "current_rainfall_mm": 0,
                "predicted_next_1h_mm": 0,
                "is_raining": False,
                "status": "error",
                "error": str(e)
            }

    def get_live_air_quality(self):
        """
        Fetches live AQI and pollution data from Open-Meteo Air Quality API with 60s cache.
        """
        cached = self._get_cached('aqi')
        if cached:
            return cached

        aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "current": ["european_aqi", "pm2_5", "pm10", "ozone", "nitrogen_dioxide"],
            "timezone": "auto"
        }
        try:
            response = requests.get(aq_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            curr = data.get('current', {})
            result = {
                "aqi": curr.get("european_aqi"),
                "pm2_5": curr.get("pm2_5"),
                "pm10": curr.get("pm10"),
                "ozone": curr.get("ozone"),
                "no2": curr.get("nitrogen_dioxide"),
                "status": "Healthy" if curr.get("european_aqi", 0) < 50 else "Moderate" if curr.get("european_aqi", 0) < 100 else "Poor"
            }
            self._set_cached('aqi', result)
            return result
        except Exception as e:
            return {"error": str(e), "aqi": 45, "status": "Simulated"} # Fallback

    def get_live_conditions(self):
        """
        Fetches current temperature and humidity from Open-Meteo with 60s cache.
        Used by SmartFeatureEngine for heatwave auto-computation.
        """
        cached = self._get_cached('conditions')
        if cached:
            return cached

        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "current": [
                "temperature_2m", "relative_humidity_2m", "precipitation",
                # NEW v3: Wind and pressure for heatwave model
                "wind_speed_10m", "wind_gusts_10m", "surface_pressure",
            ],
            "timezone": "auto"
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            current = data.get('current', {})
            result = {
                "temperature": current.get("temperature_2m", 30.0),
                "humidity": current.get("relative_humidity_2m", 50.0),
                "precipitation": current.get("precipitation", 0.0),
                # NEW v3
                "wind_speed": current.get("wind_speed_10m", 0.0),
                "wind_gusts": current.get("wind_gusts_10m", 0.0),
                "pressure": current.get("surface_pressure", 1013.0),
                "status": "success"
            }
            self._set_cached('conditions', result)
            return result
        except Exception as e:
            return {
                "temperature": 30.0, "humidity": 50.0, "precipitation": 0.0,
                "wind_speed": 0.0, "wind_gusts": 0.0, "pressure": 1013.0,
                "status": "error"
            }


if __name__ == "__main__":
    ws = WeatherService()
    print(ws.get_live_rainfall())
