import random
import time
from utils.weather_service import WeatherService

class IoTSensorHub:
    """Simulates real-time IoT sensor data for Hyderabad wards."""
    
    def __init__(self):
        # We simulate sensors for major areas/wards
        self.wards = [
            "Khairatabad", "Banjara Hills", "Jubilee Hills", 
            "Gachibowli", "Secunderabad", "LB Nagar", 
            "Uppal", "Mehdipatnam", "Charminar", "Kukatpally"
        ]

    def get_live_sensor_data(self):
        """Generates random ward data but includes REAL city-wide Air Quality."""
        ws = WeatherService()
        real_aqi = ws.get_live_air_quality()
        
        sensors = []
        for ward in self.wards:
            # 1. Nala (Drainage) Level Sensors (%)
            nala_fill = random.uniform(20, 85)
            
            # 2. Surface Temperature
            surface_temp = random.uniform(32, 44)
            
            # 3. Groundwater
            water_depth_m = random.uniform(100, 350)
            
            sensors.append({
                "ward": ward,
                "nala_fill_pct": round(nala_fill, 1),
                "surface_temp_c": round(surface_temp, 1),
                "water_depth_m": round(water_depth_m, 1),
                "status": "Critical" if nala_fill > 80 or surface_temp > 42 else "Active"
            })
            
        return {
            "timestamp": time.strftime("%H:%M:%S"),
            "sensors": sensors,
            "real_city_aqi": real_aqi, # Corrected typo from aei to aqi
            "system_health": "Optimal",
            "active_nodes": len(sensors)
        }
