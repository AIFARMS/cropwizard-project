from typing import Any, Dict
import pgeocode
import requests
from geopy.geocoders import Nominatim
import pandas as pd
from datetime import datetime

def get_coordinates(location: str) -> tuple[float, float]:
    """Get coordinates from either zip code or city name."""
    # Try as zip code first
    if location.isdigit() and len(location) == 5:
        nomi = pgeocode.Nominatim('us')
        response = nomi.query_postal_code(location)
        if not pd.isna(response.latitude):
            return response.latitude, response.longitude
    
    # If not a zip code or zip code not found, try as city name
    geolocator = Nominatim(user_agent="weathertool")
    location_data = geolocator.geocode(location)
    if location_data:
        return location_data.latitude, location_data.longitude
    
    raise ValueError(f"Could not find coordinates for location: {location}")

def get_weather(location: str) -> dict:
    """Get weather data for a given location (city or zip code)."""
    try:
        lat, lon = get_coordinates(location)
        url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,wind_speed_10m,precipitation",
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        
        # Format the response in a more user-friendly way
        current = weather_data["current"]
        hourly = weather_data["hourly"]
        
        current_time = datetime.now().isoformat()
        
        return {
            "success": True,
            "request_time": current_time,
            "current_weather": {
                "temperature_celsius": current["temperature_2m"],
                "temperature_fahrenheit": (current["temperature_2m"] * 9/5) + 32,
                "wind_speed_kmh": current["wind_speed_10m"],
                "wind_speed_mph": current["wind_speed_10m"] * 0.621371,
                "precipitation_mm": current["precipitation"],
                "precipitation_in": current["precipitation"] * 0.0393701,
                "time": current["time"]
            },
            "hourly_forecast": [{
                "time": hourly["time"][i],
                "temperature_celsius": hourly["temperature_2m"][i],
                "temperature_fahrenheit": (hourly["temperature_2m"][i] * 9/5) + 32,
                "humidity_percent": hourly["relative_humidity_2m"][i],
                "wind_speed_kmh": hourly["wind_speed_10m"][i],
                "wind_speed_mph": hourly["wind_speed_10m"][i] * 0.621371,
                "precipitation_mm": hourly["precipitation"][i],
                "precipitation_in": hourly["precipitation"][i] * 0.0393701
            } for i in range(24)]
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def main(**inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get weather data for a given location (city or zip code).
    
    This function accepts a location input and returns current weather conditions 
    and 24-hour forecast data for that location. The location can be either a 
    US ZIP code or a city name. The function provides temperature in both Celsius 
    and Fahrenheit, wind speed in km/h and mph, precipitation in mm and inches,
    and relative humidity percentage.
    
    Args:
        inputs: A dictionary containing the input parameters
            - location: A string specifying either a US ZIP code or city name
            
    Returns:
        A dictionary containing:
        - success: Boolean indicating if the request was successful
        - request_time: ISO formatted timestamp of when the request was made
        - current_weather: Dictionary of current weather conditions including
          temperature, wind speed, and precipitation
        - hourly_forecast: List of hourly weather forecasts for 24 hours
        - error: Error message if success is False
    """
    location = inputs.get("location", "").strip()
    
    if not location:
        return {
            "success": False,
            "error": "No location provided. Please provide either a city name or ZIP code."
        }
    
    return get_weather(location)
