import requests

from config import OPENWEATHER_API_KEY, logger

class WeatherAPI:
    def __init__(self):
        self.api_key = OPENWEATHER_API_KEY
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    def get_weather(self, city):
        """
        Fetch weather data for a given city
        """
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key is not set")
        
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric"
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as err:
            logger.error(f"Exception occured while calling weatehr api: {err}")
            return {"error": str(err)}
    
    def format_weather_data(self, data):
        """
        Format the weather data into a readable format
        """
        if "error" in data:
            return f"Error: {data['error']}"
        
        city = data["name"]
        country = data["sys"]["country"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        description = data["weather"][0]["description"]
        
        formatted_data = f"""
        Weather in {city}, {country}:
        - Temperature: {temp}°C
        - Feels like: {feels_like}°C
        - Humidity: {humidity}%
        - Wind speed: {wind_speed} m/s
        - Conditions: {description}
        """
        
        return formatted_data