import unittest

from unittest.mock import patch, MagicMock
from src.api.weather import WeatherAPI


class TestWeatherAPI(unittest.TestCase):

    def setUp(self):
        self.weather_api = WeatherAPI()
        self.weather_api.api_key = "test_key"
    
    @patch('requests.get')
    def test_get_weather_success(self, mock_get):
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "London",
            "sys": {"country": "GB"},
            "main": {"temp": 15.5, "feels_like": 14.0, "humidity": 75},
            "wind": {"speed": 5.0},
            "weather": [{"description": "cloudy"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Call the API
        result = self.weather_api.get_weather("London")
        
        # Verify the result
        self.assertEqual(result["name"], "London")
        self.assertEqual(result["sys"]["country"], "GB")
        self.assertEqual(result["main"]["temp"], 15.5)
    
    @patch('requests.get')
    def test_get_weather_error(self, mock_get):
        # Mock the API response with an error
        mock_get.side_effect = Exception("API error")
        
        # Call the API
        result = self.weather_api.get_weather("London")
        
        # Verify the result contains an error
        self.assertIn("error", result)
    
    def test_format_weather_data(self):
        # Test weather data formatting
        weather_data = {
            "name": "London",
            "sys": {"country": "GB"},
            "main": {"temp": 15.5, "feels_like": 14.0, "humidity": 75},
            "wind": {"speed": 5.0},
            "weather": [{"description": "cloudy"}]
        }
        
        formatted_data = self.weather_api.format_weather_data(weather_data)
        
        # Verify the formatted data contains all expected information
        self.assertIn("London", formatted_data)
        self.assertIn("GB", formatted_data)
        self.assertIn("15.5°C", formatted_data)
        self.assertIn("14.0°C", formatted_data)
        self.assertIn("75%", formatted_data)
        self.assertIn("5.0 m/s", formatted_data)
        self.assertIn("cloudy", formatted_data)