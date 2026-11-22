"""Small Open-Meteo client for current conditions and local-day forecasts."""

import json
import re
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen


# Function: extract_weather_location
# Purpose: Extract the location following weather/forecast or temperature wording.
def extract_weather_location(query: str) -> str:
    """Extract the location following weather/forecast or temperature wording."""
    match = re.search(
        r"\b(?:weather|forecast|temperature)(?:\s+report)?\s+"
        r"(?:in|for|at)\s+([A-Za-z][A-Za-z .'-]*?)(?=,|[?.;]|"
        r"\s+(?:and|today|tonight|now|tomorrow)\b|\s*\n|$)",
        query,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(
            "Please include a location, for example: 'weather in Dubai today'."
        )
    return match.group(1).strip()


# Function: _get_json
# Purpose: Fetch and decode a JSON response with a bounded network timeout.
def _get_json(url: str) -> dict:
    """Fetch and decode a JSON response with a bounded network timeout."""
    try:
        with urlopen(url, timeout=12) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Weather service request failed: {error}") from error


# Function: get_weather_report
# Purpose: Return observed conditions and today's forecast in the location's timezone.
def get_weather_report(location: str) -> str:
    """Return observed conditions and today's forecast in the location's timezone."""
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode(
        {"name": location, "count": 1, "language": "en", "format": "json"}
    )
    places = _get_json(geocode_url).get("results", [])
    if not places:
        raise ValueError(f"Could not find a weather location matching {location!r}.")

    place = places[0]
    weather_url = "https://api.open-meteo.com/v1/forecast?" + urlencode(
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,weather_code,wind_speed_10m"
            ),
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,sunrise,sunset"
            ),
            "timezone": "auto",
            "forecast_days": 1,
        }
    )
    forecast = _get_json(weather_url)
    current = forecast.get("current", {})
    daily = forecast.get("daily", {})
    if not current.get("time") or not daily.get("time"):
        raise RuntimeError("Weather service returned no current observation or daily forecast.")

    weather_description = {
        0: "clear",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        56: "light freezing drizzle",
        57: "dense freezing drizzle",
        61: "light rain",
        63: "moderate rain",
        65: "heavy rain",
        66: "light freezing rain",
        67: "heavy freezing rain",
        71: "light snow",
        73: "moderate snow",
        75: "heavy snow",
        77: "snow grains",
        80: "rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        85: "light snow showers",
        86: "heavy snow showers",
        95: "thunderstorm",
        96: "thunderstorm with light hail",
        99: "thunderstorm with heavy hail",
    }.get(current.get("weather_code"), "conditions reported")

    location_name = ", ".join(
        part
        for part in (place.get("name"), place.get("admin1"), place.get("country"))
        if part
    )
    current_units = forecast.get("current_units", {})
    daily_units = forecast.get("daily_units", {})
    return (
        f"### Weather for {location_name}\n\n"
        f"**Observed:** {current['time']} ({forecast.get('timezone', 'local time')}) — "
        f"{weather_description}, {current.get('temperature_2m')} "
        f"{current_units.get('temperature_2m', '°C')}; feels like "
        f"{current.get('apparent_temperature')} "
        f"{current_units.get('apparent_temperature', '°C')}. "
        f"Humidity: {current.get('relative_humidity_2m')}%. "
        f"Wind: {current.get('wind_speed_10m')} "
        f"{current_units.get('wind_speed_10m', 'km/h')}; "
        f"precipitation: {current.get('precipitation')} "
        f"{current_units.get('precipitation', 'mm')}.\n\n"
        f"**Today ({daily['time'][0]}):** low "
        f"{daily['temperature_2m_min'][0]} "
        f"{daily_units.get('temperature_2m_min', '°C')}, high "
        f"{daily['temperature_2m_max'][0]} "
        f"{daily_units.get('temperature_2m_max', '°C')}; maximum precipitation "
        f"probability {daily['precipitation_probability_max'][0]}%. "
        f"Sunrise {daily['sunrise'][0]}, sunset {daily['sunset'][0]}.\n\n"
        "Source: [Open-Meteo](https://open-meteo.com/)."
    )
