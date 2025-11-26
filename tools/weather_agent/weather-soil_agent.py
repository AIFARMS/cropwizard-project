#beam deploy weather-soil_agent.py:main
#beam serve weather-soil_agent.py

#weather query for sample farm -> tool collects relevant up-to-date weather forecasts, selects relevant variables that will help answer query
#-> uses that information to generate a detailed analysis/recommendation

#When should I spray my field at River Bend in the coming week or so? eg.

from beam import endpoint, Image
from typing import Any, Dict
import os
import pandas as pd
from dotenv import load_dotenv
import requests_cache
from retry_requests import retry
import openmeteo_requests
from google import genai

# Load environment variables and set up clients.
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
openmeteo = openmeteo_requests.Client()

# -------------------------------
# Global Parameter Lists (from Open-Meteo API)
# -------------------------------
daily_params = [
    "weather_code", "temperature_2m_max", "apparent_temperature_max", "temperature_2m_min",
    "apparent_temperature_min", "uv_index_clear_sky_max", "uv_index_max", "sunshine_duration",
    "daylight_duration", "sunset", "sunrise", "et0_fao_evapotranspiration_sum",
    "temperature_2m_mean", "apparent_temperature_mean", "cape_mean", "cape_min", "cape_max",
    "cloud_cover_max", "cloud_cover_mean", "cloud_cover_min", "dew_point_2m_mean",
    "dew_point_2m_max", "dew_point_2m_min", "pressure_msl_min", "pressure_msl_max",
    "pressure_msl_mean", "relative_humidity_2m_min", "snowfall_water_equivalent_sum",
    "relative_humidity_2m_max", "relative_humidity_2m_mean", "precipitation_probability_mean",
    "precipitation_probability_min", "leaf_wetness_probability_mean",
    "growing_degree_days_base_0_limit_50", "surface_pressure_mean", "surface_pressure_max",
    "surface_pressure_min", "updraft_max", "visibility_mean", "visibility_min",
    "visibility_max", "winddirection_10m_dominant", "wind_gusts_10m_mean",
    "wind_speed_10m_mean", "wind_gusts_10m_min", "wind_speed_10m_min",
    "vapour_pressure_deficit_max", "wet_bulb_temperature_2m_min", "wet_bulb_temperature_2m_max",
    "wet_bulb_temperature_2m_mean", "precipitation_probability_max", "precipitation_sum",
    "precipitation_hours", "snowfall_sum", "showers_sum", "rain_sum", "wind_speed_10m_max",
    "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum",
    "et0_fao_evapotranspiration"
]

hourly_params = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "precipitation", "precipitation_probability", "snowfall", "showers", "rain",
    "snow_depth", "pressure_msl", "cloud_cover", "surface_pressure",
    "temperature_150hPa", "temperature_50hPa", "temperature_1000hPa", "temperature_800hPa",
    "temperature_925hPa", "temperature_500hPa", "temperature_250hPa", "temperature_30hPa",
    "temperature_100hPa", "temperature_200hPa", "temperature_70hPa", "temperature_400hPa",
    "temperature_700hPa", "temperature_900hPa", "temperature_975hPa", "temperature_950hPa",
    "temperature_850hPa", "temperature_600hPa", "temperature_300hPa", "relative_humidity_1000hPa",
    "relative_humidity_925hPa", "relative_humidity_800hPa", "relative_humidity_500hPa",
    "relative_humidity_250hPa", "relative_humidity_100hPa", "relative_humidity_30hPa",
    "relative_humidity_200hPa", "relative_humidity_70hPa", "relative_humidity_400hPa",
    "relative_humidity_900hPa", "relative_humidity_700hPa", "relative_humidity_975hPa",
    "relative_humidity_950hPa", "relative_humidity_850hPa", "relative_humidity_600hPa",
    "relative_humidity_300hPa", "relative_humidity_150hPa", "relative_humidity_50hPa",
    "cloud_cover_1000hPa", "cloud_cover_925hPa", "cloud_cover_800hPa", "cloud_cover_250hPa",
    "cloud_cover_500hPa", "cloud_cover_100hPa", "cloud_cover_30hPa", "cloud_cover_70hPa",
    "cloud_cover_400hPa", "cloud_cover_200hPa", "cloud_cover_700hPa", "cloud_cover_900hPa",
    "cloud_cover_975hPa", "cloud_cover_850hPa", "cloud_cover_950hPa", "cloud_cover_300hPa",
    "cloud_cover_600hPa", "cloud_cover_150hPa", "cloud_cover_50hPa", "weather_code",
    "cloud_cover_mid", "cloud_cover_low", "cloud_cover_high", "visibility", "evapotranspiration",
    "et0_fao_evapotranspiration", "vapour_pressure_deficit", "temperature_80m",
    "temperature_120m", "temperature_180hPa", "wind_gusts_10m", "wind_direction_120m",
    "wind_direction_180m", "wind_speed_120m", "wind_speed_180m", "wind_direction_10m",
    "wind_direction_80m", "wind_speed_80m", "wind_speed_10m", "soil_temperature_0cm",
    "soil_temperature_6cm", "soil_temperature_18cm", "soil_temperature_54cm",
    "soil_moisture_1_to_3cm", "soil_moisture_0_to_1cm", "soil_moisture_3_to_9cm",
    "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm", "cape", "lifted_index", "uv_index",
    "uv_index_clear_sky", "is_day", "sunshine_duration", "shortwave_radiation",
    "diffuse_radiation", "global_tilted_irradiance", "shortwave_radiation_instant",
    "diffuse_radiation_instant", "global_tilted_irradiance_instant",
    "direct_normal_irradiance_instant", "terrestrial_radiation_instant", "direct_radiation_instant",
    "direct_normal_irradiance", "direct_radiation"
]

# -------------------------------
# Pipeline Functions
# -------------------------------
def select_relevant_farms(farms, user_query):
    """
    Gemini selects the most relevant farms based on the user query.
    Returns a list of selected farm names.
    """
    prompt = "You are an expert agriculture specialist. Below is a list of farms with their details:\n"
    for farm in farms:
        prompt += f"- {farm['name']}: {farm['description']}, located at ({farm['latitude']}, {farm['longitude']})\n"
    prompt += f'\nBased on the following user query:\n"{user_query}"\n\n' \
              "List the names of the farms most relevant to this query, separated by commas."
    response = gemini_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    selected_farm_names = [name.strip() for name in response.text.strip().split(",")]
    return selected_farm_names

def get_farm_weather_data(latitude, longitude):
    """
    Fetches weather/soil data (hourly and daily) for the given location.
    Processes the API response into two DataFrames (hourly and daily).
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": daily_params,
        "hourly": hourly_params,
        "current": [
            "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "surface_pressure",
            "pressure_msl", "cloud_cover", "weather_code", "precipitation", "rain", "showers",
            "snowfall", "is_day", "apparent_temperature", "relative_humidity_2m", "temperature_2m"
        ],
        "minutely_15": [
            "temperature_2m", "precipitation", "freezing_level_height", "wind_speed_80m", "visibility",
            "shortwave_radiation", "global_tilted_irradiance", "diffuse_radiation_instant",
            "direct_normal_irradiance_instant", "terrestrial_radiation", "direct_radiation",
            "relative_humidity_2m", "rain", "sunshine_duration", "wind_direction_10m", "cape",
            "wind_direction_80m", "weather_code", "snowfall", "dew_point_2m", "lightning_potential",
            "diffuse_radiation", "shortwave_radiation_instant", "global_tilted_irradiance_instant",
            "terrestrial_radiation_instant", "direct_radiation_instant", "direct_normal_irradiance",
            "is_day", "wind_speed_10m", "wind_gusts_10m", "apparent_temperature", "snowfall_height"
        ]
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Process hourly data.
    hourly = response.Hourly()
    hr_start = pd.to_datetime(hourly.Time(), unit="s", utc=True)
    hr_end = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
    hr_interval = pd.Timedelta(seconds=hourly.Interval())
    hourly_dates = pd.date_range(start=hr_start, end=hr_end, freq=hr_interval, inclusive="left")
    hourly_data = {"date": hourly_dates}
    for i, param in enumerate(hourly_params):
        hourly_data[param] = hourly.Variables(i).ValuesAsNumpy()
    hourly_df = pd.DataFrame(hourly_data)

    # Process daily data.
    daily = response.Daily()
    d_start = pd.to_datetime(daily.Time(), unit="s", utc=True)
    d_end = pd.to_datetime(daily.TimeEnd(), unit="s", utc=True)
    d_interval = pd.Timedelta(seconds=daily.Interval())
    daily_dates = pd.date_range(start=d_start, end=d_end, freq=d_interval, inclusive="left")
    daily_data = {"date": daily_dates}
    for i, param in enumerate(daily_params):
        daily_data[param] = daily.Variables(i).ValuesAsNumpy()
    daily_df = pd.DataFrame(daily_data)

    return hourly_df, daily_df

def select_relevant_variables(farm, weather_hourly_df, weather_daily_df, user_query):
    """
    Uses snapshots of both hourly and daily weather/soil data along with farm details
    to ask Gemini for the most critical variables.
    """
    hourly_summary = weather_hourly_df.head().to_string(index=False)
    daily_summary = weather_daily_df.head().to_string(index=False)
    prompt = f"""You are an expert agriculture specialist.
    Farm: {farm['name']}
    Description: {farm['description']}
    Location: ({farm['latitude']}, {farm['longitude']})

    Here is a snapshot of the hourly weather/soil data for the farm:
    {hourly_summary}

    Here is a snapshot of the daily weather/soil data for the farm:
    {daily_summary}

    User Query: "{user_query}"

    Based on the above, please list the most critical weather and soil variables (from the full set provided)
    that are relevant to addressing this query, and include a brief explanation for each.
    Return your answer as a plain text list.
    """
    response = gemini_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text

def get_targeted_weather_data(latitude, longitude, hourly_vars, daily_vars):
    """
    Gets a more focused dataset from the API using only the recommended variables.
    Processes the response into two DataFrames: one for hourly and one for daily data.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": hourly_vars,
        "daily": daily_vars
    }
    response = openmeteo.weather_api(url, params=params)[0]
    
    # Process targeted hourly data.
    hourly = response.Hourly()
    hr_start = pd.to_datetime(hourly.Time(), unit="s", utc=True)
    hr_end = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
    hr_interval = pd.Timedelta(seconds=hourly.Interval())
    hourly_dates = pd.date_range(start=hr_start, end=hr_end, freq=hr_interval, inclusive="left")
    targeted_hourly_data = {"date": hourly_dates}
    for i, var in enumerate(hourly_vars):
        targeted_hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
    targeted_hourly_df = pd.DataFrame(targeted_hourly_data)
    
    # Process targeted daily data.
    daily = response.Daily()
    d_start = pd.to_datetime(daily.Time(), unit="s", utc=True)
    d_end = pd.to_datetime(daily.TimeEnd(), unit="s", utc=True)
    d_interval = pd.Timedelta(seconds=daily.Interval())
    daily_dates = pd.date_range(start=d_start, end=d_end, freq=d_interval, inclusive="left")
    targeted_daily_data = {"date": daily_dates}
    for i, var in enumerate(daily_vars):
        targeted_daily_data[var] = daily.Variables(i).ValuesAsNumpy()
    targeted_daily_df = pd.DataFrame(targeted_daily_data)
    
    return targeted_hourly_df, targeted_daily_df

def build_final_prompt(farm, key_variables_text, targeted_hourly_summary, targeted_daily_summary, user_query):
    """
    Builds the final prompt to be sent for detailed analysis.
    Includes farm details, LLM key variable output, and summaries for both targeted datasets.
    """
    final_prompt = f"""Farm: {farm['name']}
    Description: {farm['description']}
    Location: ({farm['latitude']}, {farm['longitude']})

    [The targeted weather/soil dataset is available in the attached files or summaries below.]

    Key Variables Relevant to the Query:
    {key_variables_text}

    Targeted Hourly Data Summary:
    {targeted_hourly_summary}

    Targeted Daily Data Summary:
    {targeted_daily_summary}

    User Query:
    {user_query}

    Please provide a detailed analysis and tailored recommendations for this farm based on the above information.
    """
    return final_prompt

# -------------------------------
# Sample Farm Data
# -------------------------------
farms = [
    {
        "name": "Green Acres",
        "latitude": 52.52,
        "longitude": 13.41,
        "description": "Organic vegetable farm with diverse crops and emphasis on soil fertility."
    },
    {
        "name": "Sunny Fields",
        "latitude": 48.85,
        "longitude": 2.35,
        "description": "Large-scale wheat and barley production with extensive irrigation systems."
    },
    {
        "name": "River Bend",
        "latitude": 40.71,
        "longitude": -74.01,
        "description": "Dairy and livestock operation facing potential water stress during hot spells."
    }
]

# -------------------------------
# Beam Endpoint: main
# -------------------------------
@endpoint(
    name="farm_specific_weather_agent",
    cpu=1,
    memory="2Gi",
    image=Image(python_version="python3.10").add_python_packages([
        "pandas",
        "python-dotenv",
        "openmeteo_requests",
        "retry_requests",
        "requests_cache",
        "google-genai"
    ]),
    timeout=60,
    keep_warm_seconds=60 * 3
)
def main(**inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Weather agent endpoint.
    Expects an input with a "user_query" string.
    
    Example input:
    {
        "user_query": "What is the wind speed in the River Bend farm and how will it affect crops?"
    }
    """
    user_query = inputs.get("user_query", "What is the wind speed in the River Bend farm and how will it affect crops?")
    
    # Step 1: Select and filter relevant farms using Gemini.
    selected_farm_names = select_relevant_farms(farms, user_query)
    relevant_farms = [farm for farm in farms if farm["name"] in selected_farm_names]
    final_recommendations = ""
    
    for farm in relevant_farms:
        # Step 2: Fetch full weather/soil data.
        hourly_df, daily_df = get_farm_weather_data(farm["latitude"], farm["longitude"])
        
        # Step 3: Ask Gemini for the most critical variables.
        key_vars_text = select_relevant_variables(farm, hourly_df, daily_df, user_query)
        
        recommended_vars = []
        for line in key_vars_text.splitlines():
            line = line.strip()
            if line.startswith("*"):
                parts = line.split("**")
                if len(parts) >= 2:
                    var_name = parts[1].strip().rstrip(':')
                    recommended_vars.append(var_name)
        filtered_hourly_vars = [var for var in recommended_vars if var in hourly_params]
        filtered_daily_vars = [var for var in recommended_vars if var in daily_params]
        
        # Fallback defaults if necessary.
        if not filtered_hourly_vars:
            filtered_hourly_vars = ["temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover", "wind_speed_10m"]
        if not filtered_daily_vars:
            filtered_daily_vars = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "sunshine_duration"]
        
        # Step 4: Fetch targeted weather data.
        targeted_hourly_df, targeted_daily_df = get_targeted_weather_data(farm["latitude"], farm["longitude"], filtered_hourly_vars, filtered_daily_vars)
        targeted_hourly_summary = targeted_hourly_df.to_string(index=False)
        targeted_daily_summary = targeted_daily_df.to_string(index=False)
        
        # Step 5: Build final prompt for detailed analysis.
        final_prompt = build_final_prompt(farm, key_vars_text, targeted_hourly_summary, targeted_daily_summary, user_query)
        final_recommendations += final_prompt + "\n" + "="*50 + "\n"
    
    return {"final_prompt_template": final_recommendations}

if __name__ == "__main__":
    # For local testing.
    main()
