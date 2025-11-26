"""
To deploy:
    beam deploy weather-soil_agent-RAG.py:main

For local testing:
    beam serve weather-soil_agent-RAG.py:main
"""

#most optimal time to spray valor for garbanzo beans
'''
soil_temperature_0cm  soil_moisture_0_to_1cm  precipitation  temperature_2m  weather_code  relative_humidity_2m  wind_speed_10m  soil_temperature_6cm  wind_direction_10m  et0_fao_evapotranspiration
'''
#most optimal planting time for canola in my farm, river bend farm
'''
temperature_2m  soil_temperature_0cm  precipitation  weather_code  soil_moisture_0_to_1cm
'''

from beam import endpoint, Image
from typing import Any, Dict
import os
import re
import pandas as pd
import requests
from dotenv import load_dotenv
import requests_cache
import openmeteo_requests
from google import genai
import json

# Load environment variables and set up clients.
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
UIUC_API_KEY   = os.getenv("CROPWIZARD_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
openmeteo      = openmeteo_requests.Client()
requests_cache.install_cache("weather_cache")

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
    prompt = "You are an expert agriculture specialist. Below is a list of farms with their details:\n"
    for farm in farms:
        prompt += f"- {farm['name']}: {farm['description']}, located at ({farm['latitude']}, {farm['longitude']})\n"
    prompt += f'\nBased on the following user query:\n"{user_query}"\n\n'
    prompt += "List the names of the farms most relevant to this query, separated by commas."
    resp = gemini_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return [name.strip() for name in resp.text.split(",")]

def get_farm_weather_data(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "daily": daily_params, "hourly": hourly_params}
    resp = openmeteo.weather_api(url, params=params)[0]

    # Hourly
    hourly = resp.Hourly()
    hr_dates = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
    hourly_df = pd.DataFrame({
        "date": hr_dates,
        **{param: hourly.Variables(i).ValuesAsNumpy() for i, param in enumerate(hourly_params)}
    })

    # Daily
    daily = resp.Daily()
    d_dates = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )
    daily_df = pd.DataFrame({
        "date": d_dates,
        **{param: daily.Variables(i).ValuesAsNumpy() for i, param in enumerate(daily_params)}
    })

    return hourly_df, daily_df

def select_relevant_variables(farm, hourly_df, daily_df, user_query):
    # 1) Retrieve RAG contexts from UIUC
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": (
                "You are a helpful assistant. Follow instructions carefully. "
                "Answer using markdown. If no spray timing info in docs, fall back to general knowledge."
            )},
            {"role": "user", "content": user_query}
        ],
        "api_key": UIUC_API_KEY,
        "course_name": "cropwizard-1.5",
        "stream": False,
        "temperature": 0.1,
        "retrieval_only": True
    }
    r = requests.post("https://uiuc.chat/api/chat-api/chat", json=payload)
    r.raise_for_status()
    data = r.json()
    contexts = data.get("contexts", [])
    retrieval_content = "\n\n".join(c.get("text", "") for c in contexts if isinstance(c, dict))

    # 2) Build and send Gemini prompt
    prompt = f"""
You are an expert agriculture specialist. Use ONLY these variables.

HOURLY: {', '.join(hourly_params)}
DAILY:  {', '.join(daily_params)}

Passages:
{retrieval_content}

Farm: {farm['name']}
Desc: {farm['description']}
Loc:  ({farm['latitude']}, {farm['longitude']})

Query: "{user_query}"

Based on the passages, pick the critical variables *only* from the lists above.
Return as a markdown list where each item is:

*   **<VAR>**: A one‑sentence description of the variable’s relevance; then a numbered, step‑by‑step plan for how to use that variable’s data to directly answer the user’s query.
Do not include any other text.
"""
    resp = gemini_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return resp.text

def get_targeted_weather_data(lat, lon, hourly_vars, daily_vars):
    url = "https://api.open-meteo.com/v1/forecast"
    resp = openmeteo.weather_api(url, params={
        "latitude": lat, "longitude": lon,
        "hourly":  hourly_vars,
        "daily":   daily_vars
    })[0]

    # Hourly
    hourly = resp.Hourly()
    hr_dates = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
    th_df = pd.DataFrame({
        "date": hr_dates,
        **{var: hourly.Variables(i).ValuesAsNumpy() for i, var in enumerate(hourly_vars)}
    })

    # Daily
    daily = resp.Daily()
    d_dates = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )
    td_df = pd.DataFrame({
        "date": d_dates,
        **{var: daily.Variables(i).ValuesAsNumpy() for i, var in enumerate(daily_vars)}
    })

    return th_df, td_df

def build_final_prompt(farm, key_vars_text, th_summary, td_summary, user_query):
    return f"""Farm: {farm['name']}
Description: {farm['description']}
Location:    ({farm['latitude']}, {farm['longitude']})

Key Variables Relevant to the Query:
{key_vars_text}

Targeted Hourly Data Summary:
{th_summary}

Targeted Daily Data Summary:
{td_summary}

User Query:
{user_query}

Please provide a very targeted answer to the user query based on the above information. Clearly and fully answer their question, using the data provided. If you cannot answer, say so.
"""

# -------------------------------
# Beam Endpoint: main
# -------------------------------
@endpoint(
    name="weather-soil-agent-RAG",
    cpu=1,
    memory="2Gi",
    image=Image(python_version="python3.10").add_python_packages([
        "pandas",
        "python-dotenv",
        "openmeteo_requests",
        "retry_requests",
        "requests_cache",
        "google-genai",
        "requests"
    ]),
    timeout=60,
    keep_warm_seconds=60 * 3
)
def main(**inputs: Dict[str, Any]) -> Dict[str, Any]:
    # 1) Grab the user query out of the generic inputs dict
    user_query = inputs.get("user_query", "").strip()
    
    # 2) Define your sample farms
    farms = [
        {"name": "Green Acres",  "latitude": 52.52, "longitude": 13.41, "description": "Organic vegetable farm…"},
        {"name": "Sunny Fields", "latitude": 48.85, "longitude": 2.35,  "description": "Large-scale wheat/barley…"},
        {"name": "River Bend",   "latitude": 40.71, "longitude": -74.01,"description": "Dairy/livestock operation…"}
    ]

    # 3) Filter to just the farms relevant to this query
    selected = select_relevant_farms(farms, user_query)
    outputs = []

    for farm in [f for f in farms if f["name"] in selected]:
        # 4) Fetch full hourly + daily weather + soil
        h_df, d_df = get_farm_weather_data(farm["latitude"], farm["longitude"])

        # 5) Let RAG+Gemini choose the critical variables
        key_vars_text = select_relevant_variables(farm, h_df, d_df, user_query)

        # 6) Parse out the variable names from the markdown
        recs = []
        for line in key_vars_text.splitlines():
            if line.strip().startswith("*"):
                m = re.search(r"\*\*(.+?)\*\*", line)
                if not m:
                    continue
                parts = re.split(r"\s*[+,]\s*", m.group(1))
                for var in parts:
                    cleaned = re.sub(r"\s*\(.*?\)$", "", var).strip().lower()
                    recs.append(cleaned)

        # 7) Filter to the known hourly/daily params (with sensible fallbacks)
        filtered_h = [v for v in recs if v in hourly_params]
        filtered_d = [v for v in recs if v in daily_params]
        if not filtered_h:
            filtered_h = ["temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover", "wind_speed_10m"]
        if not filtered_d:
            filtered_d = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "sunshine_duration"]

        # 8) Fetch just those targeted variables
        th_df, td_df = get_targeted_weather_data(
            farm["latitude"], farm["longitude"], filtered_h, filtered_d
        )

        # 9) Build the final LLM prompt for detailed, farm‑specific advice
        final_prompt = build_final_prompt(
            farm,
            key_vars_text,
            th_df.to_string(index=False),
            td_df.to_string(index=False),
            user_query
        )

        outputs.append({
            "farm": farm["name"],
            "prompt": final_prompt
        })

    return {
        "success": True,
        "results": outputs
    }


if __name__ == "__main__":
    example_query = "What is the most optimal spraying time at River Bend Farm in the next couple of days?"
    # When calling locally, wrap the example into the inputs dict:
    print(main(user_query=example_query))
