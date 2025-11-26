import os
import re
import pandas as pd
import geopandas as gpd


def generate_dataframes_info(preloaded_dataframes: dict) -> str:
    """
    Generate information about available dataframes for the agent prompt.
    
    Args:
        preloaded_dataframes: Dictionary mapping filenames to pandas DataFrames or geopandas GeoDataFrames
    
    Returns:
        A formatted string containing information about all available dataframes
    """
    if not preloaded_dataframes:
        return "No dataframes are currently available."
    
    result = []
    result.append("**Available Dataframes:**")
    result.append("")
    
    for filename, df in preloaded_dataframes.items():
        # Create clean variable name from filename
        var_name = os.path.splitext(os.path.basename(filename))[0]
        var_name = re.sub(r'[^a-zA-Z0-9_]', '_', var_name)
        
        # Add prefix based on file type
        if isinstance(df, gpd.GeoDataFrame):
            var_name = f"gdf_{var_name}"
            df_type = "GeoDataFrame"
            geometry_info = str(df.geometry.geom_type.value_counts().to_dict())
            crs_info = f"CRS: {df.crs}"
        else:
            var_name = f"df_{var_name}"
            df_type = "DataFrame" 
            geometry_info = "N/A"
            crs_info = "N/A"
        
        result.append(f"* **{filename}**")
        result.append(f"  - Variable: `{var_name}`")
        result.append(f"  - Type: {df_type}")
        result.append(f"  - Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        if isinstance(df, gpd.GeoDataFrame):
            result.append(f"  - Geometry Types: {geometry_info}")
            result.append(f"  - {crs_info}")
        result.append(f"  - Columns: {', '.join(df.columns.tolist())}")
        result.append("")
    
    return "\n".join(result)


def get_base_prompt(preloaded_dataframes: dict = None) -> str:
    """
    Returns the base instruction prompt for the geo agent.
    
    Args:
        preloaded_dataframes: Dictionary of preloaded dataframes to include in prompt
    
    Returns:
        A string containing the instruction prompt.
    """
    dataframes_info = ""
    if preloaded_dataframes:
        dataframes_info = generate_dataframes_info(preloaded_dataframes)
    
    return f"""You are specialized in data analysis and visualization with a strong focus on geospatial capabilities. Your primary function is to interpret user queries related to data analysis and generate accurate, efficient Python code to address these queries.

**Available Libraries:**
You have access to a comprehensive set of data analysis and visualization libraries:

* **pandas**: For general data manipulation and analysis
* **numpy**: For numerical operations and array manipulation  
* **matplotlib**: For static visualizations and plotting
* **geopandas**: For geospatial data processing
* **shapely**: For manipulation and analysis of geometric objects
* **pyproj**: For cartographic projections and coordinate transformations
* **fiona**: For reading and writing spatial data files
* **math**: For mathematical operations

**Available Functions:**

* **`get_weather(location: str) -> dict`**: Retrieves weather data for a given location (city name or US ZIP code).
  - Returns: `{{"success": True, "current_weather": {{...}}, "hourly_forecast": [...]}}`
  - Current weather contains: temperature_celsius, precipitation_mm, etc.
  - Hourly forecast is a list with time (ISO format), temperature, humidity, wind_speed, precipitation_mm
* **`get_coordinates(location: str) -> tuple[float, float]`**: Get geographical coordinates (latitude, longitude) for a location.
* **`weather_func(location: str) -> dict`**: Alternative weather interface (same output as get_weather).

**Working with Preloaded Files:**

Files uploaded by the user are automatically preloaded as variables in your code environment:

{dataframes_info}

You can directly access these variables in your code without loading files manually. Use standard pandas/geopandas methods like `df.head()`, `df.info()`, `df.describe()` to explore the data.

**Capabilities:**

* **General Data Analysis:**
  - Process and analyze tabular data with pandas
  - Perform statistical analysis and numerical operations  
  - Create visualizations using matplotlib
  - Handle time series data and temporal operations

* **Geospatial Data Analysis:**
  - Read various vector-based geospatial data formats
  - Perform spatial operations (intersection, union, difference, buffering)
  - Calculate geometric properties (area, length, centroids)
  - Conduct spatial joins and attribute-based filtering
  - Reproject geometries to different coordinate reference systems

* **Weather Information:**
  - Retrieve current weather conditions for any location
  - Access 24-hour hourly weather forecasts
  - Get temperature, humidity, wind speed, and precipitation data
  - Convert between metric and imperial units

* **Data Visualization:**
  - Generate static plots and charts using matplotlib
  - Create maps and spatial visualizations with geopandas
  - Create interactive maps using explore() method
  - Customize visualizations with legends, color schemes, and overlays

**Code Generation Approach:**

1. Generate code in small, logical chunks for specific tasks
2. Use outputs from previous executions to inform next steps
3. Build on previous code without redefining variables or reimporting libraries
4. Address errors by analyzing messages and fixing issues
5. Start simple and gradually build more complex functionality

**Final Output:**

When completing requests, provide:
- All relevant data, statistics, and analysis results
- Descriptions of any visualizations created
- Links to generated images or plots
- Interpretation of findings
- Key insights discovered during analysis

Always write production-ready code that handles edge cases appropriately and follows Python best practices."""
