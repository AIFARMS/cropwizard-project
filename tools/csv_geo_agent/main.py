"""
To deploy: beam deploy main.py:main
For testing: beam serve main.py:main
"""

from beam import endpoint, Image
from typing import Dict, Any
import os
import sys
import requests
import tempfile
import pandas as pd
import geopandas as gpd

# Add the current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Add minllm to path
minllm_path = os.path.join(current_dir, "minllm")
if minllm_path not in sys.path:
    sys.path.insert(0, minllm_path)

from geo.agent import create_geo_agent

@endpoint(
    name="csv_geo_agent",
    cpu=1,
    memory="4Gi",
    image=Image(python_version="python3.10").add_python_packages([
        "pandas",
        "geopandas",
        "matplotlib",
        "shapely",
        "pyproj",
        "fiona",
        "python-dotenv",
        "supabase",
        "requests",
        "pgeocode",
        "geopy",
        "pyyaml",
    ]),
    timeout=120,
    keep_warm_seconds=60 * 5
)
def main(**inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Endpoint that creates a geo agent with preloaded files and executes queries.
    
    Args:
        **inputs: Dictionary containing:
            - query: The user's query/request
            - file_urls: Dictionary of {filename: s3_url} for files to download
            
    Returns:
        Dictionary containing the agent's response
    """
    try:
        query = inputs.get("query", "Analyze the provided geographic data")
        file_urls = inputs.get("file_urls", {})
        
        # Create temporary directory for downloaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download and preload files as dataframes
            preloaded_dataframes = {}
            
            for filename, url in file_urls.items():
                file_path = os.path.join(temp_dir, filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Download file
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(file_path, "wb") as f:
                    f.write(response.content)
                
                # Preload into appropriate dataframe type
                try:
                    if filename.lower().endswith('.csv'):
                        df = pd.read_csv(file_path)
                        preloaded_dataframes[filename] = df
                    elif filename.lower().endswith('.geojson'):
                        gdf = gpd.read_file(file_path)
                        preloaded_dataframes[filename] = gdf
                except Exception as e:
                    print(f"Error preloading {filename}: {str(e)}")
            
            # Create agent with preloaded dataframes
            agent = create_geo_agent(preloaded_dataframes)
            
            # Add file information to query if files were provided
            if file_urls:
                file_info = "\n\nThe following files have been provided for analysis:\n"
                for filename in file_urls.keys():
                    file_info += f"- {filename}\n"
                query += file_info
            
            # Execute query with agent
            response = agent(query)
            
            return {
                "success": True,
                "response": response
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        } 