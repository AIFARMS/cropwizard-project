import os
import io
import sys
import contextlib
import traceback
import re
import base64
from datetime import datetime
from typing import Dict, Any

# Import minagent
from minagent import Agent, configure_llm

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import shapely
import pyproj
import fiona
from supabase import create_client, Client

from dotenv import load_dotenv
from .prompts import get_base_prompt
from .weather import main as weather_func, get_weather, get_coordinates

load_dotenv()

# Configure LLM for minagent
openai_key = os.getenv('OPENAI_API_KEY')
openrouter_key = os.getenv('OPENROUTER_API_KEY')

if not openai_key and not openrouter_key:
    raise ValueError("Either OPENAI_API_KEY or OPENROUTER_API_KEY must be set in .env file")

configure_llm(
    provider='openai' if openai_key else 'openrouter',
    api_key=openai_key or openrouter_key,
    model='gpt-4o' if openai_key else 'anthropic/claude-3.5-sonnet'
)

# Global execution environment
_execution_globals = None
_execution_locals = None


def setup_execution_environment(preloaded_dataframes: dict):
    """
    Set up the persistent execution environment with preloaded dataframes.
    
    Args:
        preloaded_dataframes: Dictionary mapping filenames to pandas DataFrames or geopandas GeoDataFrames
    """
    global _execution_globals, _execution_locals
    
    # Initialize execution environment
    _execution_globals = {
        "pd": pd,
        "np": np,
        "gpd": gpd,
        "plt": plt,
        "Figure": Figure,
        "shapely": shapely,
        "pyproj": pyproj,
        "fiona": fiona,
        "os": os,
        "re": re,
        "__builtins__": __builtins__,
        # Add weather functions
        "get_weather": get_weather,
        "get_coordinates": get_coordinates,
        "weather_func": weather_func,
    }
    
    _execution_locals = {}
    
    # Add preloaded dataframes to execution environment
    if preloaded_dataframes:
        for filename, df in preloaded_dataframes.items():
            # Create clean variable name from filename
            var_name = os.path.splitext(os.path.basename(filename))[0]
            var_name = re.sub(r'[^a-zA-Z0-9_]', '_', var_name)
            
            # Add prefix based on file type
            if isinstance(df, gpd.GeoDataFrame):
                var_name = f"gdf_{var_name}"
            else:
                var_name = f"df_{var_name}"
            
            # Add to execution environment
            _execution_globals[var_name] = df


def run_code(reasoning: str, code: str) -> str:
    """
    Execute Python code in the persistent environment and return results.
    
    This tool allows the agent to execute Python code for data analysis and visualization.
    The execution environment persists between calls, maintaining variable state.
    
    Args:
        reasoning: Brief explanation of what the code will do
        code: Python code to execute
        
    Returns:
        String containing execution output, results, and any error messages
    """
    global _execution_globals, _execution_locals
    
    if _execution_globals is None:
        return "Error: Execution environment not initialized. Please contact support."
    
    print(f"Reasoning: {reasoning}")
    
    # Capture stdout and stderr
    stdout_buffer = io.StringIO()
    
    try:
        # Handle special commands
        if code.strip() in ["list_dataframes()", "list_dataframes"]:
            # List available dataframes
            dataframes = []
            for name, var in {**_execution_globals, **_execution_locals}.items():
                if isinstance(var, (pd.DataFrame, gpd.GeoDataFrame)) and not name.startswith("_"):
                    df_type = "GeoDataFrame" if isinstance(var, gpd.GeoDataFrame) else "DataFrame"
                    dataframes.append(f"- {name}: {df_type} ({var.shape[0]} rows × {var.shape[1]} columns)")
            
            if dataframes:
                return "Available dataframes:\n" + "\n".join(dataframes)
            else:
                return "No dataframes available in the execution environment."
        
        # Execute the code
        with (
            contextlib.redirect_stdout(stdout_buffer),
            contextlib.redirect_stderr(stdout_buffer),
        ):
            exec(code, _execution_globals, _execution_locals)
            
            # Handle figure outputs
            figure_count = 0
            supabase = None
            
            # Initialize Supabase for image uploads
            try:
                supabase_url = os.environ.get("SUPABASE_URL")
                supabase_key = os.environ.get("SUPABASE_ANON") or os.environ.get("SUPABASE_SECRET")
                
                if supabase_url and supabase_key:
                    supabase = create_client(supabase_url, supabase_key)
            except Exception as e:
                stdout_buffer.write(f"\n[Supabase initialization failed: {str(e)}]")
            
            # Save any figures that were created
            for var_name, var in list(_execution_locals.items()) + list(_execution_globals.items()):
                if isinstance(var, plt.Figure):
                    figure_count += 1
                    img_buffer = io.BytesIO()
                    var.savefig(img_buffer, format="png", dpi=150, bbox_inches='tight')
                    plt.close(var)
                    
                    if supabase:
                        try:
                            img_buffer.seek(0)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"figure_{timestamp}_{figure_count}.png"
                            
                            supabase.storage.from_("outputs").upload(
                                filename,
                                img_buffer.getvalue(),
                                {"content-type": "image/png"},
                            )
                            
                            file_url = supabase.storage.from_("outputs").get_public_url(filename)
                            stdout_buffer.write(f"\n[Figure {figure_count} saved: {file_url}]")
                        except Exception as e:
                            stdout_buffer.write(f"\n[Error saving figure {figure_count}: {str(e)}]")
                    else:
                        stdout_buffer.write(f"\n[Figure {figure_count} created but storage not available]")
        
        # Get output
        output = stdout_buffer.getvalue()
        
        # If no output, show some key variables
        if not output.strip():
            result_vars = []
            for var_name, var_value in _execution_locals.items():
                if var_name.startswith("_") or callable(var_value):
                    continue
                    
                if isinstance(var_value, (pd.DataFrame, gpd.GeoDataFrame)):
                    result_vars.append(f"\n--- {var_name} ---\n{var_value.head().to_string()}")
                    if len(var_value) > 5:
                        result_vars.append(f"[{len(var_value)-5} more rows]")
                elif not isinstance(var_value, type):
                    var_str = str(var_value)
                    if len(var_str) > 500:
                        var_str = var_str[:500] + "... [truncated]"
                    result_vars.append(f"{var_name} = {var_str}")
            
            if result_vars:
                output = "\n".join(result_vars)
        
        # Format final response
        response = f"Code executed:\n```python\n{code}\n```\n\nOutput:\n{output}"
        
        # Add context info
        variable_names = [name for name in _execution_locals.keys() 
                         if not name.startswith("_") and not callable(_execution_locals[name])]
        if variable_names:
            response += f"\n\nDefined variables: {', '.join(variable_names)}"
        
        return response
        
    except Exception as e:
        # Format error message
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        
        # Filter traceback to exclude internal runner details
        filtered_tb = []
        for line in tb_lines:
            if "run_code" not in line:
                filtered_tb.append(line)
        
        error_message = "".join(filtered_tb)
        
        return f"Code executed:\n```python\n{code}\n```\n\nError:\n{error_message}"
        
    finally:
        plt.close("all")
        stdout_buffer.close()


def create_geo_agent(preloaded_dataframes: dict) -> Agent:
    """
    Create and configure a geo analysis agent with preloaded dataframes.
    
    Args:
        preloaded_dataframes: Dictionary mapping filenames to pandas DataFrames or geopandas GeoDataFrames
        
    Returns:
        Configured Agent instance ready to process queries
    """
    # Setup execution environment with preloaded data
    setup_execution_environment(preloaded_dataframes)
    
    # Get base prompt with dataframe information
    base_prompt = get_base_prompt(preloaded_dataframes)
    
    # Create agent with tools
    agent = Agent(
        base_prompt=base_prompt,
        tools=[run_code],
        logging=True
    )
    
    return agent
