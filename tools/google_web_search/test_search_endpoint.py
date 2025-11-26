#!/usr/bin/env python3
"""
Script to test the local Beam search endpoint.
"""

import requests
import json
import argparse
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

# Constants
ENDPOINT_URL = os.getenv("BEAM_ENDPOINT_URL")
API_KEY = os.getenv("BEAM_API_KEY")

def test_search_endpoint(query):
    """Test the search endpoint with the given query."""
    
    if not ENDPOINT_URL or not API_KEY:
        print("Error: BEAM_ENDPOINT_URL and BEAM_API_KEY must be set in .env file")
        return
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    payload = {
        'query': query
    }
    
    print(f"Sending request to {ENDPOINT_URL} with query: '{query}'")
    
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload)
        
        # Print status code
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            # Pretty print JSON response if valid
            try:
                result = response.json()
                print("\nResponse:")
                print(json.dumps(result, indent=2))
            except json.JSONDecodeError:
                # If not JSON, print text response
                print("\nResponse (text):")
                print(response.text)
        else:
            print(f"Error: {response.text}")
    
    except Exception as e:
        print(f"Exception occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Beam search endpoint")
    parser.add_argument("query", nargs="?", default="What is the fastest electric car?", 
                      help="The search query to send (default: 'What is the fastest electric car?')")
    
    args = parser.parse_args()
    test_search_endpoint(args.query) 