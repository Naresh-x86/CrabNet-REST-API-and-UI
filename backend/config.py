"""
Configuration for CrabNet API

Store API keys and other configuration values here.
"""
import os

# Materials Project API Configuration
# Replace with your actual API key
MP_API_KEY = os.environ.get("MP_API_KEY", "5YFZ0vBryo4VxoQZAi3DrqDED5GYwA18")
MP_BASE_URL = "https://api.materialsproject.org"

# Server Configuration
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", 8000))

# CrabNet Configuration
CRABNET_BATCH_SIZE = 512
MAX_AUTOCOMPLETE_RESULTS = 10
MAX_RETRIEVE_RESULTS = 10
MAX_SIMILAR_MATERIALS = 5
