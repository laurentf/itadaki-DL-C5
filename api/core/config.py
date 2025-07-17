import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
try:
    load_dotenv()
except:
    pass  # ✅ FIX: Ignore si .env n'existe pas (évite les timeouts)

# API Settings
API_V1_STR = "/api/v1"
PROJECT_NAME = "Itadaki API"

# CORS
CORS_ORIGINS = ["*"]  # In production, replace with specific origins

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Itadaki specific settings
MODELS_PATH = os.getenv("MODELS_PATH", "./models")
IMAGES_PATH = os.getenv("IMAGES_PATH", "./test_recipes")
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "5242880"))  # 5MB default 