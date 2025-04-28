import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Placeholder for potential future configurations
# e.g., MODEL_NAME = os.getenv("MODEL_NAME", "default_model")

