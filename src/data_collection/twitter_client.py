import os
from dotenv import load_dotenv
from pathlib import Path
import requests

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=str(env_path))

class TwitterClient:
    def __init__(self):
        self.url = "https://api.twitterapi.io/twitter/tweet/advanced_search?query="
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("API_KEY not found in environment variables")
        self.headers = {"X-API-Key": api_key}

    def fetch_tweets(self, query: str) -> dict:
        full_url = f"{self.url}{query}"
        response = requests.get(full_url, headers=self.headers)
        response.raise_for_status()
        return response.json()