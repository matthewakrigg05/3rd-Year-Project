import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=str(env_path))

url = "https://api.twitterapi.io/twitter/tweet/advanced_search?query=politics"

headers = {
    "X-API-Key": f"{os.getenv("API_KEY")}"
}