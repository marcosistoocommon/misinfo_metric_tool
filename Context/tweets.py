import os
from pathlib import Path

from dotenv import load_dotenv
from Scweet import Scweet

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

AUTH_TOKEN = os.environ["SCWEET_AUTH_TOKEN"]

s = Scweet(
    auth_token=AUTH_TOKEN,
)

# Search for tweets about Bitcoin from 2025 onward
tweets = s.search("bitcoin", since="2025-01-01", limit=200, save=True)
print(f"Collected {len(tweets)} tweets")

