from datetime import datetime

import requests
import json
from dotenv import load_dotenv
import os
from urllib.parse import quote

load_dotenv()

USERNAME = os.getenv("DISCOURSE_USERNAME")
PASSWORD = os.getenv("DISCOURSE_PASSWORD")
BASE_URL = os.getenv("DISCOURSE_URL")
COURSE_CATEGORY_ID = "34"


if not all([BASE_URL, USERNAME, PASSWORD]):
    print(
        "Error: Please make sure DISCOURSE_URL, DISCOURSE_USERNAME, and DISCOURSE_PASSWORD are set in your .env file."
    )
    exit()


session = requests.Session()
session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
)


print(f"Fetching CSRF token from {BASE_URL}...")
csrf_url = f"{BASE_URL}/session/csrf.json"
csrf_response = session.get(csrf_url)
csrf_response.raise_for_status()

csrf_token = csrf_response.json()["csrf"]
print(f"Successfully fetched CSRF token: {csrf_token}")

session.headers.update({"X-CSRF-Token": csrf_token})

login_payload = {"login": USERNAME, "password": PASSWORD}

print(f"Attempting to log in as '{USERNAME}'...")
login_url = f"{BASE_URL}/session.json"
login_response = session.post(login_url, data=login_payload)
login_response.raise_for_status()

if "user" in login_response.json():
    print("Login successful!")

    cookies = session.cookies.get_dict()

    if "_t" in cookies:
        print(f"Authentication token (_t) captured successfully.")

else:
    print("Login failed. Check your credentials or the website's response.")
    print("Response:", login_response.text)

