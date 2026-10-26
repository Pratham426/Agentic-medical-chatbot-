
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit("GROQ_API_KEY not found in .env")

response = requests.post(
    "https://api.groq.com/openai/v1/audio/speech",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "canopylabs/orpheus-v1-english",
        "voice": "troy",
        "input": "Hello! This is a test of the medical assistant.",
        "response_format": "wav"
    },
    timeout=90
)

print("HTTP status:", response.status_code)

if response.ok:
    with open("test_speech.wav", "wb") as f:
        f.write(response.content)
    print("SUCCESS: Audio saved to test_speech.wav")
else:
    print("API error:", response.text)
