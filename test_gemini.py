import httpx
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

async def test_models():
    models_to_test = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro",
        "gemini-pro-vision"
    ]
    
    async with httpx.AsyncClient() as client:
        for model in models_to_test:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": "Hello"}]}]
            }
            try:
                resp = await client.post(url, json=payload)
                print(f"Model {model}: Status {resp.status_code}")
            except Exception as e:
                print(f"Model {model}: Error {e}")

if __name__ == "__main__":
    asyncio.run(test_models())
