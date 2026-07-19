import os
from dotenv import load_dotenv

load_dotenv()

github_token = os.getenv("GITHUB_TOKEN")
gemini_key = os.getenv("GEMINI_API_KEY")

print("GitHub token loaded:", bool(github_token))
print("Gemini key loaded:", bool(gemini_key))