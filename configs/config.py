import os
from dotenv import load_dotenv

# For local only
environment = os.environ.get("ENVIRONMENT", "local")

path = os.path.dirname(os.path.abspath(__file__))

resp = load_dotenv(f"./secrets/environments/{environment}.env")
print(f"Loaded env variables: {resp}")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
