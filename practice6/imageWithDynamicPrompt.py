import openai
import sys
import os

# Add parent directory to sys.path to import config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import my_key
from openai import OpenAI
import base64

client = OpenAI(api_key=my_key)

# Ask the user for a prompt
user_prompt = input("Describe the image you want to generate: ")

# Call the image generation API
response = client.images.generate(
    model="dall-e-3",
    prompt=user_prompt,
    size="1024x1024",
    response_format="b64_json"
)

# Get the base64 image
image_base64 = response.data[0].b64_json

# Generate filename from the prompt (safe format)
import re
safe_filename = re.sub(r'[^a-zA-Z0-9]+', '_', user_prompt.strip())[:50]
filename = f"{safe_filename}.png"

# Save the image
with open(filename, "wb") as f:
    f.write(base64.b64decode(image_base64))

print(f"Image saved as {filename}")
