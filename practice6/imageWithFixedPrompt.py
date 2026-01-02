import openai
import sys
import os

# Add parent directory to sys.path to import config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import my_key
from openai import OpenAI
import base64

client = OpenAI(api_key=my_key)

response = client.responses.create(
    model="gpt-4.1-mini",
    input="Generate an image of two boys playing in graden with own toys and also some dogs are there with food and plane flying on sky.",
    tools=[{"type": "image_generation"}],
)

# Save the image to a file
image_data = [
    output.result
    for output in response.output
    if output.type == "image_generation_call"
]

if image_data:
    image_base64 = image_data[0]
    with open("children.png", "wb") as f:
        f.write(base64.b64decode(image_base64))
