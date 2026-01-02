import openai
import sys
import os

# Add parent directory to sys.path to import config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import my_key
from openai import OpenAI

# Create a client instance with your API key
client = OpenAI(api_key=my_key)

# System prompt (optional)
messages = [{"role": "system", "content": "You are a helpful assistant."}]

while True:
    # Get dynamic user input
    user_input = input("Ask Anything to ChatGPT: ")
    if user_input:
        messages.append({"role": "user", "content": user_input})
        
    # Send request to OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Extract reply
    reply = response.choices[0].message.content
    print("\nChatGPT:", reply)

    # Save assistant's response for future context
    messages.append({"role": "assistant", "content": reply})
