import sys
import os

# Add parent directory to sys.path to import config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import my_key

from openai import OpenAI

# Create a client instance with your API key
client = OpenAI(api_key=my_key)

def get_completion(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    passage = """My car is not work due to """

    print("Original passage:\n", passage)
    print("\nCompletion:")
    completion_text = get_completion(passage)
    print(completion_text)
