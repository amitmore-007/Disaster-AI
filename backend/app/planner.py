from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_disaster_response(scenario: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a disaster response assistant."},
            {"role": "user", "content": f"Generate a disaster response plan for this scenario: {scenario}"}
        ],
        temperature=0.7,
        max_tokens=800
    )

    return response.choices[0].message.content
