from dotenv import load_dotenv
import openai
import os

load_dotenv()

def setup():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = os.getenv("ORGANIZATION")