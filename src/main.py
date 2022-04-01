from fastapi import FastAPI
from dotenv import load_dotenv
from Summarizer import Summarizer
from setup import setup
from utility import get_prompt

load_dotenv()
setup()

app = FastAPI()

summarizer = Summarizer()

def request_completion(prompt: str):
    text = summarizer.complete(prompt)
    return {"completion": text}

@app.get("/")
def read_root():
    request_completion("testi")