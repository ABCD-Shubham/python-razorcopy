import os
from dotenv import load_dotenv
from openai import OpenAI
import logging
import asyncio
from fastapi import FastAPI
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
CONCURRENT_REQUESTS = 5  
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

CUSTOM_GOOGLE_SEARCH = os.getenv('CUSTOM_GOOGLE_SEARCH')
CX_ID = os.getenv("CX_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=CUSTOM_GOOGLE_SEARCH)
model_name = "gemini-2.0-flash"
api_key = GEMINI_API_KEY

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key = OPENAI_API_KEY)
model = "ft:gpt-4.1-2024-08-06:e2m::ApEMBO4D"

genai.configure(api_key=CUSTOM_GOOGLE_SEARCH)

app = FastAPI()


origins = os.getenv('PY_PORT')


def format_reference_articles(articles):
    """Formats URLs from the scraped articles list."""
    if not articles:
        return "No reference articles available."

    formatted_articles = []
    for i, article in enumerate(articles, start=1):
        if isinstance(article, (list, tuple)):
            url = article[0] if len(article) > 0 else None
        else:
            url = article  

        if url:
            formatted_articles.append(f"{i}. {url}")

    return "\n".join(formatted_articles) if formatted_articles else "No reference articles available."
