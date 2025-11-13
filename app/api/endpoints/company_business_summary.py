from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
# from services.content_generator import generate_company_summary
import os
import requests
import logging
from bs4 import BeautifulSoup
CX_ID = os.getenv("CX_ID")
cx = CX_ID


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GOOGLE_SEARCH = os.getenv('GOOGLE_SEARCH')
api_key = GOOGLE_SEARCH

router = APIRouter()


def extract_content1(query, num=6):

    # Get Google search links
    links = company_google_search_links1(query, api_key, cx, num=num)

    if not links:
        return "No links found for the query."

    # Extract content from each link
    related_pages = []
    for link in links:
        content = scrape_page_content(link)
        if content:  # Only include links that were successfully scraped
            related_pages.append((link, content))

    if not related_pages:
        return "Failed to scrape any of the links."

    # Format the result as requested
    result_text = "Extracted Pages:\n"
    print(f"Company's URL: {query}\n")
    for idx, (url, content) in enumerate(related_pages, 1):
        result_text += f"{idx}. {url}\n"
        result_text += f"Extracted Content: {content[:13000]}...\n\n"

    return result_text

def company_google_search_links1(query, api_key, cx, num=5):
    links = []
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "q": query,
        "key": "AIzaSyB_Yhbo-GXlFGOWjyNEG0sbQFVUZAyyqa0",
        "cx": cx,
        "num": num
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        for item in data.get("items", []):
            link = item.get("link")
            if link and not link.lower().endswith(".pdf"):  # Exclude PDF links
                links.append(link)

    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred during the search: {e}")

    return links

def extract_content(query, num=6):

    # Get Google search links
    links = company_google_search_links1(query, api_key, cx, num=num)

    if not links:
        return "No links found for the query."

    # Extract content from each link
    related_pages = []
    for link in links:
        content = scrape_page_content(link)
        if content:  # Only include links that were successfully scraped
            related_pages.append((link, content))

    if not related_pages:
        return "Failed to scrape any of the links."

    # Format the result as requested
    result_text = "Extracted Pages:\n"
    print(f"Company's URL: {query}\n")
    for idx, (url, content) in enumerate(related_pages, 1):
        result_text += f"{idx}. {url}\n"
        result_text += f"Extracted Content: {content[:13000]}...\n\n"

    return result_text

def scrape_page_content(url):

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.100 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()  # Raise an error for HTTP issues
        soup = BeautifulSoup(response.text, "html.parser")
        # Extract all visible text from the page
        text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return None
    