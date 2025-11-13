from typing import List
import requests
import os
from bs4 import BeautifulSoup
import logging
import asyncio
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
CONCURRENT_REQUESTS = 5 
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
load_dotenv()

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

CUSTOM_GOOGLE_SEARCH = os.getenv('CUSTOM_GOOGLE_SEARCH')
CX_ID = os.getenv("CX_ID")

api_key = CUSTOM_GOOGLE_SEARCH
cx = CX_ID

def company_google_search_links(query: str, api_key: str, cx: str, num: int = 5) -> List[str]:
    links = []
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "q": query,
        "key": api_key,
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
        print(f"An error occurred during the search: {e}")

    return links


def article_google_search_links2(query: str, api_key: str, cx: str, num: int = 5) -> List[str]:
    links = []
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "q": query,
        "key": api_key,
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
        print(f"An error occurred during the search: {e}")

    return links

def extract_content(query, num=6):

    # Get Google search links
    links = company_google_search_links(query, api_key, cx, num=num)

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

def scrape_page_content2(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.100 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()  
        soup = BeautifulSoup(response.text, "html.parser")

        for footer in soup.find_all(["footer", "div", "section"], class_=lambda x: x and "footer" in x.lower()):
            footer.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return None

def calculate_average_word_count(word_counts):
    return sum(word_counts) / len(word_counts) if word_counts else 0

def count_words(text):
    return len(text.split())

def extract_content2(query, num=5):
    # Get Google search links separately for OpenAI and Gemini
    openai_links = article_google_search_links2(query, api_key, cx, num=num)
    gemini_links = article_google_search_links2(query, api_key, cx, num=num)

    # Extract content for OpenAI
    openai_related_pages = []
    openai_word_counts = []

    for link in openai_links:
        content = scrape_page_content2(link)
        if content:
            word_count = count_words(content)
            openai_word_counts.append(word_count)
            openai_related_pages.append((link, content, word_count))

    # Extract content for Gemini
    gemini_related_pages = []
    gemini_word_counts = []

    for link in gemini_links:
        content = scrape_page_content2(link)
        if content:
            word_count = count_words(content)
            gemini_word_counts.append(word_count)
            gemini_related_pages.append((link, content, word_count))

    # Calculate average word counts separately
    openai_avg_word_count = calculate_average_word_count(openai_word_counts)
    gemini_avg_word_count = calculate_average_word_count(gemini_word_counts)

    # Format results
    openai_result_text = "Reference Articles (OpenAI):\n"
    for idx, (url, content, word_count) in enumerate(openai_related_pages, 1):
        openai_result_text += f"{idx}. {url}\n"
        openai_result_text += f"Article Content: {content[:20000]}...\n\n"

    gemini_result_text = "Reference Articles (Gemini):\n"
    for idx, (url, content, word_count) in enumerate(gemini_related_pages, 1):
        gemini_result_text += f"{idx}. {url}\n"
        gemini_result_text += f"Article Content: {content[:20000]}...\n\n"

    return (openai_result_text, openai_avg_word_count), (gemini_result_text, gemini_avg_word_count)


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
    