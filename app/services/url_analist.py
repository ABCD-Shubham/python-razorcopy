import aiohttp
from bs4 import BeautifulSoup
import asyncio
import logging
from typing import List, Tuple
from app.models.schemas import MetaAnalysisResult
from fastapi import APIRouter, HTTPException
from app.services.scraper import extract_main_content, article_google_search_links2
import aiohttp
import os
import requests


CONCURRENT_REQUESTS = 5 
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
CUSTOM_GOOGLE_SEARCH = os.getenv('CUSTOM_GOOGLE_SEARCH')

api_key = CUSTOM_GOOGLE_SEARCH
CX_ID = os.getenv("CX_ID")

cx = CX_ID


BROWSER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Cache-Control': 'max-age=0'
} 


logger = logging.getLogger(__name__)

async def fetch_and_parse_url(url: str) -> Tuple[str, BeautifulSoup, str]:
    """Fetch URL content and return HTML and parsed soup"""
    async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Failed to fetch URL. Status code: {response.status}"
                )
            html = await response.text()
            soup = BeautifulSoup(html, 'lxml')
            return html, soup, str(response.status)

async def analyze_single_url(url: str, keywords: str = None) -> MetaAnalysisResult:
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout, headers=BROWSER_HEADERS) as session:
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        return MetaAnalysisResult(
                            url=url,
                            status="error",
                            error=f"HTTP {response.status}"
                        )
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'lxml')
                    
                    # Extract existing meta tags
                    meta_title = soup.title.string if soup.title else None
                    meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
                    meta_description = meta_desc_tag.get('content') if meta_desc_tag else None
                    
                    # Generate new meta tags
                    page_content = await extract_main_content(html)
                    #generated_title, generated_description = await generate_meta_tags(
                      #  url, 
                     #   page_content,
                     #   keywords
                  #  )
                    
                    return MetaAnalysisResult(
                        url=url,
                        status="success",
                        meta_title=meta_title,
                        meta_description=meta_description,
                        #generated_title=generated_title,
                        #generated_description=generated_description,
                        keywords=keywords
                    )
            except asyncio.TimeoutError:
                return MetaAnalysisResult(
                    url=url,
                    status="error",
                    error="Request timed out"
                )
    except Exception as e:
        return MetaAnalysisResult(
            url=url,
            status="error",
            error=str(e)
        )

async def analyze_batch_urls(urls: List[str]) -> List[MetaAnalysisResult]:
    tasks = [analyze_single_url(url) for url in urls]
    return await asyncio.gather(*tasks) 

async def process_urls_in_batches(urls, batch_size=50):
    """
    Processes URLs in smaller batches to prevent memory exhaustion.
    """
    results = []
    for i in range(0, len(urls), batch_size):
        batch = urls[i : i + batch_size]
        async with semaphore:
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(urls)//batch_size) + 1}")
            batch_results = await analyze_batch_urls(batch) 
            results.extend(batch_results)

        await asyncio.sleep(0.1)
    
    return results

def scrape_page_content2(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.100 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for HTTP issues
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove footer content if it exists
        for footer in soup.find_all(["footer", "div", "section"], class_=lambda x: x and "footer" in x.lower()):
            footer.decompose()

        # Extract all visible text from the cleaned page
        text = soup.get_text(separator=" ", strip=True)
        # print("textttttttttttt", text)
        return text
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return None


def count_words(text):
    return len(text.split())


def calculate_average_word_count(word_counts):
    return sum(word_counts) / len(word_counts) if word_counts else 0

def extract_content_google1(query, num=5):
    # Get Google search links
    links = article_google_search_links2(query, api_key, cx, num=num)
    # print("Google Search Links-----------:", links)

    # Extract content from each link
    related_pages = []
    word_counts = []

    for link in links:
        content = scrape_page_content2(link)

        if content:  # Only include links that were successfully scraped
            word_count = count_words(content)
            word_counts.append(word_count)
            related_pages.append((link, content, word_count))


    # Calculate average word count first
    avg_word_count = calculate_average_word_count(word_counts)

    result_text = "Reference Articles:\n"
    #print(f"Article's Title: {query}\n")

    for idx, (url, content, word_count) in enumerate(related_pages, 1):
        result_text += f"{idx}. {url}\n"
        #result_text += f"Word Count: {word_count}\n"
        result_text += f"Article Content: {content[:20000]}...\n\n"

    return result_text, avg_word_count