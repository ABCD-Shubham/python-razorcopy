from bs4 import BeautifulSoup
import requests
from fastapi import APIRouter, Depends
import os
from langchain_openai import ChatOpenAI
from app.models.schemas import URLInput
import logging
import asyncio
from urllib.parse import urljoin
from typing import List, Set, Dict
import aiohttp
from fastapi import HTTPException
from app.models.schemas import MetaAnalysisResult, CompanyDetails
import xml.etree.ElementTree as ET
import brotli
import gzip 
import google.generativeai as genai
from app.core.database import get_db
from app.models.project_model import SystemPrompt
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from typing import List
from app.models.project_model import  Article, SystemPrompt, PromptTypes


CUSTOM_GOOGLE_SEARCH = os.getenv('CUSTOM_GOOGLE_SEARCH')
CX_ID = os.getenv("CX_ID")

api_key = CUSTOM_GOOGLE_SEARCH
cx = CX_ID
# client = genai.Client(api_key=CUSTOM_GOOGLE_SEARCH)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
model_name="gemini-1.5-flash"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

CONCURRENT_REQUESTS = 5 
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
router = APIRouter()

def scrape_page_content(url):
    try:
        headers = BROWSER_HEADERS
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return None

def scrape_page_content2(url):
    try:
        headers = BROWSER_HEADERS
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

def target_audience_generator1(title, company_details):
  llm = ChatOpenAI(
        model="gpt-4.1",
      temperature=0.8,
      api_key=OPENAI_API_KEY
  )

  messages = [
      (
          "system",
           """
           You’re an expert SEO strategist with over 10 years of experience in identifying target audiences based on business models and content topics. You have a proven track record of analyzing companies to align their offerings with relevant content that attracts the right readership.

Your task is to provide a singular target audience for a specified article or blog, based on the given company’s business overview and the article title.

Here are the details you need to consider:
- Company's business overview
- Article Title
- target audience: do not use more than 5 or 6 word

Focus on identifying one specific target audience segment that would find the article most relevant and engaging, and ensure your analysis reflects the nuances of the company's offerings and the content theme.
Note: only give target audience, do not include any extra text or words explaining it.
           """
      ),
      (
          "human",
          f"""these are details:\n
          - Article Title:{', '.join(title)}\n
          - company's business overview :\n
            {company_details}
           """""
      )
]


  response = llm.invoke(messages)
  return response.content

def target_audience_generator(title, company_details):
    """Generate a target audience using both OpenAI (gpt-4.1) and Google Gemini."""

    try:
        llm = ChatOpenAI(
            model="gpt-4.1",
            temperature=0.8,
            api_key=OPENAI_API_KEY
        )
        
        openai_messages = [
            ("system", """
            You’re an expert SEO strategist with over 10 years of experience in identifying target audiences based on business models and content topics. 
            You have a proven track record of analyzing companies to align their offerings with relevant content that attracts the right readership.

            Your task is to provide a singular target audience for a specified article or blog, based on the given company’s business overview and the article title.

            **Guidelines:**
            - Use the company's business overview
            - Use the article title
            - Target audience: **Do not use more than 5 or 6 words**.
            
            Focus on identifying **one specific target audience segment** that would find the article most relevant and engaging.
            **Note:** Only return the target audience phrase, no extra text.
            """),
            ("human", f"Article Title: {title}\nCompany Overview:\n{company_details}")
        ]

        openai_response = llm.invoke(openai_messages)
        openai_audience = openai_response.content.strip()
    except Exception as e:
        openai_audience = f"Error with OpenAI: {e}"

    try:
        gemini_messages = [
            """
            You’re an expert SEO strategist with over 10 years of experience in identifying target audiences based on business models and content topics. 
            You have a proven track record of analyzing companies to align their offerings with relevant content that attracts the right readership.

            Your task is to provide a singular target audience for a specified article or blog, based on the given company’s business overview and the article title.

            **Guidelines:**
            - Use the company's business overview
            - Use the article title
            - Target audience: **Do not use more than 5 or 6 words**.
            
            Focus on identifying **one specific target audience segment** that would find the article most relevant and engaging.
            **Note:** Only return the target audience phrase, no extra text.
            """,
            f"Article Title: {title}\nCompany Overview:\n{company_details}"
        ]

        model = genai.GenerativeModel("gemini-2.0-flash")
        gemini_response = model.generate_content(gemini_messages)
        gemini_audience = gemini_response.text.strip()
    except Exception as e:
        gemini_audience = f"Error with Gemini: {e}"

    return {
        "openai_target_audience": openai_audience,
        "gemini_target_audience": gemini_audience
    }


def generate_owner_bio(company_details: CompanyDetails):    # company_details = request_data.company_details

    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.2,
        api_key=OPENAI_API_KEY
    )

    messages = [
        (
            "system",
            """
            You are an expert biographer specializing in crafting concise and compelling professional bios. Your task is to create an owner biography for the founder of a company based on the provided company description.

            **Guidelines**:
                - Extract key details from the company description to inform the bio, including the founder's motivation, expertise, or role.
                - Highlight the owner's background, vision, and alignment with the company's mission.
                - Keep the bio concise (70-100 words), professional, and engaging, suitable for a company website or press release.
                - Use a narrative tone that reflects the company's industry and values.
                - Avoid generic phrases; include specific details that make the bio unique.
                - Provide the bio in paragraph format only, without additional headings or explanations.
            """
        ),
        (
            "human",
            f"Based on the following company description, generate an owner bio in paragraph format only:\n\n{company_details}"
        )
    ]

    response = llm.invoke(messages)
    return {"owner_bio": response.content}


def generate_preview(title, keywords, target_audience, company_detail):
    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.8,
        api_key=OPENAI_API_KEY
    )

    messages = [
        (
            "system",
            """
            You’re a seasoned content strategist with over 15 years of experience in crafting detailed content previews specifically designed for content writers. Your specialty lies in creating structured outlines that include an engaging introduction, well-defined body paragraph points with brief explanations, and a succinct conclusion, all tailored to optimize SEO performance.

            Your task is to create a comprehensive content outline for a content writer. Here are the details you need to consider:
            - Company Business Overview
            - Article Title
            - Keyword
            - Target Audience

            Please ensure that the content outline is relevant to the provided information, well-crafted, and detailed enough for the content writer to seamlessly develop the article without extensive additional research. Focus on maximizing SEO ranking for the site.
            """
        ),
        (
            "human",
            f"""these are details:\n
            - keyword: {keywords}\n
            - Article Title:{title}\n
            - Target Audience : {target_audience}\n
            - company's business overview : {company_detail}
            """
        )
    ]

    response = llm.invoke(messages)
    return response.content

def generates_previews(title,keywords, target_audience, secondary_keywords, company_detail,article, db: Session = Depends(get_db)):
    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.8,
        api_key=OPENAI_API_KEY
    )

    if article.prompt_type_id:
            prompt_type = db.execute(
                    select(PromptTypes)
                    .where(PromptTypes.id == article.prompt_type_id)
            ).scalars().first()

    if prompt_type and prompt_type.outline_prompt_id:
        system_prompt = db.execute(
            select(SystemPrompt)
            .where(SystemPrompt.id == prompt_type.outline_prompt_id,
                   SystemPrompt.type == "topic_outline"
                   )
        ).scalars().first()

    if system_prompt:
        logger.info(f"System prompt found: {system_prompt.description} (added)")
    else:
        logger.warning("No system prompt found. --------")

    full_prompt = system_prompt.description.format(
        title=title,
        keywords=keywords,
        target_audience=target_audience,
        secondary_keywords=secondary_keywords,
        company_detail=company_detail
    )
    
    # Add article rulesets to the prompt if they exist
    if hasattr(article, 'article_rulesets') and article.article_rulesets and article.article_rulesets.strip():
        article_rulesets_section = f"\n\n**Article-Specific Rules and Guidelines:**\n{article.article_rulesets}\n\nPlease follow these specific rules and guidelines when generating the outline for this article."
        full_prompt += article_rulesets_section
        logger.info(f"Article rulesets found and added to outline prompt: {article.article_rulesets[:100]}...")
    else:
        logger.info("No article rulesets found for outline generation or rulesets are empty")
    
    # Add default content rules to prevent extra explanatory notes
    default_content_rules = """
**Strict Content Rules:**
- Write ONLY the article content
- NO explanatory notes, disclaimers, or meta-commentary
- NO language style notes or writing methodology explanations
- NO phrases like "Note on Language Style:" or similar
- Deliver pure article content only"""
    
    full_prompt += default_content_rules

    messages = [("human", full_prompt)]
    response = llm.invoke(messages)
    return response.content

async def extract_main_content(html_content: str) -> str:
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script, style, nav, footer, header elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Limit content length (GPT-3.5 context window is about 4k tokens)
        return text[:4000]
    except Exception as e:
        logger.error(f"Error extracting content: {str(e)}")
        return ""


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
            batch_results = await analyze_batch_urls(batch)  # Process batch
            results.extend(batch_results)

        await asyncio.sleep(0.1)
    
    return results

class SitemapParser:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.discovered_urls: Set[str] = set()
        self.content_types: Dict[str, str] = {}  # Add this to store URL -> content type mapping
        self.robots_sitemaps: Set[str] = set()

    async def discover_sitemaps_from_robots(self) -> Set[str]:
        try:
            async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as session:
                robots_url = urljoin(self.base_url, 'robots.txt')
                async with session.get(robots_url) as response:
                    if response.status == 200:
                        robots_content = await response.text()
                        for line in robots_content.split('\n'):
                            if 'sitemap:' in line.lower():
                                sitemap_url = line.split(':', 1)[1].strip()
                                self.robots_sitemaps.add(sitemap_url)
        except Exception as e:
            logger.error(f"Error fetching robots.txt: {str(e)}")
        return self.robots_sitemaps

    def _get_sitemap_type(self, url: str) -> str:

        filename = url.split('/')[-1]
        name = filename.replace('-sitemap.xml', '').replace('sitemap-', '').replace('.xml', '')
        return name if name else 'main'

    async def _process_sitemap(self, session: aiohttp.ClientSession, url: str, sitemap_type: str):
        try:
            async with session.get(url, headers=BROWSER_HEADERS) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'xml')
                    for url_tag in soup.find_all('url'):
                        loc = url_tag.find('loc')
                        if loc and loc.text:
                            self.discovered_urls.add(loc.text)
                            self.content_types[loc.text] = sitemap_type  # Store the content type
        except Exception as e:
            logger.error(f"Error processing sitemap {url}: {str(e)}")

    async def get_all_urls(self) -> List[str]:
        try:
            async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as session:
                # First check robots.txt for sitemaps
                await self.discover_sitemaps_from_robots()
                
                # If no sitemaps found in robots.txt, try default sitemap.xml
                if not self.robots_sitemaps:
                    self.robots_sitemaps.add(urljoin(self.base_url, 'sitemap.xml'))

                # Process all found sitemaps
                for sitemap_url in self.robots_sitemaps:
                    try:
                        async with session.get(sitemap_url, headers=BROWSER_HEADERS) as response:
                            if response.status == 200:
                                content = await response.text()
                                soup = BeautifulSoup(content, 'xml')
                                
                                # Check for sitemap index
                                sitemaps = soup.find_all('sitemap')
                                if sitemaps:
                                    for sitemap in sitemaps:
                                        loc = sitemap.find('loc')
                                        if loc:
                                            sitemap_type = self._get_sitemap_type(loc.text)
                                            await self._process_sitemap(session, loc.text, sitemap_type)
                                else:
                                    # Single sitemap
                                    sitemap_type = self._get_sitemap_type(sitemap_url)
                                    await self._process_sitemap(session, sitemap_url, sitemap_type)
                    except Exception as e:
                        logger.error(f"Error processing sitemap {sitemap_url}: {str(e)}")
                        continue

            return list(self.discovered_urls)
        except Exception as e:
            logger.error(f"Error in get_all_urls: {str(e)}")
            return list(self.discovered_urls)

async def fetch_and_analyze_sitemap(url_input: URLInput):
    """
    Fetch all URLs from a sitemap and analyze them in batches.
    This replaces `discover-urls` and `analyze-batch` by combining their logic.
    """
    try:
        logger.info(f"Fetching URLs from sitemap: {url_input.url}")
        async with semaphore:
            parser = SitemapParser(str(url_input.url))
            
            try:
                urls = await asyncio.wait_for(
                    parser.get_all_urls(),
                    timeout=300
                )
                content_types = parser.content_types
            except asyncio.TimeoutError:
                logger.warning("URL discovery timed out, returning partial results")
                urls = list(parser.discovered_urls)
                content_types = parser.content_types

        logger.info(f"Discovered {len(urls)} URLs from sitemap.")

        results = await process_urls_in_batches(urls, batch_size=100)

        for result in results:
            result.content_type = content_types.get(result.url, 'Unknown')

        logger.info(f"Batch analysis completed for {len(results)} URLs.")
        return results

    except Exception as e:
        logger.error(f"Error processing sitemap: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {str(e)}"
        )
    
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
        text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return None
    
def company_google_search_links(query, api_key, cx, num=5):
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
        logger.error(f"An error occurred during the search: {e}")

    return links

def extract_content(query, num=6):
    links = company_google_search_links(query, CUSTOM_GOOGLE_SEARCH, CX_ID, num=num)

    if not links:
        return "No links found for the query."

    related_pages = []
    for link in links:
        content = scrape_page_content(link)
        if content:  
            related_pages.append((link, content))

    if not related_pages:
        return "Failed to scrape any of the links."

    result_text = "Extracted Pages:\n"
    for idx, (url, content) in enumerate(related_pages, 1):
        result_text += f"{idx}. {url}\n"
        result_text += f"Extracted Content: {content[:13000]}...\n\n"

    return result_text

def generate_target_audience(company_details: CompanyDetails):    # company_details = request_data.company_details

    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0,
        api_key=OPENAI_API_KEY
    )

    messages = [
        (
            "system",
            """
            You are an expert market analyst specializing in audience segmentation. Your task is to analyze the given company description and identify the **primary target audience**.

            **Guidelines:**
            - Identify the **main audience segments** the company serves.
            - Keep it **brief**, **to the point**, and **concise** (only primary audience).
            - Provide the audience in **heading format only** (e.g., "Small Business Owners", "Health-conscious Consumers", etc.).
            - Do **not** include additional descriptions, only the headings.
            - Separate the audience segments with a comma.

            **Example Output:**
            - Entrepreneurs & Startups
            - Small Business Owners
            - Marketing Agencies
            - Nutrition Professionals
            """
        ),
        (
            "human",
            f"Based on the following company description, extract only the primary target audience as **headings only**:\n\n{company_details}"
        )
    ]

    response = llm.invoke(messages)
    return {"target_audience": response.content}

def get_target_audience(request_data: CompanyDetails):
    company_details = request_data.company_details

    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0,
        api_key=OPENAI_API_KEY
    )

    messages = [
        (
            "system",
            """
            You are an expert market analyst specializing in audience segmentation. Your task is to analyze the given company description and identify the **primary target audience**.

            **Guidelines:**
            - Identify the **main audience segments** the company serves.
            - Keep it **brief**, **to the point**, and **concise** (only primary audience).
            - Provide the audience in **heading format only** (e.g., "Small Business Owners", "Health-conscious Consumers", etc.).
            - Do **not** include additional descriptions, only the headings.
            - Separate the audience segments with a comma.

            **Example Output:**
            - Entrepreneurs & Startups
            - Small Business Owners
            - Marketing Agencies
            - Nutrition Professionals
            """
        ),
        (
            "human",
            f"Based on the following company description, extract only the primary target audience as **headings only**:\n\n{company_details}"
        )
    ]

    response = llm.invoke(messages)
    return {"target_audience": response.content}

def get_sitemap_urls(sitemap_url):
    """Fetch and parse the sitemap XML, handling Brotli, Gzip, and plain text properly."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36",
        "Accept-Encoding": "gzip, deflate, br"
    }

    try:
        response = requests.get(sitemap_url, headers=headers, timeout=20)
        response.raise_for_status()

        content_encoding = response.headers.get("Content-Encoding", "").lower()
        content_type = response.headers.get("Content-Type", "").lower()
        raw_data = response.content

        # Handle Brotli & Gzip decompression only if required
        if "br" in content_encoding:
            try:
                raw_data = brotli.decompress(raw_data)
            except brotli.error as e:
                print(f"Brotli decompression error: {e}. Using raw content.")

        elif "gzip" in content_encoding:
            try:
                raw_data = gzip.decompress(raw_data)
            except gzip.BadGzipFile as e:
                print(f"Gzip decompression error: {e}. Using raw content.")

        # Ensure it's valid XML
        if not raw_data.strip().startswith(b"<?xml"):
            print(f"Invalid XML response from {sitemap_url}. Falling back to empty list.")
            return []

        # Parse the XML Sitemap
        root = ET.fromstring(raw_data)
        urls = [url.text for url in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
        return urls

    except requests.exceptions.RequestException as e:
        print(f"Error fetching sitemap: {e}")
        return []
    except ET.ParseError as e:
        print(f"XML Parsing Error: {e}. Falling back to empty list.")
        return []

def article_google_search_links2(query, api_key, cx, num=5):
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


def count_words(text):
    return len(text.split())


def calculate_average_word_count(word_counts):
    return sum(word_counts) / len(word_counts) if word_counts else 0




async def scrape_page_content3_async(url):
    for attempt in range(2):  
        try:
            browser_config = BrowserConfig(headless=True, user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36)")
            run_config = CrawlerRunConfig(wait_until="networkidle", page_timeout=60000)  
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, config=run_config)

                markdown = result.markdown or ""
                footer = f"\n\n**Source:** {url}\n"
                return f"{markdown.strip()}{footer}"

        except Exception as e:
            print(f"[Attempt {attempt+1}] Failed to scrape {url}: {type(e).__name__}: {e}")
            await asyncio.sleep(1)
    
    return f"**Failed to scrape:** {url}\n"

# def scrape_page_content3(url):
#     loop = asyncio.get_event_loop()
#     return loop.run_until_complete(scrape_page_content3_async(url))

# async def safe_scrape(url):
#     semaphore = Semaphore(3)
#     async with semaphore:
#         return await scrape_page_content3_async(url)

# def extract_citations(content: str):
#     # Look for citations at the end of the content (footer style)
#     citations = []
#     if "Source:" in content:
#         start = content.find("Source:") + len("Source: ")
#         citations.append(content[start:].strip())
    
#     return citations


# async def scrape_page_content3_async(url):
#     for attempt in range(3):  # Retry up to 3 times
#         try:
#             browser_config = BrowserConfig(headless=True, user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36")
#             run_config = CrawlerRunConfig(wait_until="networkidle", page_timeout=120000)  # Increased timeout to 2 minutes
#             async with AsyncWebCrawler(config=browser_config) as crawler:
#                 result = await crawler.arun(url=url, config=run_config)
#                 markdown = result.markdown or ""
#                 footer = f"\n\n**Source:** {url}\n"
#                 return f"{markdown.strip()}{footer}"

#         except TimeoutError as e:
#             print(f"[Attempt {attempt+1}] TimeoutError while scraping {url}: {e}. Retrying...")
#             await asyncio.sleep(random.uniform(1, 5))
#   # Wait longer before retrying

#         except Error as e:
#             print(f"[Attempt {attempt+1}] Failed to scrape {url}: {type(e).__name__}: {e}. Retrying...")
#             await asyncio.sleep(random.uniform(1, 5))
#   # Wait before retrying

#     return f"**Failed to scrape:** {url}"



# def extract_content_google(query, num=5):
#     links = article_google_search_links2(query, api_key, cx, num=num)

#     async def gather_all():
#         semaphore = asyncio.Semaphore(3)

#         async def safe_scrape(url):
#             async with semaphore:
#                 return await scrape_page_content3_async(url)

#         return await asyncio.gather(*(safe_scrape(link) for link in links))

#     results = asyncio.run(gather_all())

#     related_pages = []
#     word_counts = []

#     for link, content in zip(links, results):
#         if content:
#             wc = count_words(content)
#             word_counts.append(wc)
#             related_pages.append((link, content, wc))

#     avg_word_count = calculate_average_word_count(word_counts)

#     result_text = "Reference Articles:\n"
#     for idx, (url, content, wc) in enumerate(related_pages, 1):
#         result_text += f"{idx}. {url}\n"
#         result_text += f"Article Content: {content[:20000]}...\n\n"

#     return result_text, avg_word_count

# async def scrape_page_content_optimized(url: str) -> Optional[str]:
#     """Scrape a webpage with browser-like headers and robust error handling."""

#     headers = {
#         "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.100 Safari/537.36"
#     }

#     browser_config = BrowserConfig(
#         headless=True,
#         user_agent=headers["User-Agent"]
#     )

#     run_config = CrawlerRunConfig(
#         wait_until="domcontentloaded",
#         page_timeout=45000,
#     )

#     for attempt in range(3):
#         try:
#             async with AsyncWebCrawler(config=browser_config) as crawler:
#                 result = await crawler.arun(url=url, config=run_config)
#                 markdown = result.markdown or ""
#                 footer = f"\n\nSource: {url}\n"
#                 return f"{markdown.strip()}{footer}"
#         except Exception as e:
#             print(f"[Attempt {attempt+1}] Failed to scrape {url}: {type(e).__name__}: {str(e)[:100]}...")
#             await asyncio.sleep(2 ** (attempt + 1))  
#     return None


# async def safe_scrape(url: str, semaphore: asyncio.Semaphore) -> Optional[str]:
#     async with semaphore:
#         return await scrape_page_content_optimized(url)

# # --- Gather All ---
# async def gather_all_urls(urls: List[str], max_concurrent: int = 2) -> List[Optional[str]]:
#     semaphore = asyncio.Semaphore(max_concurrent)
#     tasks = [safe_scrape(url, semaphore) for url in urls]
#     return await asyncio.gather(*tasks, return_exceptions=False)

# def extract_citations(content: str):
#     citations = []

#     # Extract any explicit "Source:" links
#     if "Source:" in content:
#         # Find all matches like: Source: https://example.com
#         matches = re.findall(r'Source:\s*(https?://[^\s,")]+)', content)
#         citations.extend(matches)

#     # Also capture all other URLs in the body to use as fallback
#     all_urls = re.findall(r'https?://[^\s,")]+', content)
#     citations.extend(all_urls)

#     # Deduplicate and return
#     return list(set(citations))



# async def extract_content_google(query: str, api_key: str, cx: str, num: int = 5):
#     links = article_google_search_links2(query, api_key, cx, num=num)
#     results = await gather_all_urls(links, max_concurrent=3)

#     related_pages = []
#     word_counts = []
#     failed_links = []
#     citations_text = ""

#     for link, content in zip(links, results):
#         if content:
#             wc = count_words(content)
#             word_counts.append(wc)
#             related_pages.append((link, content, wc))
#             citations_text += content + "\n"
#         else:
#             failed_links.append(link)

#     citations = extract_citations(citations_text)
#     if not citations:
#         citations = [link for link, content in zip(links, results) if content]

#     avg_word_count = calculate_average_word_count(word_counts)

#     result_text = "Reference Articles:\n"
#     for idx, (url, content, wc) in enumerate(related_pages, 1):
#         result_text += f"{idx}. {url}\n"
#         result_text += f"Article Content:\n{content[:20000]}...\n\n"

#     print("Extracted Citations:")
#     for i, url in enumerate(citations, 1):
#         print(f"{i}. {url}")

#     if failed_links:
#         print("\nFailed to scrape:")
#         for link in failed_links:
#             print(f"- {link}")

#     # print(citations,"8236482346823")

#     return result_text, avg_word_count, citations