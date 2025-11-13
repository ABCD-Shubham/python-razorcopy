import logging
import aiohttp
import ssl
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Set, Dict

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

class SitemapParser:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.discovered_urls: Set[str] = set()
        self.content_types: Dict[str, str] = {} 
        self.robots_sitemaps: Set[str] = set()
        # Create SSL context that doesn't verify certificates
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        self.connector = aiohttp.TCPConnector(ssl=self.ssl_context)

    async def discover_sitemaps_from_robots(self, session: aiohttp.ClientSession) -> Set[str]:
        try:
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
        # Extract a meaningful name from the sitemap URL
        # Example: blog-sitemap.xml -> blog
        filename = url.split('/')[-1]
        name = filename.replace('-sitemap.xml', '').replace('sitemap-', '').replace('.xml', '')
        return name if name else 'main'

    async def _process_sitemap(self, session: aiohttp.ClientSession, url: str, sitemap_type: str):
        try:
            async with session.get(url, headers=BROWSER_HEADERS, ssl=self.ssl_context) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'lxml')  # Use lxml parser
                    for url_tag in soup.find_all('url'):
                        loc = url_tag.find('loc')
                        if loc and loc.text:
                            self.discovered_urls.add(loc.text)
                            self.content_types[loc.text] = sitemap_type
        except Exception as e:
            logger.error(f"Error processing sitemap {url}: {str(e)}")

    async def get_all_urls(self) -> List[str]:
        try:
            async with aiohttp.ClientSession(headers=BROWSER_HEADERS, connector=self.connector) as session:
                # First check robots.txt for sitemaps
                await self.discover_sitemaps_from_robots(session)
                
                # If no sitemaps found in robots.txt, try default sitemap.xml
                if not self.robots_sitemaps:
                    self.robots_sitemaps.add(urljoin(self.base_url, 'sitemap.xml'))

                # Process all found sitemaps
                for sitemap_url in self.robots_sitemaps:
                    try:
                        async with session.get(sitemap_url, headers=BROWSER_HEADERS, ssl=self.ssl_context) as response:
                            if response.status == 200:
                                content = await response.text()
                                soup = BeautifulSoup(content, 'lxml')  # Use lxml parser
                                
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