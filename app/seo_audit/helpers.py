import sys
import re
import requests
import time
import json
import xml.etree.ElementTree as ET
from requests.exceptions import RequestException
from requests.exceptions import SSLError
from urllib.parse import urlparse, urljoin, quote

# HTTP headers to use for all requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
    'Cache-Control': 'no-cache',
}

# Global session for all requests
SESSION = requests.Session()

def get_user_id():
    """Return a fixed user_id for demo purposes (no authentication)."""
    return 1

def determine_audit_status(recommendations):
    """
    Determine audit status based on recommendations.
    Handles both old emoji format (✅, ❌, ⚠️) and new text format ([PASS], [FAIL], [WARNING]).
    """
    if not recommendations:
        return "In Progress"
    if any("Error generating recommendations" in str(r) for r in recommendations):
        return "Failed"
    if any("Fatal error" in str(r) for r in recommendations):
        return "Failed"
    if any("Failed to fetch URL" in str(r) for r in recommendations):
        return "Failed"
    if any("[FAIL]" in str(r) for r in recommendations):
        if any("Failed to fetch URL" in str(r) for r in recommendations):
            return "Failed"
        return "Issues Found"
    elif any("[PASS]" in str(r) for r in recommendations):
        return "Passed"
    if any("❌" in str(r) for r in recommendations):
        return "Issues Found"
    elif any("✅" in str(r) for r in recommendations):
        return "Passed"
    if len(recommendations) > 0:
        return "Passed"
    return "In Progress"

def normalize_url(url: str) -> str:
    """
    Normalize URL by adding https:// scheme if no scheme is provided.
    """
    if not url:
        return url
    url = url.strip()
    if url.startswith(('http://', 'https://')):
        return url
    if url.startswith('www.'):
        return f'https://{url}'
    if '.' in url and not url.startswith(('http://', 'https://')):
        return f'https://{url}'
    return url 

###########################################################


# Function to get page content
def get_page_content(url, follow_redirects=True, verify_ssl=True):
    try:
        print(f"[DEBUG] get_page_content: url={url}", file=sys.stderr)
        print(f"[DEBUG] get_page_content: headers={HEADERS}", file=sys.stderr)
        print(f"[DEBUG] get_page_content: follow_redirects={follow_redirects}, verify_ssl={verify_ssl}", file=sys.stderr)
        response = SESSION.get(
            url,
            headers=HEADERS,
            timeout=10,
            allow_redirects=follow_redirects,
            verify=verify_ssl
        )
        print(f"[DEBUG] get_page_content: status_code={response.status_code}, final_url={response.url}", file=sys.stderr)
        return response.text, response.status_code, response.url
    except requests.RequestException as e:
        print(f"[ERROR] get_page_content: exception={e}", file=sys.stderr)
        return None, str(e), None
    

# Function to analyze meta tags
def analyze_meta_tags(soup):
    results = {
        'title': soup.title.string if soup.title else None,
        'title_length': len(soup.title.string) if soup.title and soup.title.string else 0,
        'meta_description': None,
        'meta_description_length': 0,
        'has_viewport': False,
        'has_robots': False,
        'robots_content': None,
        'has_canonical': False,
        'canonical_url': None,
        'has_hreflang': False,
        'hreflang_tags': []
    }
    
    for meta in soup.find_all('meta'):
        if meta.get('name', '').lower() == 'description':
            results['meta_description'] = meta.get('content', '')
            results['meta_description_length'] = len(meta.get('content', ''))
        elif meta.get('name', '').lower() == 'viewport':
            results['has_viewport'] = True
        elif meta.get('name', '').lower() == 'robots':
            results['has_robots'] = True
            results['robots_content'] = meta.get('content', '')
    
    for link in soup.find_all('link'):
        if link.get('rel') and 'canonical' in link.get('rel'):
            results['has_canonical'] = True
            results['canonical_url'] = link.get('href')
        elif link.get('rel') and 'alternate' in link.get('rel') and link.get('hreflang'):
            results['has_hreflang'] = True
            results['hreflang_tags'].append({
                'hreflang': link.get('hreflang'),
                'href': link.get('href')
            })
    
    return results


# Function to analyze headings
def analyze_headings(soup):
    headings = {
        'h1': [],
        'h2': [],
        'h3': [],
        'h4': [],
        'h5': [],
        'h6': []
    }
    
    for level in range(1, 7):
        for heading in soup.find_all(f'h{level}'):
            if heading.text.strip():
                headings[f'h{level}'].append(heading.text.strip())
    
    return headings


# Function to analyze images
def analyze_images(soup, base_url):
    images = []
    for img in soup.find_all('img'):
        src = img.get('src', '')
        if not src:
            # Skip images with missing or empty src
            continue
        # Handle relative URLs
        if not src.startswith(('http://', 'https://')):
            src = urljoin(base_url, src)
        images.append({
            'src': src,
            'alt': img.get('alt', ''),
            'has_alt': bool(img.get('alt'))
        })
    return images


# Function to analyze content
def analyze_content(soup):
    # Get all text content
    text_content = soup.get_text(separator=' ', strip=True)
    words = re.findall(r'\b\w+\b', text_content.lower())
    
    # Count words
    word_count = len(words)
    
    # Calculate keyword density (for simplicity, just finding most common words)
    word_freq = {}
    for word in words:
        if len(word) > 3:  # Only consider words with more than 3 characters
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'word_count': word_count,
        'top_keywords': sorted_words[:20] if sorted_words else []
    }


# Function to analyze links
def analyze_links(soup, base_url):
    internal_links = []
    external_links = []
    base_domain = urlparse(base_url).netloc
    
    for a in soup.find_all('a', href=True):
        href = a.get('href')
        if href.startswith('#') or not href:  # Skip anchor links and empty hrefs
            continue
            
        # Handle relative URLs
        if not href.startswith(('http://', 'https://')):
            full_url = urljoin(base_url, href)
        else:
            full_url = href
            
        link_domain = urlparse(full_url).netloc
        link_text = a.text.strip() or "[No Text]"
        
        if link_domain == base_domain:
            internal_links.append({
                'url': full_url,
                'text': link_text,
                'nofollow': 'nofollow' in a.get('rel', [])
            })
        else:
            external_links.append({
                'url': full_url,
                'text': link_text,
                'nofollow': 'nofollow' in a.get('rel', [])
            })
    
    return {
        'internal_links': internal_links,
        'external_links': external_links
    }

        
# Function to parse robots.txt and check for search engine accessibility
def check_robots_txt(base_url):
    robots_url = urljoin(base_url, '/robots.txt')
    
    # Define search engine user agents to check
    search_engines = [
        'googlebot',       # Google
        'bingbot',         # Bing
        'slurp',           # Yahoo
        'duckduckbot',     # DuckDuckGo
        'baiduspider',     # Baidu
        'yandexbot',       # Yandex
        'facebookexternalhit', # Facebook
        'applebot',        # Apple
        'twitterbot'       # Twitter
    ]
    
    try:
        # Use Googlebot user agent to fetch robots.txt
        googlebot_headers = HEADERS.copy()
        
        response = requests.get(robots_url, headers=googlebot_headers, timeout=5)
        
        if response.status_code == 200:
            content = response.text
            
            # Check for sitemap
            has_sitemap = 'Sitemap:' in content
            
            # Parse robots.txt content
            lines = content.split('\n')
            
            # Variables to track the current section and rules
            current_agent = None
            rules_by_agent = {}
            global_disallow_all = False
            
            # Process each line
            for line in lines:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#') or line.startswith(';'):
                    continue
                
                # Parse User-agent lines
                if line.lower().startswith('user-agent:'):
                    agent = line[11:].strip().lower()
                    current_agent = agent
                    if current_agent not in rules_by_agent:
                        rules_by_agent[current_agent] = {'allow': [], 'disallow': []}
                
                # Parse Disallow directives
                elif line.lower().startswith('disallow:'):
                    if current_agent is not None:
                        path = line[9:].strip()
                        if path:
                            rules_by_agent[current_agent]['disallow'].append(path)
                            # Check for global disallow all for wildcard agents
                            if current_agent == '*' and path == '/':
                                global_disallow_all = True
                
                # Parse Allow directives
                elif line.lower().startswith('allow:'):
                    if current_agent is not None:
                        path = line[6:].strip()
                        if path:
                            rules_by_agent[current_agent]['allow'].append(path)
            
            # Analyze if search engines are blocked
            blocks_search_engines = False
            blocked_engines = []
            allowed_engines = []
            
            # First, check if wildcard (*) blocks all
            if '*' in rules_by_agent and '/' in rules_by_agent['*']['disallow']:
                # Check if there are specific allows that override this
                if not any(rule for rule in rules_by_agent['*']['allow'] if rule == '/' or rule == ''):
                    global_disallow_all = True
            
            # Check each search engine
            for engine in search_engines:
                engine_blocked = False
                
                # Check for direct matches to this engine
                for agent in rules_by_agent:
                    if engine in agent.lower():
                        if '/' in rules_by_agent[agent]['disallow']:
                            # Check if there are any specific allows that override
                            if not any(rule for rule in rules_by_agent[agent]['allow'] if rule == '/' or rule == ''):
                                engine_blocked = True
                                blocked_engines.append(engine)
                                break
                        else:
                            allowed_engines.append(engine)
                            break
                
                # If we don't have specific rules for this engine, apply wildcard rules
                if engine not in blocked_engines and engine not in allowed_engines:
                    if global_disallow_all:
                        engine_blocked = True
                        blocked_engines.append(engine)
            
            # If major search engines like Google, Bing, or Yahoo are blocked, consider it blocking search engines
            major_engines = ['googlebot', 'bingbot', 'slurp']
            blocks_search_engines = any(engine in blocked_engines for engine in major_engines)
            
            return {
                'exists': True,
                'content': content,
                'blocks_search_engines': blocks_search_engines,
                'blocked_engines': blocked_engines,
                'allowed_engines': allowed_engines,
                'has_sitemap': has_sitemap,
                'global_disallow_all': global_disallow_all
            }
            
        return {'exists': False}
    except requests.RequestException:
        return {'exists': False}



# Function to check WWW resolve
def check_www_resolve(url):
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme
    domain = parsed_url.netloc
    
    # Prepare www and non-www URLs
    if domain.startswith('www.'):
        www_url = url
        non_www_url = f"{scheme}://{domain[4:]}{parsed_url.path}"
    else:
        non_www_url = url
        www_url = f"{scheme}://www.{domain}{parsed_url.path}"
    
    # Check both URLs
    try:
        www_response = requests.head(www_url, headers=HEADERS, timeout=5, allow_redirects=True)
        non_www_response = requests.head(non_www_url, headers=HEADERS, timeout=5, allow_redirects=True)
        resolves_to_same = www_response.url == non_www_response.url
        # Determine preferred_url
        preferred_url = None
        if resolves_to_same:
            preferred_url = www_response.url
        else:
            # Prefer the one with status 200, else www
            if www_response.status_code == 200:
                preferred_url = www_response.url
            elif non_www_response.status_code == 200:
                preferred_url = non_www_response.url
            else:
                preferred_url = www_response.url
        # Determine if either redirects
        redirects = (
            (www_response.url != www_url) or (non_www_response.url != non_www_url)
        )
        return {
            'www_url': www_url,
            'www_status': www_response.status_code,
            'www_final_url': www_response.url,
            'non_www_url': non_www_url,
            'non_www_status': non_www_response.status_code,
            'non_www_final_url': non_www_response.url,
            'resolves_to_same': resolves_to_same,
            'preferred_url': preferred_url,
            'redirects': redirects
        }
    except requests.RequestException as e:
        return {
            'www_url': www_url,
            'non_www_url': non_www_url,
            'error': str(e)
        }


# Function to check redirect chains
def check_redirect_chain(url):
    try:
        # Initialize session to track history
        session = requests.Session()
        response = session.get(url, headers=HEADERS, timeout=10)
        
        # Get history from the response
        redirect_history = [{
            'url': h.url,
            'status_code': h.status_code
        } for h in response.history]
        
        # Add the final destination
        redirect_history.append({
            'url': response.url,
            'status_code': response.status_code
        })
        
        return {
            'chain_length': len(redirect_history) - 1,  # Subtract 1 to get only redirects
            'chain': redirect_history,
            'has_chain': len(redirect_history) > 1
        }
    except requests.RequestException as e:
        return {
            'error': str(e),
            'has_chain': False
        }
    

# Function to check for Google Analytics and Tag Manager
def check_analytics(html_content):
    results = {
        'has_google_analytics': False,
        'has_ga4': False,
        'has_universal_analytics': False,
        'has_google_tag_manager': False,
        'gtm_container_id': None,
        'ga_tracking_ids': []
    }
    
    # Check for Google Analytics
    # Universal Analytics (UA-)
    ua_pattern = re.compile(r'UA-[0-9]+-[0-9]+')
    ua_matches = ua_pattern.findall(html_content)
    if ua_matches:
        results['has_google_analytics'] = True
        results['has_universal_analytics'] = True
        results['ga_tracking_ids'].extend(ua_matches)
    
    # GA4 (G-)
    ga4_pattern = re.compile(r'G-[A-Z0-9]+')
    ga4_matches = ga4_pattern.findall(html_content)
    if ga4_matches:
        results['has_google_analytics'] = True
        results['has_ga4'] = True
        results['ga_tracking_ids'].extend(ga4_matches)
    
    # Check for Google Tag Manager
    gtm_pattern = re.compile(r'GTM-[A-Z0-9]+')
    gtm_matches = gtm_pattern.findall(html_content)
    if gtm_matches:
        results['has_google_tag_manager'] = True
        results['gtm_container_id'] = gtm_matches[0] if gtm_matches else None
    
    return results


# Function to check for custom 404 page
def check_custom_404(base_url):
    # Generate a random URL that is unlikely to exist
    random_path = f"/this-page-does-not-exist-{int(time.time())}"
    not_found_url = urljoin(base_url, random_path)
    
    try:
        response = requests.get(not_found_url, headers=HEADERS, timeout=5)
        return {
            'status_code': response.status_code,
            'has_custom_404': response.status_code == 404 and len(response.text) > 500,
            'is_soft_404': response.status_code == 200 and "not found" in response.text.lower()
        }
    except requests.RequestException as e:
        return {
            'error': str(e),
            'has_custom_404': False
        }
    

# Function to check HTTPS and SSL
def check_https(url):
    parsed_url = urlparse(url)
    is_https = parsed_url.scheme == 'https'
    
    # If site is already HTTPS, check certificate
    ssl_info = {}
    if is_https:
        try:
            response = requests.get(url, headers=HEADERS, timeout=5, verify=True)
            ssl_info['valid_certificate'] = True
        except SSLError as e:
            ssl_info['valid_certificate'] = False
            ssl_info['error'] = str(e)
    
    # Check if HTTP redirects to HTTPS
    http_url = None
    redirect_to_https = False
    if is_https:
        http_url = f"http://{parsed_url.netloc}{parsed_url.path}"
        try:
            response = requests.get(http_url, headers=HEADERS, timeout=5, allow_redirects=True)
            redirect_to_https = response.url.startswith('https://')
        except requests.RequestException:
            pass
    
    return {
        'is_https': is_https,
        'http_url': http_url,
        'redirects_to_https': redirect_to_https,
        'ssl_info': ssl_info
    }


# Helper function to process sitemap response
def process_sitemap_response(response):
    try:
        root = ET.fromstring(response.content)
        namespaces = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        print(f"[DEBUG] process_sitemap_response: root tag={root.tag}", file=sys.stderr)
        # Extract URL entries
        url_elems = root.findall('.//sm:url', namespaces) or root.findall('.//url')
        print(f"[DEBUG] process_sitemap_response: found {len(url_elems)} <url> elements", file=sys.stderr)
        url_list = []
        for ue in url_elems:
            loc = ue.find('sm:loc', namespaces)
            if loc is None:
                loc = ue.find('loc')
            if loc is not None and loc.text:
                url_list.append(loc.text.strip())
        # Extract sitemap index entries
        sitemap_elems = root.findall('.//sm:sitemap', namespaces) or root.findall('.//sitemap')
        print(f"[DEBUG] process_sitemap_response: found {len(sitemap_elems)} <sitemap> elements", file=sys.stderr)
        sitemap_list = []
        for se in sitemap_elems:
            loc = se.find('sm:loc', namespaces)
            if loc is None:
                loc = se.find('loc')
            if loc is not None and loc.text:
                sitemap_list.append(loc.text.strip())
        is_index = len(sitemap_list) > 0
        print(f"[DEBUG] process_sitemap_response: is_index={is_index}, sitemap_list_sample={sitemap_list[:3]}", file=sys.stderr)
        return {
            'exists': True,
            'is_valid_xml': True,
            'url_count': len(url_list),
            'sitemap_count': len(sitemap_list),
            'is_index': is_index,
            'url_list': url_list,
            'sitemap_list': sitemap_list
        }
    except ET.ParseError:
        return {
            'exists': True,
            'is_valid_xml': False
        }
    except Exception:  # Catch any other issues parsing the content
        return {'exists': False, 'error': 'Content is not valid XML'}


# Function to check for sitemap.xml and other common sitemap patterns
def check_sitemap(base_url):
    import sys
    print(f"[DEBUG] check_sitemap: base_url={base_url}", file=sys.stderr)
    # List of common sitemap patterns to check
    sitemap_patterns = [
        '/sitemap.xml',           # Standard sitemap
        '/sitemap_index.xml',     # Common alternative
        '/sitemaps.xml',          # Plural version sometimes used
        '/sitemap-index.xml',     # Hyphenated version
        '/sitemap/',              # Directory with sitemaps
        '/sitemaps/',             # Plural directory
        '/sitemap1.xml',          # Numbered sitemaps
    ]
    fetch_log = []
    # Also check for sitemap URL in robots.txt
    try:
        robots_url = urljoin(base_url, '/robots.txt')
        print(f"[DEBUG] Fetching robots.txt: {robots_url}", file=sys.stderr)
        robots_headers = HEADERS.copy()
        robots_headers['Cache-Control'] = 'no-cache'
        robots_response = requests.get(robots_url, headers=robots_headers, timeout=10)
        fetch_log.append({'url': robots_url, 'status': robots_response.status_code})
        print(f"[DEBUG] robots.txt status: {robots_response.status_code}", file=sys.stderr)
        if robots_response.status_code == 200 or robots_response.status_code == 202:
            if robots_response.status_code == 202:
                print(f"[WARNING] robots.txt returned 202, attempting to parse anyway", file=sys.stderr)
            import re
            sitemap_urls_in_robots = []
            for line in robots_response.text.split('\n'):
                match = re.match(r'^\s*sitemap\s*:\s*(.+)$', line, re.IGNORECASE)
                if match:
                    sitemap_url = match.group(1).strip()
                    print(f"[DEBUG] Found sitemap in robots.txt: {sitemap_url}", file=sys.stderr)
                    sitemap_urls_in_robots.append(sitemap_url)
            # Try sitemaps specified in robots.txt first
            for sitemap_url in sitemap_urls_in_robots:
                try:
                    print(f"[DEBUG] Fetching sitemap from robots.txt: {sitemap_url}", file=sys.stderr)
                    sitemap_headers = HEADERS.copy()
                    sitemap_headers['Cache-Control'] = 'no-cache'
                    response = requests.get(sitemap_url, headers=sitemap_headers, timeout=15)
                    fetch_log.append({'url': sitemap_url, 'status': response.status_code})
                    print(f"[DEBUG] Sitemap status: {response.status_code}", file=sys.stderr)
                    if response.status_code == 200 or response.status_code == 202:
                        if response.status_code == 202:
                            print(f"[WARNING] Sitemap pattern {sitemap_url} returned 202, attempting to parse anyway", file=sys.stderr)
                        try:
                            result = process_sitemap_response(response)
                        except Exception as e:
                            fetch_log.append({'url': sitemap_url, 'error': f'XML parse error: {e}'})
                            return {'exists': False, 'error': f'XML parse error: {e}', 'fetch_log': fetch_log}
                        # If it's a sitemap index, fetch child sitemaps to gather URLs
                        if result.get('is_index'):
                            print(f"[DEBUG] Entering sitemap index processing for {sitemap_url}", file=sys.stderr)
                            raw_sitemap_list = result.get('sitemap_list', [])
                            sitemap_list = raw_sitemap_list if isinstance(raw_sitemap_list, list) else []
                            print(f"[DEBUG] Found {len(sitemap_list)} child sitemaps in index. Sample: {sitemap_list[:3]}", file=sys.stderr)
                            urls = []
                            for child_sitemap_url in sitemap_list:
                                try:
                                    print(f"[DEBUG] Fetching child sitemap: {child_sitemap_url}", file=sys.stderr)
                                    child_headers = HEADERS.copy()
                                    child_headers['Cache-Control'] = 'no-cache'
                                    child_response = requests.get(child_sitemap_url, headers=child_headers, timeout=15)
                                    fetch_log.append({'url': child_sitemap_url, 'status': child_response.status_code})
                                    print(f"[DEBUG] Child sitemap status: {child_response.status_code}", file=sys.stderr)
                                    if child_response.status_code == 200 or child_response.status_code == 202:
                                        if child_response.status_code == 202:
                                            print(f"[WARNING] Sitemap pattern {child_sitemap_url} returned 202, attempting to parse anyway", file=sys.stderr)
                                        raw_child_urls = process_sitemap_response(child_response).get('url_list', [])
                                        child_urls = raw_child_urls if isinstance(raw_child_urls, list) else []
                                        print(f"[DEBUG] Found {len(child_urls)} URLs in child sitemap {child_sitemap_url}. Sample: {child_urls[:3]}", file=sys.stderr)
                                        urls.extend(child_urls)
                                    else:
                                        print(f"[WARNING] Failed to fetch child sitemap {child_sitemap_url}: status {child_response.status_code}", file=sys.stderr)
                                except Exception as e:
                                    print(f"[ERROR] Exception fetching child sitemap {child_sitemap_url}: {e}", file=sys.stderr)
                            result['url_list'] = urls if isinstance(urls, list) else []
                            print(f"[DEBUG] After processing sitemap index: found {len(urls)} URLs. Sample: {urls[:5]}", file=sys.stderr)
                            if not urls:
                                print(f"[WARNING] No URLs found in sitemap index {sitemap_url}", file=sys.stderr)
                        result['found_at'] = sitemap_url
                        result['found_via'] = 'robots.txt'
                        result['fetch_log'] = fetch_log
                        return result
                except requests.RequestException as e:
                    fetch_log.append({'url': sitemap_url, 'error': str(e)})
                    print(f"[DEBUG] Error fetching sitemap from robots.txt: {e}", file=sys.stderr)
                    continue
    except requests.RequestException as e:
        fetch_log.append({'url': robots_url, 'error': str(e)})
        print(f"[DEBUG] Error fetching robots.txt: {e}", file=sys.stderr)
        pass  # Continue with pattern matching if robots.txt check fails
    # Check each pattern
    for pattern in sitemap_patterns:
        sitemap_url = urljoin(base_url, pattern)
        try:
            print(f"[DEBUG] Fetching sitemap pattern: {sitemap_url}", file=sys.stderr)
            sitemap_headers = HEADERS.copy()
            sitemap_headers['Cache-Control'] = 'no-cache'
            response = requests.get(sitemap_url, headers=sitemap_headers, timeout=15)
            fetch_log.append({'url': sitemap_url, 'status': response.status_code})
            print(f"[DEBUG] Sitemap pattern status: {response.status_code}", file=sys.stderr)
            if response.status_code == 200 or response.status_code == 202:
                if response.status_code == 202:
                    print(f"[WARNING] Sitemap pattern {sitemap_url} returned 202, attempting to parse anyway", file=sys.stderr)
                try:
                    result = process_sitemap_response(response)
                except Exception as e:
                    fetch_log.append({'url': sitemap_url, 'error': f'XML parse error: {e}'})
                    return {'exists': False, 'error': f'XML parse error: {e}', 'fetch_log': fetch_log}
                # If it's a sitemap index, fetch child sitemaps to gather URLs
                if result.get('is_index'):
                    urls = []
                    for child_sitemap in result.get('sitemap_list', []):
                        try:
                            print(f"[DEBUG] Fetching child sitemap: {child_sitemap}", file=sys.stderr)
                            child_headers = HEADERS.copy()
                            child_headers['Cache-Control'] = 'no-cache'
                            r2 = requests.get(child_sitemap, headers=child_headers, timeout=15)
                            fetch_log.append({'url': child_sitemap, 'status': r2.status_code})
                            print(f"[DEBUG] Child sitemap status: {r2.status_code}", file=sys.stderr)
                            if r2.status_code == 200 or r2.status_code == 202:
                                if r2.status_code == 202:
                                    print(f"[WARNING] Sitemap pattern {child_sitemap} returned 202, attempting to parse anyway", file=sys.stderr)
                                try:
                                    sub = process_sitemap_response(r2)
                                    urls.extend(sub.get('url_list', []))
                                except Exception as e:
                                    fetch_log.append({'url': child_sitemap, 'error': f'XML parse error: {e}'})
                                    continue
                        except requests.RequestException as e:
                            fetch_log.append({'url': child_sitemap, 'error': str(e)})
                            print(f"[DEBUG] Error fetching child sitemap: {e}", file=sys.stderr)
                            continue
                    result['url_list'] = urls
                result['found_at'] = sitemap_url
                result['found_via'] = 'direct_check'
                result['fetch_log'] = fetch_log
                return result
        except requests.RequestException as e:
            fetch_log.append({'url': sitemap_url, 'error': str(e)})
            print(f"[DEBUG] Error fetching sitemap pattern: {e}", file=sys.stderr)
            continue
    # If we get here, no sitemap was found
    print(f"[DEBUG] No sitemap found for {base_url}", file=sys.stderr)
    return {'exists': False, 'checked_patterns': sitemap_patterns, 'fetch_log': fetch_log}



# Function to detect schema markup
def check_schema_markup(html_content, soup):
    results = {
        'has_schema': False,
        'schema_types': [],
        'json_ld_count': 0,
        'microdata_count': 0,
        'rdfa_count': 0
    }
    
    # Check for JSON-LD
    json_ld_scripts = soup.find_all('script', {'type': 'application/ld+json'})
    if json_ld_scripts:
        results['has_schema'] = True
        results['json_ld_count'] = len(json_ld_scripts)
        
        # Extract schema types
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string.strip() if script.string else "{}")
                # Handle @type at the root level
                if '@type' in data:
                    results['schema_types'].append(data['@type'])
                
                # Check for @graph array
                if '@graph' in data and isinstance(data['@graph'], list):
                    for item in data['@graph']:
                        if isinstance(item, dict) and '@type' in item:
                            results['schema_types'].append(item['@type'])
                
                # Check for nested types in other properties
                for key, value in data.items():
                    if isinstance(value, dict) and '@type' in value:
                        results['schema_types'].append(value['@type'])
            except json.JSONDecodeError:
                pass
    
    # Check for Microdata
    microdata_elements = soup.find_all(itemscope=True)
    if microdata_elements:
        results['has_schema'] = True
        results['microdata_count'] = len(microdata_elements)
        
        # Extract schema types from microdata
        for element in microdata_elements:
            if element.get('itemtype'):
                schema_type = element.get('itemtype').split('/')[-1]
                results['schema_types'].append(schema_type)
    
    # Check for RDFa
    rdfa_elements = soup.find_all(attrs={"typeof": True})
    if rdfa_elements:
        results['has_schema'] = True
        results['rdfa_count'] = len(rdfa_elements)
        
        # Extract schema types from RDFa
        for element in rdfa_elements:
            typeof_value = element.get('typeof')
            if typeof_value:
                # RDFa typeof can contain multiple space-separated types
                if ' ' in typeof_value:
                    for type_value in typeof_value.split():
                        if type_value.strip():
                            results['schema_types'].append(type_value.strip())
                else:
                    results['schema_types'].append(typeof_value)
    
    # Ensure all schema types are strings and remove duplicates
    processed_schema_types = []
    for schema_type in results['schema_types']:
        # Check if it's a list
        if isinstance(schema_type, list):
            for sub_type in schema_type:
                if isinstance(sub_type, str):
                    processed_schema_types.append(sub_type)
        # Check if it's a string
        elif isinstance(schema_type, str):
            processed_schema_types.append(schema_type)
        # Try to convert to string if possible
        else:
            try:
                processed_schema_types.append(str(schema_type))
            except:
                pass
                
    # Remove duplicates while preserving order
    seen = set()
    results['schema_types'] = [x for x in processed_schema_types if not (x in seen or seen.add(x))]
    
    return results



# Function to check if site is accessible to search engines
def check_search_engine_accessibility(url, robots_result, meta_tags):
    # Check if robots.txt blocks search engines
    blocked_by_robots_txt = robots_result.get('blocks_search_engines', False) if robots_result.get('exists', False) else False
    blocked_engines = robots_result.get('blocked_engines', []) if robots_result.get('exists', False) else []
    allowed_engines = robots_result.get('allowed_engines', []) if robots_result.get('exists', False) else []
    
    # Check meta robots tag
    meta_robots = meta_tags.get('robots_content', '')
    blocked_by_meta_robots = False
    if meta_robots and ('noindex' in meta_robots.lower() or 'none' in meta_robots.lower()):
        blocked_by_meta_robots = True
    
    # Check X-Robots-Tag header using Googlebot user agent
    try:
        googlebot_headers = HEADERS.copy()
        googlebot_headers['User-Agent'] = 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'
        
        response = requests.head(url, headers=googlebot_headers, timeout=5)
        x_robots_tag = response.headers.get('X-Robots-Tag', '')
        blocked_by_header = 'noindex' in x_robots_tag.lower() or 'none' in x_robots_tag.lower()
    except requests.RequestException:
        blocked_by_header = False
    
    return {
        'is_accessible': not (blocked_by_robots_txt or blocked_by_meta_robots or blocked_by_header),
        'blocked_by_robots_txt': blocked_by_robots_txt,
        'blocked_engines': blocked_engines,
        'allowed_engines': allowed_engines,
        'blocked_by_meta_robots': blocked_by_meta_robots,
        'blocked_by_x_robots_header': blocked_by_header
    }


# Function to check Google PageSpeed Insights
def check_pagespeed(url, api_key=None, run_pagespeed=True, timeout=600):  # Increased timeout to 60 seconds
    if not run_pagespeed:
        return {'success': False, 'error': 'PageSpeed analysis disabled'}
    
    try:
        # Encode URL for API request
        encoded_url = quote(url, safe='')
        
        # Build API URL with key if provided
        api_base = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
        params = f"url={encoded_url}&strategy=mobile&category=performance&category=accessibility&category=best-practices&category=seo"
        
        if api_key:
            api_url = f"{api_base}?{params}&key={api_key}"
        else:
            api_url = f"{api_base}?{params}"
        
        # Try to fetch mobile results with increased timeout
        response = requests.get(api_url, timeout=timeout)
        
        # Handle rate limiting specifically
        if response.status_code == 429:
            return {
                'success': False, 
                'error': "API Rate Limit Exceeded", 
                'resolution': "Consider adding a Google PageSpeed API key in the settings or try again later."
            }
        
        # Process successful response
        if response.status_code == 200:
            data = response.json()
            
            # Extract mobile results
            mobile_results = {
                'success': True,
                'performance_score': data.get('lighthouseResult', {}).get('categories', {}).get('performance', {}).get('score', 0) * 100,
                'accessibility_score': data.get('lighthouseResult', {}).get('categories', {}).get('accessibility', {}).get('score', 0) * 100,
                'best_practices_score': data.get('lighthouseResult', {}).get('categories', {}).get('best-practices', {}).get('score', 0) * 100,
                'seo_score': data.get('lighthouseResult', {}).get('categories', {}).get('seo', {}).get('score', 0) * 100,
                'first_contentful_paint': data.get('lighthouseResult', {}).get('audits', {}).get('first-contentful-paint', {}).get('displayValue', 'N/A'),
                'speed_index': data.get('lighthouseResult', {}).get('audits', {}).get('speed-index', {}).get('displayValue', 'N/A'),
                'largest_contentful_paint': data.get('lighthouseResult', {}).get('audits', {}).get('largest-contentful-paint', {}).get('displayValue', 'N/A'),
                'time_to_interactive': data.get('lighthouseResult', {}).get('audits', {}).get('interactive', {}).get('displayValue', 'N/A'),
                'total_blocking_time': data.get('lighthouseResult', {}).get('audits', {}).get('total-blocking-time', {}).get('displayValue', 'N/A'),
                'cumulative_layout_shift': data.get('lighthouseResult', {}).get('audits', {}).get('cumulative-layout-shift', {}).get('displayValue', 'N/A'),
            }
            
            # Get desktop results (only if mobile was successful)
            desktop_results = {'success': False}
            try:
                # Build desktop API URL
                desktop_params = f"url={encoded_url}&strategy=desktop&category=performance&category=accessibility&category=best-practices&category=seo"
                if api_key:
                    desktop_api_url = f"{api_base}?{desktop_params}&key={api_key}"
                else:
                    desktop_api_url = f"{api_base}?{desktop_params}"
                    
                desktop_response = requests.get(desktop_api_url, timeout=timeout)  # Use the same timeout value
                if desktop_response.status_code == 200:
                    desktop_data = desktop_response.json()
                    desktop_results = {
                        'success': True,
                        'performance_score': desktop_data.get('lighthouseResult', {}).get('categories', {}).get('performance', {}).get('score', 0) * 100,
                        'accessibility_score': desktop_data.get('lighthouseResult', {}).get('categories', {}).get('accessibility', {}).get('score', 0) * 100,
                        'best_practices_score': desktop_data.get('lighthouseResult', {}).get('categories', {}).get('best-practices', {}).get('score', 0) * 100,
                        'seo_score': desktop_data.get('lighthouseResult', {}).get('categories', {}).get('seo', {}).get('score', 0) * 100,
                        'first_contentful_paint': desktop_data.get('lighthouseResult', {}).get('audits', {}).get('first-contentful-paint', {}).get('displayValue', 'N/A'),
                        'speed_index': desktop_data.get('lighthouseResult', {}).get('audits', {}).get('speed-index', {}).get('displayValue', 'N/A'),
                        'largest_contentful_paint': desktop_data.get('lighthouseResult', {}).get('audits', {}).get('largest-contentful-paint', {}).get('displayValue', 'N/A'),
                        'time_to_interactive': desktop_data.get('lighthouseResult', {}).get('audits', {}).get('interactive', {}).get('displayValue', 'N/A'),
                        'total_blocking_time': desktop_data.get('lighthouseResult', {}).get('audits', {}).get('total-blocking-time', {}).get('displayValue', 'N/A'),
                        'cumulative_layout_shift': desktop_data.get('lighthouseResult', {}).get('audits', {}).get('cumulative-layout-shift', {}).get('displayValue', 'N/A'),
                    }
                else:
                    # If desktop fails but mobile worked, just note the desktop error
                    desktop_results = {'success': False, 'error': f"API Error: {desktop_response.status_code}"}
            except Exception as e:
                # If desktop analysis fails, we can still return mobile results
                desktop_results = {'success': False, 'error': str(e)}
                
            # Get opportunities for improvement
            opportunities = []
            for audit_id, audit in data.get('lighthouseResult', {}).get('audits', {}).items():
                if audit.get('details', {}).get('type') == 'opportunity':
                    opportunities.append({
                        'title': audit.get('title', ''),
                        'description': audit.get('description', ''),
                        'score': audit.get('score', 0),
                        'display_value': audit.get('displayValue', 'N/A')
                    })
            
            return {
                'mobile': mobile_results,
                'desktop': desktop_results,
                'opportunities': opportunities,
                'success': True
            }
        else:
            return {'success': False, 'error': f"API Error: {response.status_code}"}
    except requests.exceptions.Timeout:
        return {
            'success': False, 
            'error': 'Connection Timeout', 
            'resolution': 'The PageSpeed API is taking too long to respond. Try again later or analyze a simpler page.'
        }
    except requests.exceptions.ConnectionError:
        return {
            'success': False, 
            'error': 'Connection Error', 
            'resolution': 'Unable to connect to the PageSpeed API. Check your internet connection or try again later.'
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_seo_recommendations(analysis):
    recommendations = []
    # Defensive extraction with defaults
    meta = analysis.get('meta_tags', {})
    headings = analysis.get('headings', {})
    images = analysis.get('images', [])
    content = analysis.get('content', {})
    links = analysis.get('links', {})
    robots = analysis.get('robots', {})
    www_resolve = analysis.get('www_resolve', {})
    redirects = analysis.get('redirects', {})
    analytics = analysis.get('analytics', {})
    custom_404 = analysis.get('custom_404', {})
    https = analysis.get('https', {})
    sitemap = analysis.get('sitemap', {})
    schema = analysis.get('schema', {})
    accessibility = analysis.get('accessibility', {})
    pagespeed = analysis.get('pagespeed', {'success': False})
    
    # Title recommendations
    title = meta.get('title')
    title_length = meta.get('title_length', 0)
    if not title:
        recommendations.append("[FAIL] Page is missing a title tag.")
    elif title_length < 30:
        recommendations.append(f"[WARNING] Title tag is too short ({title_length} chars). Aim for 50-60 characters.")
    elif title_length > 60:
        recommendations.append(f"[WARNING] Title tag is too long ({title_length} chars). Keep it under 60 characters.")
    else:
        recommendations.append("[PASS] Title tag length is optimal.")
    
    # Meta description recommendations
    meta_description = meta.get('meta_description')
    meta_description_length = meta.get('meta_description_length', 0)
    if not meta_description:
        recommendations.append("[FAIL] Page is missing a meta description.")
    elif meta_description_length < 120:
        recommendations.append(f"[WARNING] Meta description is too short ({meta_description_length} chars). Aim for 120-160 characters.")
    elif meta_description_length > 160:
        recommendations.append(f"[WARNING] Meta description is too long ({meta_description_length} chars). Keep it under 160 characters.")
    else:
        recommendations.append("[PASS] Meta description length is optimal.")
    
    # Heading recommendations
    h1 = headings.get('h1', [])
    h2 = headings.get('h2', [])
    if not h1:
        recommendations.append("[FAIL] Page is missing an H1 heading.")
    elif len(h1) > 1:
        recommendations.append(f"[WARNING] Page has multiple H1 headings ({len(h1)}). Consider using only one.")
    else:
        recommendations.append("[PASS] Page has one H1 heading.")
    
    if not h2:
        recommendations.append("[WARNING] Page has no H2 headings. Consider adding some for better structure.")
    
    # Image recommendations
    images_without_alt = sum(1 for img in images if isinstance(img, dict) and not img.get('has_alt')) if isinstance(images, list) else 0
    if isinstance(images, list) and images_without_alt > 0:
        recommendations.append(f"[WARNING] {images_without_alt} out of {len(images)} images are missing alt text.")
    elif isinstance(images, list) and images:
        recommendations.append("[PASS] All images have alt text.")
    
    # Content recommendations
    word_count = content.get('word_count', 0)
    if word_count < 300:
        recommendations.append(f"[WARNING] Content is thin ({word_count} words). Aim for at least 300 words.")
    else:
        recommendations.append(f"[PASS] Content length is good ({word_count} words).")
    
    # Link recommendations
    external_links = links.get('external_links', [])
    if isinstance(external_links, list) and len(external_links) > 0 and all(isinstance(link, dict) and not link.get('nofollow') for link in external_links):
        recommendations.append("[WARNING] Consider adding 'nofollow' to external links where appropriate.")
    
    # Mobile optimization
    if not meta.get('has_viewport', False):
        recommendations.append("[FAIL] Page is missing a viewport meta tag for mobile optimization.")
    
    # Canonical URL
    if not meta.get('has_canonical', False):
        recommendations.append("[WARNING] Page is missing a canonical URL tag.")
    
    # Search engine accessibility
    if not accessibility.get('is_accessible', True):
        recommendations.append("[FAIL] The website is not accessible to search engines.")
        if accessibility.get('blocked_by_robots_txt'):
            recommendations.append("[FAIL] The website is blocked by robots.txt.")
        if accessibility.get('blocked_by_meta_robots'):
            recommendations.append("[FAIL] The page is blocked by meta robots tag.")
        if accessibility.get('blocked_by_x_robots_header'):
            recommendations.append("[FAIL] The page is blocked by X-Robots-Tag header.")
    else:
        recommendations.append("[PASS] The website is accessible to search engines.")
    
    # WWW resolve
    if 'error' not in www_resolve and not www_resolve.get('resolves_to_same', True):
        recommendations.append("[WARNING] WWW and non-WWW versions do not resolve to the same URL. This can cause duplicate content issues.")
    
    # Redirect chain
    if 'error' not in redirects and redirects.get('chain_length', 0) > 1:
        recommendations.append(f"[WARNING] The page has a redirect chain with {redirects.get('chain_length', 0)} redirects. Keep redirects to a minimum.")
    
    # Google Analytics
    if not analytics.get('has_google_analytics', False) and not analytics.get('has_google_tag_manager', False):
        recommendations.append("[WARNING] No Google Analytics or Tag Manager detected. Consider adding analytics to track website performance.")
    
    # Custom 404 page
    if 'error' not in custom_404 and not custom_404.get('has_custom_404', False):
        recommendations.append("[WARNING] No custom 404 error page detected. Consider creating a user-friendly 404 page.")
    if 'error' not in custom_404 and custom_404.get('is_soft_404', False):
        recommendations.append("[FAIL] The website is using soft 404s (returning 200 status code for non-existent pages). This can confuse search engines.")
    
    # HTTPS
    if not https.get('is_https', False):
        recommendations.append("[FAIL] The website is not using HTTPS. Consider upgrading to HTTPS for better security and SEO.")
    elif 'ssl_info' in https and not https['ssl_info'].get('valid_certificate', True):
        recommendations.append("[FAIL] The website has an invalid SSL certificate. Fix SSL issues for better security and user trust.")
    else:
        recommendations.append("[PASS] The website is using HTTPS with a valid SSL certificate.")
    
    if https.get('is_https', False) and not https.get('redirects_to_https', True):
        recommendations.append("[WARNING] HTTP version of the site does not redirect to HTTPS. Set up proper redirects.")
    
    # Sitemap
    if not sitemap.get('exists', False):
        recommendations.append("[WARNING] No sitemap found. Consider creating a sitemap to help search engines index your content.")
    elif sitemap.get('exists', False) and not sitemap.get('is_valid_xml', True):
        if 'found_at' in sitemap:
            recommendations.append(f"[FAIL] The sitemap at {sitemap['found_at']} exists but is not valid XML. Fix the sitemap format.")
        else:
            recommendations.append("[FAIL] The sitemap exists but is not valid XML. Fix the sitemap format.")
    elif sitemap.get('exists', False) and sitemap.get('is_valid_xml', True) and sitemap.get('url_count', 0) == 0 and not sitemap.get('is_index', False):
        recommendations.append("[WARNING] The sitemap exists but contains no URLs. Add content to your sitemap.")
    elif sitemap.get('exists', False) and sitemap.get('is_valid_xml', True):
        if sitemap.get('found_via') == 'robots.txt':
            recommendations.append("[PASS] Valid sitemap found and correctly declared in robots.txt.")
        else:
            recommendations.append("[PASS] Valid sitemap found.")
            if sitemap.get('found_via') != 'robots.txt':
                recommendations.append("[WARNING] Consider declaring your sitemap in the robots.txt file for better search engine discovery.")
    
    # Schema markup
    if not schema.get('has_schema', False):
        recommendations.append("[WARNING] No structured data (schema markup) detected. Consider adding schema markup for better search results.")
    else:
        recommendations.append("[PASS] Schema markup detected.")
    
    # Robots.txt
    if not robots.get('exists', False):
        recommendations.append("[WARNING] No robots.txt file found. Consider creating one to guide search engines.")
    elif robots.get('blocks_search_engines', False):
        if robots.get('blocked_engines'):
            blocked_engines_list = ', '.join(robots.get('blocked_engines', []))
            recommendations.append(f"[FAIL] The robots.txt file appears to block the following search engines: {blocked_engines_list}. Verify this is intentional.")
        elif robots.get('global_disallow_all', False):
            recommendations.append("[FAIL] The robots.txt file has a global 'Disallow: /' rule that may block search engines from crawling the site.")
        else:
            recommendations.append("[FAIL] The robots.txt file contains rules that may prevent search engines from indexing the site.")
    else:
        recommendations.append("[PASS] Robots.txt is properly configured to allow search engines to crawl the site.")
    
    # PageSpeed recommendations
    if pagespeed.get('success', False):
        mobile = pagespeed.get('mobile', {})
        
        # Performance score recommendations
        mobile_score = mobile.get('performance_score', 0)
        if mobile_score < 50:
            recommendations.append(f"[FAIL] Mobile page speed is poor ({int(mobile_score)}/100). This will negatively impact user experience and SEO.")
        elif mobile_score < 90:
            recommendations.append(f"[WARNING] Mobile page speed needs improvement ({int(mobile_score)}/100). Consider optimizing for better user experience.")
        else:
            recommendations.append(f"[PASS] Mobile page speed is good ({int(mobile_score)}/100).")
        
        # Core Web Vitals recommendations
        lcp_value = mobile.get('largest_contentful_paint')
        if lcp_value and lcp_value != 'N/A' and 's' in lcp_value:
            try:
                lcp_seconds = float(lcp_value.replace('s', ''))
                if lcp_seconds > 2.5:
                    recommendations.append(f"[WARNING] Largest Contentful Paint (LCP) is too slow at {lcp_value}. Aim for under 2.5 seconds.")
            except Exception:
                pass
        
        cls_value = mobile.get('cumulative_layout_shift')
        if cls_value and cls_value != 'N/A':
            try:
                cls_value_num = float(cls_value)
                if cls_value_num > 0.1:
                    recommendations.append(f"[WARNING] Cumulative Layout Shift (CLS) is too high at {cls_value}. Aim for under 0.1.")
            except Exception:
                pass
        
        tbt_value = mobile.get('total_blocking_time')
        if tbt_value and tbt_value != 'N/A' and 'ms' in tbt_value:
            try:
                tbt_ms = int(tbt_value.replace('ms', '').replace(',', ''))
                if tbt_ms > 200:
                    recommendations.append(f"[WARNING] Total Blocking Time (TBT) is too high at {tbt_value}. Aim for under 200 ms.")
            except Exception:
                pass
        
        # SEO score from PageSpeed
        seo_score = mobile.get('seo_score', 0)
        if seo_score < 90:
            recommendations.append(f"[WARNING] PageSpeed SEO score needs improvement ({int(seo_score)}/100).")
        
        # Add opportunities as recommendations
        opportunities = pagespeed.get('opportunities', [])
        for opportunity in opportunities[:3]:  # Limit to top 3 opportunities
            title = opportunity.get('title', 'Opportunity')
            display_value = opportunity.get('display_value', '')
            recommendations.append(f"[WARNING] {title}: {display_value}")
    
    # Defensive: If any major section was missing, add a warning
    for key in ['meta_tags', 'headings', 'images', 'content', 'links', 'robots', 'www_resolve', 'redirects', 'analytics', 'custom_404', 'https', 'sitemap', 'schema', 'accessibility', 'pagespeed']:
        if key not in analysis:
            recommendations.append(f"[WARNING] Analysis section '{key}' was missing from input data.")
    
    return recommendations