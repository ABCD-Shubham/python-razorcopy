import xml.etree.ElementTree as ET
import json
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def decode_sitemap(sitemap_data):
    """
    Decodes a sitemap stored as XML or JSON and extracts URLs, titles, and meta titles.

    :param sitemap_data: Raw sitemap content (XML/JSON string)
    :return: List of dictionaries containing 'url', 'title', and 'metaTitle'
    """
    try:
        if sitemap_data.strip().startswith("<"):  # XML format
            root = ET.fromstring(sitemap_data)
            urls = []
            for url in root.findall(".//url"):
                loc = url.find("loc").text if url.find("loc") is not None else ""
                title = url.find("title").text if url.find("title") is not None else "N/A"
                meta_title = url.find("metaTitle").text if url.find("metaTitle") is not None else "N/A"
                urls.append({"url": loc, "title": title, "metaTitle": meta_title})
            return urls
        
        elif sitemap_data.strip().startswith("{") or sitemap_data.strip().startswith("["):  # JSON format
            parsed_data = json.loads(sitemap_data)
            return parsed_data if isinstance(parsed_data, list) else [parsed_data]
        
        else:
            raise ValueError("Unsupported sitemap format")
    
    except Exception as e:
        logger.error(f"Failed to decode sitemap: {e}")
        return []