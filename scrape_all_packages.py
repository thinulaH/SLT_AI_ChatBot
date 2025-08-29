import csv
import json
import re
import os
import requests
import time
from pathlib import Path
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import logging

# Vector database imports
try:
    from langchain.schema import Document
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vector database configuration
PACKAGES_CHROMA_DIR = "./packages_chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def scrape_slt_packages():
    """Scrape SLT broadband packages from multiple pages using different methods"""
    urls = [
        {
            "url": "https://www.slt.lk/en/broadband/packages",
            "type": "postpaid",
            "name": "Regular Broadband Packages",
            "method": "playwright"  # Dynamic content requiring JS
        },
        {
            "url": "https://www.slt.lk/en/broadband/Prepaid-packages", 
            "type": "prepaid",
            "name": "Prepaid Broadband Packages",
            "method": "playwright"  # Dynamic content requiring JS
        },
        {
            "url": "https://www.slt.lk/en/personal/broadband/ftth/new-connection-charges", 
            "type": "connection_charges",
            "name": "New Connection Charges",
            "method": "requests"  # Static content, normal scraping
        },
        {
            "url": "https://www.slt.lk/en/personal/broadband/vdsl",
            "type": "vdsl",
            "name": "VDSL Packages",
            "method": "requests"
        },
        {
            "url": "https://www.slt.lk/en/personal/broadband/adsl",
            "type": "adsl", 
            "name": "ADSL Packages",
            "method": "requests"
        },
        {
            "url": "https://www.slt.lk/en/personal/broadband/ftth",
            "type": "ftth",
            "name": "FTTH Packages", 
            "method": "requests"
        }
    ]
    
    all_packages = []
    
    # Initialize Playwright - FIXED VERSION
    playwright = None
    browser = None
    
    try:
        # Group URLs by scraping method
        playwright_urls = [url for url in urls if url["method"] == "playwright"]
        requests_urls = [url for url in urls if url["method"] == "requests"]
        
        # Handle Playwright URLs
        if playwright_urls:
            playwright = sync_playwright().start()  # Fixed: Call start() directly
            browser = playwright.chromium.launch(headless=True)
            
            for url_info in playwright_urls:
                try:
                    logger.info(f"Scraping {url_info['name']} using Playwright from: {url_info['url']}")
                    packages = scrape_with_playwright(browser, url_info)
                    
                    # Add package type to each package
                    for package in packages:
                        package['package_type'] = url_info['type']
                        package['source_page'] = url_info['name']
                        package['scrape_method'] = 'playwright'
                    
                    all_packages.extend(packages)
                    logger.info(f"Found {len(packages)} packages from {url_info['name']}")
                    
                except Exception as e:
                    logger.error(f"Error scraping {url_info['name']} with Playwright: {e}")
                    continue
        
        # Handle Requests URLs
        for url_info in requests_urls:
            try:
                logger.info(f"Scraping {url_info['name']} using Requests from: {url_info['url']}")
                packages = scrape_with_requests(url_info)
                
                # Add package type to each package
                for package in packages:
                    package['package_type'] = url_info['type']
                    package['source_page'] = url_info['name']
                    package['scrape_method'] = 'requests'
                
                all_packages.extend(packages)
                logger.info(f"Found {len(packages)} packages from {url_info['name']}")
                
            except Exception as e:
                logger.error(f"Error scraping {url_info['name']} with Requests: {e}")
                continue
        
    finally:
        # Clean up Playwright resources - FIXED VERSION
        if browser:
            try:
                browser.close()
                logger.info("Browser closed successfully")
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
        
        if playwright:
            try:
                playwright.stop()
                logger.info("Playwright stopped successfully")
            except Exception as e:
                logger.warning(f"Error stopping Playwright: {e}")
    
    return all_packages

def scrape_with_playwright(browser, url_info):
    """Scrape packages using Playwright for dynamic content"""
    page = browser.new_page()
    
    try:
        # Set user agent to avoid detection
        page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        logger.info(f"Loading page with Playwright: {url_info['url']}")
        page.goto(url_info['url'], timeout=60000, wait_until='networkidle')
        
        # Wait for dynamic content to load - try different selectors for different page types
        wait_selectors = [
            'div.card-wrapp',
            'div.color-card', 
            'div.package-card',
            'div[class*="card"]',
            '.package-item',
            '.broadband-package'
        ]
        
        selector_found = False
        for selector in wait_selectors:
            try:
                page.wait_for_selector(selector, timeout=10000)
                selector_found = True
                logger.info(f"Found elements with selector: {selector}")
                break
            except:
                continue
        
        if not selector_found:
            logger.warning(f"No package cards found with standard selectors on {url_info['url']}")
        
        # Additional wait for dynamic content
        page.wait_for_timeout(5000)
        
        content = page.content()
        
    except Exception as e:
        logger.error(f"Error loading page {url_info['url']}: {e}")
        return []
    finally:
        try:
            page.close()
        except Exception as e:
            logger.warning(f"Error closing page: {e}")
    
    return parse_dynamic_page_content(content, url_info['type'])

def scrape_with_requests(url_info):
    """Scrape packages using requests for static content"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    try:
        logger.info(f"Making HTTP request to: {url_info['url']}")
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(url_info['url'], timeout=30)
        response.raise_for_status()
        
        logger.info(f"Successfully fetched content (Status: {response.status_code})")
        
        # Add delay to be respectful
        time.sleep(2)
        
        return parse_static_page_content(response.text, url_info)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching {url_info['url']}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error processing {url_info['url']}: {e}")
        return []

def parse_dynamic_page_content(content, page_type):
    """Parse the HTML content from Playwright to extract package information"""
    soup = BeautifulSoup(content, "html.parser")
    packages = []
    
    # Expanded card selectors for different page structures
    card_selectors = [
        "div.card-wrapp.any-time",
        "div.card-wrapp",
        "div.color-card",
        "div.package-card",
        "div[class*='card']",
        ".package-item",
        ".broadband-package",
        ".plan-card",
        "div[class*='package']",
        "div[class*='plan']"
    ]
    
    cards_found = []
    for selector in card_selectors:
        cards = soup.select(selector)
        if cards:
            logger.info(f"Found {len(cards)} cards with selector: {selector}")
        cards_found.extend(cards)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_cards = []
    for card in cards_found:
        card_str = str(card)
        if card_str not in seen:
            seen.add(card_str)
            unique_cards.append(card)
    
    logger.info(f"Found {len(unique_cards)} unique package cards")
    
    # If no cards found with standard selectors, try to find any div with package-related content
    if not unique_cards:
        logger.info("Trying alternative extraction methods...")
        
        # Look for any div containing pricing or package information
        potential_cards = soup.find_all('div', string=re.compile(r'Rs\.|GB|Mbps|package', re.I))
        potential_cards.extend(soup.find_all('div', class_=re.compile(r'price|package|plan|card', re.I)))
        
        for card in potential_cards:
            if card not in unique_cards:
                unique_cards.append(card)
        
        logger.info(f"Found {len(unique_cards)} potential package containers")
    
    for i, card in enumerate(unique_cards):
        try:
            package_data = extract_package_info_dynamic(card, page_type)
            if package_data and any(package_data.values()):  # Only add if has some data
                package_data['card_index'] = i + 1
                packages.append(package_data)
                logger.info(f"Extracted: {package_data.get('title', 'Untitled')} - {package_data.get('price', 'No price')}")
        except Exception as e:
            logger.warning(f"Error extracting data from card {i+1}: {e}")
    
    return packages

def parse_static_page_content(content, url_info):
    """Parse static HTML content from requests"""
    soup = BeautifulSoup(content, "html.parser")
    packages = []
    page_type = url_info['type']
    
    logger.info(f"Parsing static content for {url_info['name']}")
    
    # Different parsing strategies based on page type
    if page_type == "connection_charges":
        packages = parse_connection_charges(soup, url_info)
    elif page_type in ["vdsl", "adsl", "ftth"]:
        packages = parse_broadband_packages_static(soup, url_info)
    else:
        packages = parse_generic_packages_static(soup, url_info)
    
    return packages

def parse_connection_charges(soup, url_info):
    """Parse connection charges and fees"""
    packages = []
    
    # Look for tables with pricing information
    tables = soup.find_all('table')
    logger.info(f"Found {len(tables)} tables on connection charges page")
    
    for i, table in enumerate(tables):
        try:
            rows = table.find_all('tr')
            headers = []
            
            # Extract headers
            header_row = rows[0] if rows else None
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            
            # Extract data rows
            for j, row in enumerate(rows[1:], 1):
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    
                    # Create package entry
                    package = {
                        'title': row_data[0] if row_data else f'Connection Item {j}',
                        'data_amount': None,
                        'bundle_name': 'Connection Charges',
                        'price': row_data[1] if len(row_data) > 1 else None,
                        'speed': None,
                        'validity': None,
                        'description': ' | '.join(row_data[2:]) if len(row_data) > 2 else None,
                        'table_index': i + 1,
                        'row_index': j
                    }
                    
                    # Only add if has meaningful title and price
                    if (package['title'] and 
                        len(package['title']) > 3 and 
                        not package['title'].lower().startswith('rs')):
                        packages.append(package)
                        logger.info(f"Extracted connection charge: {package['title']} - {package['price']}")
        
        except Exception as e:
            logger.warning(f"Error parsing table {i+1}: {e}")
    
    # Look for list items with pricing
    list_items = soup.find_all('li')
    for i, item in enumerate(list_items):
        item_text = item.get_text(strip=True)
        if ('Rs.' in item_text or '₨' in item_text) and len(item_text) > 10:
            # Try to extract title and price
            price_match = re.search(r'(Rs\.?\s*[\d,]+(?:\.\d+)?|₨\s*[\d,]+(?:\.\d+)?)', item_text)
            if price_match:
                price = price_match.group(1)
                title = item_text.replace(price, '').strip(' :-')
                
                if title and len(title) > 3:
                    package = {
                        'title': title,
                        'data_amount': None,
                        'bundle_name': 'Connection Charges',
                        'price': price,
                        'speed': None,
                        'validity': None,
                        'description': None,
                        'list_item_index': i + 1
                    }
                    packages.append(package)
                    logger.info(f"Extracted from list: {title} - {price}")
    
    return packages

def parse_broadband_packages_static(soup, url_info):
    """Parse broadband packages from static pages (VDSL, ADSL, FTTH)"""
    packages = []
    page_type = url_info['type']
    
    # Look for package cards or containers
    card_selectors = [
        'div[class*="package"]',
        'div[class*="plan"]', 
        'div[class*="card"]',
        'div[class*="offer"]',
        '.package-container',
        '.plan-container',
        'div[class*="broadband"]'
    ]
    
    cards_found = []
    for selector in card_selectors:
        cards = soup.select(selector)
        if cards:
            logger.info(f"Found {len(cards)} potential package cards with selector: {selector}")
            cards_found.extend(cards)
    
    # Remove duplicates
    unique_cards = list({str(card): card for card in cards_found}.values())
    
    if not unique_cards:
        # Try to find tables with package information
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')[1:]  # Skip header
            for i, row in enumerate(rows):
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    package = create_package_from_table_row(row_data, page_type, i)
                    if package:
                        packages.append(package)
    else:
        # Parse package cards
        for i, card in enumerate(unique_cards):
            try:
                package_data = extract_package_info_static(card, page_type)
                if package_data and any(package_data.values()):
                    package_data['card_index'] = i + 1
                    packages.append(package_data)
                    logger.info(f"Extracted static package: {package_data.get('title', 'Untitled')}")
            except Exception as e:
                logger.warning(f"Error extracting static package data from card {i+1}: {e}")
    
    # If still no packages, try parsing div elements with pricing
    if not packages:
        divs_with_price = soup.find_all('div', string=re.compile(r'Rs\.|₨|\d+\s*GB|\d+\s*Mbps', re.I))
        for i, div in enumerate(divs_with_price[:20]):  # Limit to prevent spam
            parent = div.parent
            if parent:
                package = extract_package_info_static(parent, page_type)
                if package and package.get('title'):
                    package['div_index'] = i + 1
                    packages.append(package)
    
    return packages

def parse_generic_packages_static(soup, url_info):
    """Generic parser for other static pages"""
    packages = []
    
    # Look for any content that might contain package information
    content_selectors = [
        'div[class*="content"]',
        'div[class*="main"]',
        'div[class*="body"]',
        '.container'
    ]
    
    for selector in content_selectors:
        content_divs = soup.select(selector)
        for div in content_divs:
            # Look for pricing patterns
            price_elements = div.find_all(string=re.compile(r'Rs\.?\s*\d+|₨\s*\d+', re.I))
            for price_text in price_elements:
                parent = price_text.parent
                if parent:
                    package = extract_package_info_static(parent, url_info['type'])
                    if package and package.get('title'):
                        packages.append(package)
                        if len(packages) >= 20:  # Limit to prevent spam
                            break
    
    return packages

def create_package_from_table_row(row_data, page_type, index):
    """Create a package dict from table row data"""
    if len(row_data) < 2:
        return None
    
    title = row_data[0] if row_data[0] and len(row_data[0]) > 3 else None
    if not title or title.lower().startswith('rs'):
        return None
    
    package = {
        'title': title,
        'data_amount': None,
        'bundle_name': page_type.upper(),
        'price': row_data[1] if len(row_data) > 1 else None,
        'speed': None,
        'validity': None,
        'description': ' | '.join(row_data[2:]) if len(row_data) > 2 else None,
        'row_index': index + 1
    }
    
    # Try to extract additional info from description
    description = package.get('description', '')
    if description:
        # Look for speed info
        speed_match = re.search(r'(\d+(?:\.\d+)?)\s*(Mbps|MB/s)', description, re.I)
        if speed_match:
            package['speed'] = speed_match.group(0)
        
        # Look for data info
        data_match = re.search(r'(\d+(?:\.\d+)?)\s*(GB|TB|MB)', description, re.I)
        if data_match:
            package['data_amount'] = data_match.group(0)
    
    return package

def extract_package_info_dynamic(card, page_type):
    """Extract package information from a card element (for dynamic content)"""
    package = {
        "title": None,
        "data_amount": None,
        "bundle_name": None,
        "price": None,
        "speed": None,
        "validity": None,
        "description": None
    }
    
    # Extract title (try multiple selectors)
    title_selectors = [
        "h4.color-card-title",
        "h3.color-card-title", 
        ".color-card-title",
        "h4", "h3", "h2", "h5",
        ".package-name",
        ".plan-name",
        ".title"
    ]
    
    for selector in title_selectors:
        title_elem = card.select_one(selector)
        if title_elem:
            # Remove strong tags to get clean title
            for strong in title_elem.select("strong"):
                strong.extract()
            title = title_elem.get_text(strip=True)
            if title and len(title) > 2:  # Ensure title is meaningful
                package["title"] = title
                break
    
    # Extract data amount with expanded patterns
    data_selectors = [
        "span.data-val-amount",
        ".data-val-amount",
        "span[class*='data']",
        ".data-amount",
        ".quota"
    ]
    
    for selector in data_selectors:
        data_elem = card.select_one(selector)
        if data_elem:
            package["data_amount"] = data_elem.get_text(strip=True)
            break
    
    # If no data amount found, try regex on card text
    if not package["data_amount"]:
        card_text = card.get_text()
        data_match = re.search(r'(\d+(?:\.\d+)?)\s*(GB|TB|MB)', card_text, re.IGNORECASE)
        if data_match:
            package["data_amount"] = data_match.group(0)
    
    # Extract bundle name
    bundle_selectors = [
        "h5.bundle-name",
        ".bundle-name",
        "h5",
        ".package-subtitle",
        ".plan-subtitle"
    ]
    
    for selector in bundle_selectors:
        bundle_elem = card.select_one(selector)
        if bundle_elem:
            bundle_text = bundle_elem.get_text(strip=True)
            if bundle_text and bundle_text != package["title"]:
                package["bundle_name"] = bundle_text
                break
    
    # Extract price with more comprehensive patterns
    price_selectors = [
        "div.color-card-price",
        ".color-card-price",
        ".price",
        "div[class*='price']",
        ".cost",
        ".amount"
    ]
    
    for selector in price_selectors:
        price_elem = card.select_one(selector)
        if price_elem:
            price_text = price_elem.get_text(strip=True)
            # Clean up price text but preserve important parts
            if 'Rs' in price_text or '₨' in price_text or re.search(r'\d+', price_text):
                package["price"] = price_text
                break
    
    # If no price found with selectors, try regex on card text
    if not package["price"]:
        card_text = card.get_text()
        price_patterns = [
            r'Rs\.?\s*[\d,]+(?:\.\d+)?',
            r'₨\s*[\d,]+(?:\.\d+)?',
            r'LKR\s*[\d,]+(?:\.\d+)?',
            r'[\d,]+(?:\.\d+)?\s*(?:Rs|₨|LKR)',
        ]
        
        for pattern in price_patterns:
            price_match = re.search(pattern, card_text, re.IGNORECASE)
            if price_match:
                package["price"] = price_match.group(0)
                break
    
    # Try to extract speed information with expanded patterns
    card_text = card.get_text()
    speed_patterns = [
        r'(\d+(?:\.\d+)?)\s*(Mbps|MB/s|Gbps|GB/s)',
        r'up\s+to\s+(\d+(?:\.\d+)?)\s*(Mbps|MB/s)',
        r'speed[:\s]*(\d+(?:\.\d+)?)\s*(Mbps|MB/s)'
    ]
    
    for pattern in speed_patterns:
        speed_match = re.search(pattern, card_text, re.IGNORECASE)
        if speed_match:
            package["speed"] = speed_match.group(0)
            break
    
    # Try to extract validity with expanded patterns
    validity_patterns = [
        r'(\d+)\s*(days?|months?|weeks?|hrs?|hours?)',
        r'valid\s+for\s+(\d+)\s*(days?|months?|weeks?)',
        r'validity[:\s]*(\d+)\s*(days?|months?|weeks?)'
    ]
    
    for pattern in validity_patterns:
        validity_match = re.search(pattern, card_text, re.IGNORECASE)
        if validity_match:
            package["validity"] = validity_match.group(0)
            break
    
    # Extract any additional description
    desc_selectors = [
        ".color-card-body",
        ".card-body",
        ".description",
        ".details",
        "p"
    ]
    
    for selector in desc_selectors:
        desc_elem = card.select_one(selector)
        if desc_elem:
            desc_text = desc_elem.get_text(strip=True)
            if (len(desc_text) > 10 and 
                desc_text not in [package["title"], package["bundle_name"]] and
                not any(desc_text == package[key] for key in package if package[key])):
                package["description"] = desc_text[:200] + "..." if len(desc_text) > 200 else desc_text
                break
    
    return package

def extract_package_info_static(card, page_type):
    """Extract package information from static HTML elements"""
    package = {
        "title": None,
        "data_amount": None,
        "bundle_name": None,
        "price": None,
        "speed": None,
        "validity": None,
        "description": None
    }
    
    card_text = card.get_text()
    
    # Extract title - look for headings first
    title_elements = card.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    for elem in title_elements:
        title = elem.get_text(strip=True)
        if title and len(title) > 3 and not title.lower().startswith('rs'):
            package["title"] = title
            break
    
    # If no heading found, try to extract from strong or first meaningful text
    if not package["title"]:
        strong_elements = card.find_all(['strong', 'b'])
        for elem in strong_elements:
            title = elem.get_text(strip=True)
            if title and len(title) > 3 and not title.lower().startswith('rs'):
                package["title"] = title
                break
    
    # Bundle name based on page type
    if page_type in ['vdsl', 'adsl', 'ftth']:
        package["bundle_name"] = page_type.upper()
    elif page_type == 'connection_charges':
        package["bundle_name"] = 'Connection Charges'
    
    # Extract price using regex
    price_patterns = [
        r'Rs\.?\s*[\d,]+(?:\.\d+)?',
        r'₨\s*[\d,]+(?:\.\d+)?',
        r'LKR\s*[\d,]+(?:\.\d+)?'
    ]
    
    for pattern in price_patterns:
        price_match = re.search(pattern, card_text, re.IGNORECASE)
        if price_match:
            package["price"] = price_match.group(0)
            break
    
    # Extract data amount
    data_patterns = [
        r'(\d+(?:\.\d+)?)\s*(GB|TB|MB)',
        r'(\d+(?:\.\d+)?)\s*GB',
        r'unlimited'
    ]
    
    for pattern in data_patterns:
        data_match = re.search(pattern, card_text, re.IGNORECASE)
        if data_match:
            package["data_amount"] = data_match.group(0)
            break
    
    # Extract speed
    speed_patterns = [
        r'(\d+(?:\.\d+)?)\s*(Mbps|MB/s|Gbps)',
        r'up\s+to\s+(\d+(?:\.\d+)?)\s*Mbps'
    ]
    
    for pattern in speed_patterns:
        speed_match = re.search(pattern, card_text, re.IGNORECASE)
        if speed_match:
            package["speed"] = speed_match.group(0)
            break
    
    # Extract validity
    validity_patterns = [
        r'(\d+)\s*(days?|months?|weeks?)',
        r'valid\s+for\s+(\d+)\s*(days?|months?)'
    ]
    
    for pattern in validity_patterns:
        validity_match = re.search(pattern, card_text, re.IGNORECASE)
        if validity_match:
            package["validity"] = validity_match.group(0)
            break
    
    # Extract description (first paragraph or meaningful text)
    paragraphs = card.find_all('p')
    for p in paragraphs:
        desc_text = p.get_text(strip=True)
        if len(desc_text) > 15 and desc_text != package["title"]:
            package["description"] = desc_text[:150] + "..." if len(desc_text) > 150 else desc_text
            break
    
    return package

def filter_valid_packages(packages):
    """Filter out invalid packages with null titles or price-only titles"""
    valid_packages = []
    skipped_count = 0
    
    for package in packages:
        title = package.get('title', '').strip() if package.get('title') else ''
        
        # Skip packages with null/empty titles
        if not title:
            skipped_count += 1
            logger.debug(f"Skipped package with null/empty title: {package.get('card_index', 'unknown')}")
            continue
            
        # Skip packages where title starts with "Rs."
        if title.lower().startswith('rs.'):
            skipped_count += 1
            logger.debug(f"Skipped package with price-only title '{title}': card_index {package.get('card_index', 'unknown')}")
            continue
            
        # Skip packages that are just "Data Bundle"
        if title.lower().strip() == 'data bundle':
            skipped_count += 1
            logger.debug(f"Skipped generic 'Data Bundle' title: card_index {package.get('card_index', 'unknown')}")
            continue
            
        valid_packages.append(package)
    
    logger.info(f"Filtered {len(valid_packages)} valid packages, skipped {skipped_count} invalid packages")
    return valid_packages

def remove_duplicates(packages):
    """Remove duplicate packages based on multiple fields"""
    seen = set()
    unique_packages = []
    
    for package in packages:
        title = package.get('title', '').strip() if package.get('title') else ''
        
        # Create a unique key for deduplication
        key = (
            title.lower(),
            package.get('data_amount', ''),
            package.get('bundle_name', ''),
            package.get('price', ''),
            package.get('speed', ''),
            package.get('validity', ''),
            package.get('package_type', ''),
            package.get('scrape_method', '')
        )
        
        if key not in seen:
            seen.add(key)
            unique_packages.append(package)
    
    logger.info(f"Removed duplicates: {len(packages)} -> {len(unique_packages)} packages")
    return unique_packages

def clean_and_parse_price(price_str):
    """Extract monthly rental and startup fee from price string"""
    result = {"monthly": None, "startup": None, "monthly_num": 0, "startup_num": 0, "original": price_str}
    
    if not price_str:
        return result
    
    # Convert to string if it's not already
    price_str = str(price_str) if price_str is not None else ''
    
    # Clean up the price string first
    price_clean = re.sub(r'[Rr]{2,}s\.', 'Rs.', price_str)
    
    # Extract monthly rental - multiple patterns
    monthly_patterns = [
        r'Monthly\s+Rental\s*:?\s*Rs\.?\s*(\d+(?:,\d+)*)',
        r'Rs\.?\s*(\d+(?:,\d+)*)\s*/?month',
        r'Rs\.?\s*(\d+(?:,\d+)*)\s*monthly',
        r'(?:Reload/Top-up\s+amount)?Rs\.?\s*(\d+(?:,\d+)*)',
    ]
    
    for pattern in monthly_patterns:
        monthly_match = re.search(pattern, price_clean, re.IGNORECASE)
        if monthly_match:
            result["monthly"] = f"Rs.{monthly_match.group(1)}"
            result["monthly_num"] = int(monthly_match.group(1).replace(',', ''))
            break
    
    # Extract startup fee
    startup_patterns = [
        r'Startup\s+Fee\s*:?\s*Rs\.?\s*(\d+(?:,\d+)*)',
        r'Setup\s+Fee\s*:?\s*Rs\.?\s*(\d+(?:,\d+)*)',
        r'Installation\s*:?\s*Rs\.?\s*(\d+(?:,\d+)*)'
    ]
    
    for pattern in startup_patterns:
        startup_match = re.search(pattern, price_clean, re.IGNORECASE)
        if startup_match:
            result["startup"] = f"Rs.{startup_match.group(1)}"
            result["startup_num"] = int(startup_match.group(1).replace(',', ''))
            break
    
    return result

def normalize_data_amount(data_str):
    """Normalize and extract meaningful info from data amount"""
    result = {"original": data_str, "normalized": "", "is_unlimited": False, "gb_amount": 0, "tb_amount": 0}
    
    if not data_str:
        return result
    
    data_lower = data_str.lower()
    
    # Check for unlimited indicators
    if any(term in data_lower for term in ['unlimited', 'up to', 'daily']):
        result["is_unlimited"] = True
        if 'daily' in data_lower:
            result["normalized"] = "Daily unlimited usage"
        else:
            result["normalized"] = "Unlimited usage"
    else:
        # Extract TB amount first (higher priority)
        tb_match = re.search(r'(\d+(?:\.\d+)?)\s*tb', data_str, re.IGNORECASE)
        if tb_match:
            result["tb_amount"] = float(tb_match.group(1))
            result["gb_amount"] = int(result["tb_amount"] * 1000)
            result["normalized"] = f"{tb_match.group(1)}TB"
        else:
            # Extract GB amount
            gb_match = re.search(r'(\d+(?:\.\d+)?)\s*gb', data_str, re.IGNORECASE)
            if gb_match:
                result["gb_amount"] = int(float(gb_match.group(1)))
                result["normalized"] = f"{result['gb_amount']}GB"
            else:
                # Handle cases where it's just a number (assume GB)
                num_match = re.search(r'^(\d+(?:\.\d+)?)', data_str.strip())
                if num_match:
                    result["gb_amount"] = int(float(num_match.group(1)))
                    result["normalized"] = f"{result['gb_amount']}GB"
                else:
                    result["normalized"] = data_str
    
    return result

def normalize_speed(speed_str):
    """Normalize speed information"""
    result = {"original": speed_str, "normalized": "", "mbps": 0}
    
    if not speed_str:
        return result
    
    # Extract speed in Mbps
    mbps_match = re.search(r'(\d+(?:\.\d+)?)\s*mbps', speed_str, re.IGNORECASE)
    if mbps_match:
        result["mbps"] = int(float(mbps_match.group(1)))
        result["normalized"] = f"{result['mbps']} Mbps"
    else:
        # Try to extract just numbers and assume Mbps
        num_match = re.search(r'(\d+(?:\.\d+)?)', speed_str)
        if num_match:
            result["mbps"] = int(float(num_match.group(1)))
            result["normalized"] = f"{result['mbps']} Mbps"
        else:
            result["normalized"] = speed_str
    
    return result

def normalize_validity(validity_str):
    """Normalize validity/duration information"""
    result = {"original": validity_str, "normalized": "", "days": 0}
    
    if not validity_str:
        return result
    
    validity_lower = validity_str.lower()
    
    # Extract days
    if 'day' in validity_lower:
        days_match = re.search(r'(\d+)\s*days?', validity_str, re.IGNORECASE)
        if days_match:
            result["days"] = int(days_match.group(1))
            result["normalized"] = f"{result['days']} days"
    
    # Extract months (convert to days)
    elif 'month' in validity_lower:
        months_match = re.search(r'(\d+)\s*months?', validity_str, re.IGNORECASE)
        if months_match:
            result["days"] = int(months_match.group(1)) * 30
            result["normalized"] = f"{months_match.group(1)} months"
    
    # Extract weeks (convert to days)
    elif 'week' in validity_lower:
        weeks_match = re.search(r'(\d+)\s*weeks?', validity_str, re.IGNORECASE)
        if weeks_match:
            result["days"] = int(weeks_match.group(1)) * 7
            result["normalized"] = f"{weeks_match.group(1)} weeks"
    
    if not result["normalized"]:
        result["normalized"] = validity_str
    
    return result

def create_package_documents(packages):
    """Create comprehensive package documents for vector database"""
    if not VECTOR_DB_AVAILABLE:
        logger.warning("Vector database libraries not available. Skipping document creation.")
        return []
    
    documents = []
    
    # Filter and process packages
    valid_packages = filter_valid_packages(packages)
    unique_packages = remove_duplicates(valid_packages)
    
    logger.info(f"Creating documents from {len(unique_packages)} unique valid packages...")
    
    for i, package in enumerate(unique_packages):
        title = package.get('title', '').strip() if package.get('title') else ''
        bundle_name = package.get('bundle_name', '').strip() if package.get('bundle_name') else ''
        data_info = normalize_data_amount(package.get('data_amount', ''))
        price_info = clean_and_parse_price(package.get('price', ''))
        speed_info = normalize_speed(package.get('speed', ''))
        validity_info = normalize_validity(package.get('validity', ''))
        description = package.get('description', '').strip() if package.get('description') else ''
        package_type = package.get('package_type', 'unknown')
        source_page = package.get('source_page', '')
        card_index = package.get('card_index', i + 1)
        scrape_method = package.get('scrape_method', 'unknown')
        
        # Skip packages that don't have meaningful data
        if not any([title, bundle_name, data_info['normalized'], price_info.get('monthly'), speed_info['normalized']]):
            logger.debug(f"Skipping package with insufficient data: card_index {card_index}")
            continue
        
        # Document 1: Primary package information
        primary_content = f"""SLT {title} {'Package' if not title.endswith('Package') else ''}

Package Name: {title}
Package Type: {package_type.title().replace('_', ' ')}
Connection Type: {bundle_name if bundle_name else 'Broadband'}
Data Allowance: {data_info['normalized'] if data_info['normalized'] else 'Contact SLT'}
Speed: {speed_info['normalized'] if speed_info['normalized'] else 'High Speed'}
Validity: {validity_info['normalized'] if validity_info['normalized'] else 'Monthly' if package_type == 'postpaid' else 'Limited Period'}
{'Monthly Price' if package_type == 'postpaid' else 'Price'}: {price_info.get('monthly', 'Contact SLT')}
{'Setup Fee' if package_type == 'postpaid' else 'Additional Fee'}: {price_info.get('startup', 'Rs.0')}

This is an SLT {package_type.replace('_', ' ')} {'broadband internet package' if package_type in ['postpaid', 'prepaid'] else 'service'} offering {data_info['normalized'] if data_info['normalized'] else 'reliable service'} {'data' if package_type in ['postpaid', 'prepaid'] else ''} with {bundle_name if bundle_name else 'broadband'} connection technology{f' at {speed_info["normalized"]}' if speed_info['normalized'] else ''}.
{'Perfect for home and business internet needs in Sri Lanka.' if package_type in ['postpaid', 'vdsl', 'adsl', 'ftth'] else 'Ideal for flexible usage without monthly commitments.' if package_type == 'prepaid' else 'Essential service for SLT broadband connections.'}

Service Details:
- {'Fast and reliable' if bundle_name else 'Quality'} {bundle_name if bundle_name else 'broadband'} {'connection' if package_type in ['postpaid', 'prepaid', 'vdsl', 'adsl', 'ftth'] else 'service'}
- {'Unlimited internet usage' if data_info['is_unlimited'] else f'{data_info["normalized"]} data allowance' if data_info['normalized'] else 'Quality service'}
{f'- Connection speed: {speed_info["normalized"]}' if speed_info['normalized'] else ''}
{f'- Validity: {validity_info["normalized"]}' if validity_info['normalized'] else ''}
- {'Monthly subscription' if package_type == 'postpaid' else 'Service fee'}: {price_info.get('monthly', 'Available on request')}
- {'One-time setup' if package_type == 'postpaid' else 'Additional charges'}: {price_info.get('startup', 'Rs.0')}
- Available island-wide through SLT network
{f'- Additional info: {description}' if description else ''}
"""
        
        documents.append(Document(
            page_content=primary_content,
            metadata={
                "source": "slt_packages_database",
                "title": title,
                "type": "broadband_package",
                "subtype": "primary_info",
                "package_type": package_type,
                "bundle_name": bundle_name,
                "connection_type": bundle_name.lower() if bundle_name else "broadband",
                "data_amount": data_info['normalized'],
                "data_gb": data_info['gb_amount'],
                "data_tb": data_info['tb_amount'],
                "is_unlimited": data_info['is_unlimited'],
                "speed": speed_info['normalized'],
                "speed_mbps": speed_info['mbps'],
                "validity": validity_info['normalized'],
                "validity_days": validity_info['days'],
                "monthly_price": price_info.get('monthly', ''),
                "monthly_price_num": price_info['monthly_num'],
                "startup_fee": price_info.get('startup', ''),
                "startup_fee_num": price_info['startup_num'],
                "description": description,
                "source_page": source_page,
                "card_index": card_index,
                "package_id": i + 1,
                "scrape_method": scrape_method
            }
        ))
        
        # Document 2: Search-optimized content
        search_terms = []
        
        # Title variations
        search_terms.extend([
            f"SLT {title}", f"{title} package", f"{title} plan", f"{title} broadband",
            f"{title} internet", f"SLT {title} package", f"SLT {title} plan"
        ])
        
        # Package type variations
        if package_type == 'prepaid':
            search_terms.extend([
                "prepaid broadband", "prepaid internet", "top-up internet", "reload internet",
                "no contract internet", "flexible internet", "temporary internet"
            ])
        elif package_type == 'postpaid':
            search_terms.extend([
                "monthly broadband", "contract internet", "permanent internet", "home broadband"
            ])
        elif package_type == 'connection_charges':
            search_terms.extend([
                "connection charges", "setup fees", "installation costs", "new connection",
                "broadband setup", "internet installation", "SLT charges"
            ])
        elif package_type in ['vdsl', 'adsl', 'ftth']:
            search_terms.extend([
                f"{package_type.upper()} broadband", f"{package_type} internet",
                f"SLT {package_type.upper()}", f"{package_type} connection"
            ])
        
        # Connection type variations
        if bundle_name and bundle_name.lower() in ['fibre', 'fiber', 'ftth']:
            search_terms.extend([
                "fiber broadband", "fibre broadband", "fiber internet", "fibre internet",
                "fiber optic", "FTTH", "high speed fiber", "fast internet",
                "fiber connection", "optical fiber", "fiber to home"
            ])
        elif bundle_name and '4g' in bundle_name.lower():
            search_terms.extend([
                "4G broadband", "mobile broadband", "wireless broadband", "LTE broadband",
                "4G internet", "mobile internet", "wireless internet", "portable internet"
            ])
        elif bundle_name and 'adsl' in bundle_name.lower():
            search_terms.extend([
                "ADSL broadband", "ADSL internet", "phone line internet",
                "traditional broadband", "copper line internet"
            ])
        elif bundle_name and 'vdsl' in bundle_name.lower():
            search_terms.extend([
                "VDSL broadband", "VDSL internet", "high speed ADSL",
                "enhanced broadband", "fast copper internet"
            ])
        
        # Search-optimized content
        search_content = f"""SLT Service Search: {title}

Package Type: {package_type.title().replace('_', ' ')}
Search Terms: {' | '.join(search_terms[:15])}

{title} is a {package_type.replace('_', ' ')} {bundle_name if bundle_name else 'service'} {'package' if package_type in ['postpaid', 'prepaid'] else 'offering'} {data_info['normalized'] if data_info['normalized'] else 'reliable'} {'internet connectivity' if package_type in ['postpaid', 'prepaid', 'vdsl', 'adsl', 'ftth'] else 'service'}{f' at {speed_info["normalized"]}' if speed_info['normalized'] else ''} for {price_info.get('monthly', 'competitive pricing')}.

Perfect for customers looking for: {' '.join(search_terms[:10])}
"""
        
        documents.append(Document(
            page_content=search_content,
            metadata={
                "source": "slt_packages_database",
                "title": title,
                "type": "broadband_package",
                "subtype": "search_optimized",
                "package_type": package_type,
                "bundle_name": bundle_name,
                "connection_type": bundle_name.lower() if bundle_name else "broadband",
                "data_amount": data_info['normalized'],
                "data_gb": data_info['gb_amount'],
                "data_tb": data_info['tb_amount'],
                "is_unlimited": data_info['is_unlimited'],
                "speed": speed_info['normalized'],
                "speed_mbps": speed_info['mbps'],
                "validity": validity_info['normalized'],
                "validity_days": validity_info['days'],
                "monthly_price": price_info.get('monthly', ''),
                "monthly_price_num": price_info['monthly_num'],
                "startup_fee": price_info.get('startup', ''),
                "startup_fee_num": price_info['startup_num'],
                "description": description,
                "source_page": source_page,
                "card_index": card_index,
                "package_id": i + 1,
                "scrape_method": scrape_method
            }
        ))
    
    logger.info(f"Created {len(documents)} package documents")
    return documents

def create_packages_vector_database(documents):
    """Create a vector database for SLT packages"""
    if not VECTOR_DB_AVAILABLE:
        logger.warning("Vector database libraries not available. Skipping vector database creation.")
        return None
    
    if not documents:
        logger.warning("No documents provided for vector database creation.")
        return None
    
    # Remove existing database if it exists
    if Path(PACKAGES_CHROMA_DIR).exists():
        import shutil
        logger.info(f"Removing existing packages database at {PACKAGES_CHROMA_DIR}")
        shutil.rmtree(PACKAGES_CHROMA_DIR)
    
    logger.info(f"Creating new packages vector database at {PACKAGES_CHROMA_DIR}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Create new vector database
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PACKAGES_CHROMA_DIR
    )
    
    logger.info(f"Packages vector database created with {len(documents)} documents")
    return vector_store

def test_vector_database(vector_store):
    """Test the vector database with sample queries"""
    if not vector_store:
        return
    
    logger.info("\nTesting vector database with sample queries...")
    
    test_queries = [
        "fiber broadband packages",
        "unlimited internet",
        "prepaid 4G packages", 
        "cheap broadband under 2000",
        "connection charges",
        "VDSL packages",
        "ADSL internet",
        "FTTH fiber",
        "100GB data package"
    ]
    
    for query in test_queries:
        results = vector_store.similarity_search(query, k=2)
        logger.info(f"Query: '{query}' -> Found {len(results)} results")
        for i, doc in enumerate(results):
            title = doc.metadata.get('title', 'Unknown')
            package_type = doc.metadata.get('package_type', 'Unknown')
            scrape_method = doc.metadata.get('scrape_method', 'Unknown')
            logger.info(f"  {i+1}. {title} ({package_type}) - {scrape_method}")

def save_results(packages, filename_prefix="slt_all_packages"):
    """Save results to CSV and JSON files"""
    if not packages:
        logger.warning("No packages found to save")
        return
    
    # Normalize all packages to have the same fields
    all_fieldnames = set()
    for package in packages:
        all_fieldnames.update(package.keys())
    
    # Convert set to sorted list for consistent ordering
    fieldnames = sorted(list(all_fieldnames))
    
    # Ensure all packages have all fields (fill missing with None)
    normalized_packages = []
    for package in packages:
        normalized_package = {}
        for field in fieldnames:
            normalized_package[field] = package.get(field, None)
        normalized_packages.append(normalized_package)
    
    # Save to CSV
    csv_filename = f"{filename_prefix}.csv"
    try:
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(normalized_packages)
        logger.info(f"✅ Saved {len(packages)} packages to {csv_filename}")
    except Exception as e:
        logger.error(f"Error saving CSV: {e}")
    
    # Save to JSON
    json_filename = f"{filename_prefix}.json"
    try:
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(normalized_packages, f, ensure_ascii=False, indent=4)
        logger.info(f"✅ Saved {len(packages)} packages to {json_filename}")
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")

def main():
    """Main function to run the enhanced scraper"""
    logger.info("Starting enhanced SLT package scraper with mixed scraping methods...")
    
    # Check if vector database libraries are available
    if VECTOR_DB_AVAILABLE:
        logger.info("Vector database support: ENABLED")
    else:
        logger.warning("Vector database support: DISABLED (install langchain and chromadb to enable)")
    
    # Scrape packages
    results = scrape_slt_packages()
    
    if results:
        # Save to files
        save_results(results)
        
        # Create vector database if libraries are available
        if VECTOR_DB_AVAILABLE:
            logger.info("Creating vector database...")
            documents = create_package_documents(results)
            if documents:
                vector_store = create_packages_vector_database(documents)
                if vector_store:
                    test_vector_database(vector_store)
                    logger.info(f"Vector database created successfully at: {PACKAGES_CHROMA_DIR}")
                else:
                    logger.error("Failed to create vector database")
            else:
                logger.warning("No documents created for vector database")
        
        # Print summary
        print(f"\n📊 SCRAPING SUMMARY")
        print(f"{'='*60}")
        print(f"Total packages scraped: {len(results)}")
        
        # Group by package type and scraping method
        by_type = {}
        by_method = {}
        
        for p in results:
            pkg_type = p.get('package_type', 'unknown')
            scrape_method = p.get('scrape_method', 'unknown')
            
            by_type[pkg_type] = by_type.get(pkg_type, 0) + 1
            by_method[scrape_method] = by_method.get(scrape_method, 0) + 1
        
        print("\nBy Package Type:")
        for pkg_type, count in sorted(by_type.items()):
            print(f"  {pkg_type.title().replace('_', ' ')}: {count}")
        
        print("\nBy Scraping Method:")
        for method, count in sorted(by_method.items()):
            print(f"  {method.title()}: {count}")
        
        # Filter valid packages for summary
        valid_packages = filter_valid_packages(results)
        unique_valid = remove_duplicates(valid_packages)
        print(f"\nValid unique packages: {len(unique_valid)}")
        
        if VECTOR_DB_AVAILABLE and os.path.exists(PACKAGES_CHROMA_DIR):
            print(f"Vector database: CREATED ({PACKAGES_CHROMA_DIR})")
        else:
            print(f"Vector database: NOT CREATED")
        
        print(f"{'='*60}")
        
        # Show first few valid packages from each type
        shown_by_type = {}
        for pkg in unique_valid:
            pkg_type = pkg.get('package_type', 'unknown')
            if shown_by_type.get(pkg_type, 0) < 3:  # Show max 3 per type
                shown_by_type[pkg_type] = shown_by_type.get(pkg_type, 0) + 1
                
                print(f"\n{pkg_type.title().replace('_', ' ')} Example:")
                print(f"   Title: {pkg.get('title', 'N/A')}")
                print(f"   Source: {pkg.get('source_page', 'N/A')}")
                print(f"   Method: {pkg.get('scrape_method', 'N/A')}")
                if pkg.get('data_amount'):
                    print(f"   Data: {pkg.get('data_amount')}")
                if pkg.get('price'):
                    print(f"   Price: {pkg.get('price')}")
                if pkg.get('speed'):
                    print(f"   Speed: {pkg.get('speed')}")
                if pkg.get('validity'):
                    print(f"   Validity: {pkg.get('validity')}")
            
    else:
        logger.warning("No packages were scraped. Check the website structure or selectors.")

if __name__ == "__main__":
    main()