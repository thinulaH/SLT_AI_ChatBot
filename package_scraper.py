import csv
import json
import re
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scrape_slt_packages():
    """Scrape SLT broadband packages from their website"""
    url = "https://www.slt.lk/en/broadband/packages"
    
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Set user agent to avoid detection
            page.set_extra_http_headers({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            logger.info(f"Loading page: {url}")
            page.goto(url, timeout=60000, wait_until='networkidle')
            
            # Wait for dynamic content to load
            page.wait_for_selector('div.card-wrapp, div.color-card', timeout=30000)
            page.wait_for_timeout(5000)
            
            content = page.content()
            browser.close()
            
        except Exception as e:
            logger.error(f"Error loading page: {e}")
            if 'browser' in locals():
                browser.close()
            return []
    
    soup = BeautifulSoup(content, "html.parser")
    packages = []
    
    # Look for different card types based on the HTML structure
    card_selectors = [
        "div.card-wrapp.any-time",
        "div.color-card",
        "div[class*='card']"
    ]
    
    cards_found = []
    for selector in card_selectors:
        cards = soup.select(selector)
        cards_found.extend(cards)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_cards = []
    for card in cards_found:
        card_str = str(card)
        if card_str not in seen:
            seen.add(card_str)
            unique_cards.append(card)
    
    logger.info(f"Found {len(unique_cards)} package cards")
    
    for i, card in enumerate(unique_cards):
        try:
            package_data = extract_package_info(card)
            if package_data and any(package_data.values()):  # Only add if has some data
                package_data['card_index'] = i + 1
                packages.append(package_data)
                logger.info(f"Extracted: {package_data}")
        except Exception as e:
            logger.warning(f"Error extracting data from card {i+1}: {e}")
    
    return packages

def extract_package_info(card):
    """Extract package information from a card element"""
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
        "h4",
        "h3"
    ]
    
    for selector in title_selectors:
        title_elem = card.select_one(selector)
        if title_elem:
            # Remove strong tags to get clean title
            for strong in title_elem.select("strong"):
                strong.extract()
            title = title_elem.get_text(strip=True)
            if title:
                package["title"] = title
                break
    
    # Extract data amount
    data_selectors = [
        "span.data-val-amount",
        ".data-val-amount",
        "span[class*='data']"
    ]
    
    for selector in data_selectors:
        data_elem = card.select_one(selector)
        if data_elem:
            package["data_amount"] = data_elem.get_text(strip=True)
            break
    
    # Extract bundle name
    bundle_selectors = [
        "h5.bundle-name",
        ".bundle-name",
        "h5"
    ]
    
    for selector in bundle_selectors:
        bundle_elem = card.select_one(selector)
        if bundle_elem:
            package["bundle_name"] = bundle_elem.get_text(strip=True)
            break
    
    # Extract price
    price_selectors = [
        "div.color-card-price",
        ".color-card-price",
        ".price",
        "div[class*='price']"
    ]
    
    for selector in price_selectors:
        price_elem = card.select_one(selector)
        if price_elem:
            price_text = price_elem.get_text(strip=True)
            # Clean up price text
            price_clean = re.sub(r'[^\d.,Rs\s/-]', '', price_text)
            package["price"] = price_clean if price_clean else price_text
            break
    
    # Try to extract speed information
    card_text = card.get_text()
    speed_match = re.search(r'(\d+)\s*(Mbps|MB|GB)', card_text, re.IGNORECASE)
    if speed_match:
        package["speed"] = speed_match.group(0)
    
    # Try to extract validity
    validity_match = re.search(r'(\d+)\s*(days?|months?|weeks?)', card_text, re.IGNORECASE)
    if validity_match:
        package["validity"] = validity_match.group(0)
    
    # Extract any additional description
    desc_selectors = [
        ".color-card-body",
        ".card-body",
        "p"
    ]
    
    for selector in desc_selectors:
        desc_elem = card.select_one(selector)
        if desc_elem:
            desc_text = desc_elem.get_text(strip=True)
            if len(desc_text) > 10 and desc_text not in [package["title"], package["bundle_name"]]:
                package["description"] = desc_text[:200] + "..." if len(desc_text) > 200 else desc_text
                break
    
    return package

def save_results(packages, filename_prefix="slt_packages"):
    """Save results to CSV and JSON files"""
    if not packages:
        logger.warning("No packages found to save")
        return
    
    # Save to CSV
    csv_filename = f"{filename_prefix}.csv"
    fieldnames = list(packages[0].keys())
    
    try:
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(packages)
        logger.info(f"✅ Saved {len(packages)} packages to {csv_filename}")
    except Exception as e:
        logger.error(f"Error saving CSV: {e}")
    
    # Save to JSON
    json_filename = f"{filename_prefix}.json"
    try:
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(packages, f, ensure_ascii=False, indent=4)
        logger.info(f"✅ Saved {len(packages)} packages to {json_filename}")
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")

def main():
    """Main function to run the scraper"""
    logger.info("Starting SLT package scraper...")
    
    results = scrape_slt_packages()
    
    if results:
        save_results(results)
        
        # Print summary
        print(f"\n📊 SCRAPING SUMMARY")
        print(f"{'='*50}")
        print(f"Total packages found: {len(results)}")
        print(f"{'='*50}")
        
        for i, pkg in enumerate(results, 1):
            print(f"\n{i}. {pkg.get('title', 'N/A')}")
            print(f"   Data: {pkg.get('data_amount', 'N/A')}")
            print(f"   Price: {pkg.get('price', 'N/A')}")
            print(f"   Speed: {pkg.get('speed', 'N/A')}")
    else:
        logger.warning("No packages were scraped. Check the website structure or selectors.")

if __name__ == "__main__":
    main()