import json
import os
import re
from pathlib import Path
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
PACKAGES_CHROMA_DIR = "./packages_chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PACKAGES_JSON_FILE = "slt_packages.json"

def load_packages_from_json(json_file: str) -> list[dict]:
    """Load packages data from JSON file."""
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found.")
        return []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        packages = json.load(f)
    
    return packages

def remove_duplicates(packages: list[dict]) -> list[dict]:
    """Remove duplicate packages."""
    seen = set()
    unique_packages = []
    
    for package in packages:
        key = (package.get('title', ''), package.get('data_amount', ''), package.get('price', ''))
        if key not in seen:
            seen.add(key)
            unique_packages.append(package)
    
    return unique_packages

def clean_and_parse_price(price_str: str) -> dict:
    """Extract monthly rental and startup fee from price string."""
    result = {"monthly": None, "startup": None, "monthly_num": 0, "startup_num": 0}
    
    if not price_str:
        return result
    
    # Extract monthly rental
    monthly_match = re.search(r'Monthly\s+Rental\s*Rs\.?\s*(\d+(?:,\d+)*)', price_str, re.IGNORECASE)
    if monthly_match:
        result["monthly"] = f"Rs.{monthly_match.group(1)}"
        result["monthly_num"] = int(monthly_match.group(1).replace(',', ''))
    
    # Extract startup fee
    startup_match = re.search(r'Startup\s+Fee\s*:?\s*Rs\.?\s*(\d+(?:,\d+)*)', price_str, re.IGNORECASE)
    if startup_match:
        result["startup"] = f"Rs.{startup_match.group(1)}"
        result["startup_num"] = int(startup_match.group(1).replace(',', ''))
    
    return result

def normalize_data_amount(data_str: str) -> dict:
    """Normalize and extract meaningful info from data amount."""
    result = {"original": data_str, "normalized": "", "is_unlimited": False, "gb_amount": 0}
    
    if not data_str:
        return result
    
    data_lower = data_str.lower()
    
    # Check for unlimited indicators
    if any(term in data_lower for term in ['unlimited', 'up', 'daily']):
        result["is_unlimited"] = True
        if 'daily' in data_lower:
            result["normalized"] = "Daily unlimited usage"
        else:
            result["normalized"] = "Unlimited usage"
    else:
        # Extract GB amount
        gb_match = re.search(r'(\d+)\s*gb', data_str, re.IGNORECASE)
        if gb_match:
            result["gb_amount"] = int(gb_match.group(1))
            result["normalized"] = f"{result['gb_amount']}GB"
        else:
            # Handle cases where it's just a number (assume GB)
            num_match = re.search(r'^(\d+)$', data_str.strip())
            if num_match:
                result["gb_amount"] = int(num_match.group(1))
                result["normalized"] = f"{result['gb_amount']}GB"
            else:
                result["normalized"] = data_str
    
    return result

def create_package_documents(packages: list[dict]) -> list[Document]:
    """Create comprehensive package documents for dedicated vector database."""
    documents = []
    unique_packages = remove_duplicates(packages)
    
    print(f"Creating documents from {len(unique_packages)} unique packages...")
    
    for i, package in enumerate(unique_packages):
        title = package.get('title', '').strip()
        bundle_name = package.get('bundle_name', '').strip()
        data_info = normalize_data_amount(package.get('data_amount', ''))
        price_info = clean_and_parse_price(package.get('price', ''))
        
        # Create multiple document types per package for maximum coverage
        
        # Document 1: Primary package information
        primary_content = f"""SLT {title} Broadband Package

Package Name: {title}
Connection Type: {bundle_name}
Data Allowance: {data_info['normalized']}
Monthly Price: {price_info.get('monthly', 'Contact SLT')}
Setup Fee: {price_info.get('startup', 'Rs.0')}

This is an SLT broadband internet package offering {data_info['normalized']} data with {bundle_name} connection technology.
Perfect for home and business internet needs in Sri Lanka.

Package Details:
- Fast and reliable {bundle_name} connection
- {'Unlimited internet usage' if data_info['is_unlimited'] else f'{data_info["gb_amount"]}GB monthly data allowance'}
- Monthly subscription: {price_info.get('monthly', 'Available on request')}
- One-time setup: {price_info.get('startup', 'Rs.0')}
- Available island-wide through SLT network
"""
        
        documents.append(Document(
            page_content=primary_content,
            metadata={
                "source": "slt_packages_database",
                "title": title,
                "type": "broadband_package",
                "subtype": "primary_info",
                "bundle_name": bundle_name,
                "connection_type": bundle_name.lower(),
                "data_amount": data_info['normalized'],
                "data_gb": data_info['gb_amount'],
                "is_unlimited": data_info['is_unlimited'],
                "monthly_price": price_info.get('monthly', ''),
                "monthly_price_num": price_info['monthly_num'],
                "startup_fee": price_info.get('startup', ''),
                "package_id": i + 1
            }
        ))
        
        # Document 2: Search-optimized content with many variations
        search_terms = []
        
        # Title variations
        search_terms.extend([
            f"SLT {title}", f"{title} package", f"{title} plan", f"{title} broadband",
            f"{title} internet", f"SLT {title} package", f"SLT {title} plan"
        ])
        
        # Connection type variations
        if bundle_name.lower() == 'fibre':
            search_terms.extend([
                "fiber broadband", "fibre broadband", "fiber internet", "fibre internet",
                "fiber optic", "FTTH", "high speed fiber", "fast internet",
                "fiber connection", "optical fiber", "fiber to home"
            ])
        elif bundle_name.lower() == '4g':
            search_terms.extend([
                "4G broadband", "mobile broadband", "wireless broadband", "LTE broadband",
                "4G internet", "mobile internet", "wireless internet", "portable internet",
                "4G connection", "LTE connection", "mobile data"
            ])
        elif bundle_name.lower() == 'adsl':
            search_terms.extend([
                "ADSL broadband", "ADSL internet", "phone line internet",
                "traditional broadband", "copper line internet", "ADSL connection"
            ])
        
        # Data-based terms
        if data_info['is_unlimited']:
            search_terms.extend([
                "unlimited internet", "unlimited broadband", "unlimited data",
                "no data limit", "unrestricted usage", "unlimited usage",
                "no cap internet", "unlimited download", "unlimited streaming"
            ])
            if 'daily' in data_info['normalized'].lower():
                search_terms.extend([
                    "daily unlimited", "daily quota", "daily reset", "daily allowance"
                ])
        else:
            if data_info['gb_amount'] > 0:
                gb = data_info['gb_amount']
                search_terms.extend([
                    f"{gb}GB package", f"{gb}GB broadband", f"{gb}GB internet",
                    f"{gb} gigabyte", f"{gb}GB plan", f"{gb}GB data",
                    f"{gb}GB monthly", f"{gb}GB allowance"
                ])
        
        # Price-based terms
        if price_info['monthly_num'] > 0:
            monthly = price_info['monthly_num']
            search_terms.extend([
                f"Rs {monthly} package", f"package for Rs {monthly}", f"broadband Rs {monthly}",
                f"internet Rs {monthly}", f"Rs.{monthly} monthly", f"rupees {monthly}"
            ])
            
            # Price categories
            if monthly < 2000:
                search_terms.extend([
                    "cheap broadband", "budget internet", "affordable broadband",
                    "low cost internet", "economical package", "inexpensive internet"
                ])
            elif monthly < 5000:
                search_terms.extend([
                    "mid range broadband", "moderate price internet", "reasonable broadband",
                    "affordable high speed", "mid price package"
                ])
            else:
                search_terms.extend([
                    "premium broadband", "high end internet", "professional broadband",
                    "business grade internet", "premium package"
                ])
        
        search_content = f"""SLT Broadband Package Search: {title}

Common Search Terms: {' | '.join(search_terms[:15])}

Package Overview:
- Package: {title}
- Type: {bundle_name} broadband connection
- Data: {data_info['normalized']}
- Price: {price_info.get('monthly', 'Contact SLT')}

Perfect for customers looking for:
{' '.join(search_terms[:20])}

This {bundle_name} package offers {data_info['normalized']} internet connectivity for {price_info.get('monthly', 'competitive pricing')}.
Suitable for home users, businesses, and anyone needing reliable internet in Sri Lanka.
"""
        
        documents.append(Document(
            page_content=search_content,
            metadata={
                "source": "slt_packages_database",
                "title": title,
                "type": "broadband_package", 
                "subtype": "search_optimized",
                "bundle_name": bundle_name,
                "connection_type": bundle_name.lower(),
                "data_amount": data_info['normalized'],
                "data_gb": data_info['gb_amount'],
                "is_unlimited": data_info['is_unlimited'],
                "monthly_price": price_info.get('monthly', ''),
                "monthly_price_num": price_info['monthly_num'],
                "startup_fee": price_info.get('startup', ''),
                "package_id": i + 1
            }
        ))
        
        # Document 3: FAQ and conversational queries
        faq_content = f"""SLT {title} Package - Questions and Answers

What is the SLT {title} package?
The {title} is a {bundle_name} broadband internet package that provides {data_info['normalized']} data allowance for home and business use.

How much does the {title} package cost?
The {title} package costs {price_info.get('monthly', 'contact SLT for current pricing')} per month with a setup fee of {price_info.get('startup', 'Rs.0')}.

What type of connection does {title} use?
{title} uses {bundle_name} connection technology {'which provides high-speed fiber optic internet' if bundle_name.lower() == 'fibre' else 'for reliable internet access'}.

Is {title} good for home use?
Yes, {title} is excellent for home internet with {data_info['normalized']} {'allowing unlimited browsing, streaming, and downloading' if data_info['is_unlimited'] else 'providing ample data for typical home usage'}.

Can I upgrade or downgrade from {title}?
You can contact SLT customer service to discuss changing your package from {title} to other available options.

How do I apply for {title}?
Contact SLT at 1212 or visit any SLT branch to apply for the {title} package. Online applications may also be available.

What areas can get {title}?
{title} availability depends on {bundle_name} network coverage in your area. Contact SLT to check availability at your location.
"""
        
        documents.append(Document(
            page_content=faq_content,
            metadata={
                "source": "slt_packages_database",
                "title": title,
                "type": "broadband_package",
                "subtype": "faq",
                "bundle_name": bundle_name,
                "connection_type": bundle_name.lower(),
                "data_amount": data_info['normalized'],
                "data_gb": data_info['gb_amount'],
                "is_unlimited": data_info['is_unlimited'],
                "monthly_price": price_info.get('monthly', ''),
                "monthly_price_num": price_info['monthly_num'],
                "startup_fee": price_info.get('startup', ''),
                "package_id": i + 1
            }
        ))
    
    print(f"Created {len(documents)} package documents ({len(documents)//len(unique_packages)} per package)")
    return documents

def create_packages_vector_database(documents: list[Document]):
    """Create a dedicated vector database for SLT packages."""
    
    # Remove existing packages database if it exists
    if Path(PACKAGES_CHROMA_DIR).exists():
        import shutil
        print(f"Removing existing packages database at {PACKAGES_CHROMA_DIR}")
        shutil.rmtree(PACKAGES_CHROMA_DIR)
    
    print(f"Creating new packages vector database at {PACKAGES_CHROMA_DIR}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Create new vector database specifically for packages
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PACKAGES_CHROMA_DIR
    )
    
    print(f"Packages vector database created with {len(documents)} documents")
    return vector_store

def test_packages_database(vector_store):
    """Test the packages database with comprehensive queries."""
    print("\n" + "="*60)
    print("TESTING DEDICATED PACKAGES DATABASE")
    print("="*60)
    
    test_queries = [
        ("unlimited broadband packages", "unlimited packages"),
        ("fiber internet under 5000", "affordable fiber plans"),
        ("4G mobile broadband", "4G packages"),
        ("100GB package", "specific data amounts"),
        ("cheap internet under 2000", "budget packages"),
        ("SLT Trio Vibe", "specific package name"),
        ("ADSL broadband", "ADSL packages"),
        ("daily unlimited usage", "daily unlimited plans"),
        ("Rs 3530 package", "exact price match"),
        ("fibre package", "fiber packages"),
        ("startup fee", "setup costs"),
        ("monthly rental", "subscription pricing"),
    ]
    
    successful_tests = 0
    total_tests = len(test_queries)
    
    for query, description in test_queries:
        print(f"\nQuery: '{query}' (Looking for: {description})")
        results = vector_store.similarity_search(query, k=3)
        
        if results:
            print(f"  Found {len(results)} results:")
            for i, doc in enumerate(results[:3]):
                title = doc.metadata.get('title', 'Unknown')
                connection = doc.metadata.get('connection_type', 'N/A')
                data = doc.metadata.get('data_amount', 'N/A') 
                price = doc.metadata.get('monthly_price', 'N/A')
                subtype = doc.metadata.get('subtype', 'N/A')
                
                print(f"    {i+1}. {title} ({connection}) - {data} - {price} [{subtype}]")
            
            successful_tests += 1
        else:
            print("    No results found")
    
    print(f"\n" + "="*60)
    print(f"SUCCESS RATE: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
    print("="*60)

def main():
    """Main function to create dedicated packages vector database."""
    print("Creating dedicated SLT packages vector database...")
    
    # Load packages
    packages = load_packages_from_json(PACKAGES_JSON_FILE)
    if not packages:
        print("No packages found. Exiting.")
        return
    
    print(f"Loaded {len(packages)} packages from JSON")
    
    # Create comprehensive package documents
    documents = create_package_documents(packages)
    
    # Create dedicated packages vector database
    vector_store = create_packages_vector_database(documents)
    
    # Test the database
    test_packages_database(vector_store)
    
    print(f"\nPackages database created successfully!")
    print(f"Database location: {PACKAGES_CHROMA_DIR}")
    print(f"Total documents: {len(documents)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

if __name__ == "__main__":
    main()