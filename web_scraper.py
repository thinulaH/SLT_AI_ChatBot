import os
import asyncio
import time
from collections import deque
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from bs4 import BeautifulSoup
import re

# --- Configuration ---
WEBSITE_URL = "https://www.slt.lk"
MUST_SCRAPE_URLS = [
    "https://www.slt.lk/en/personal/broadband/fiber-unlimited",
    "https://www.slt.lk/en/personal/broadband/ftth/new-connection-charges",
    "https://www.slt.lk/en/personal/broadband/ftth/ftth-plans",
    "https://www.slt.lk/en/broadband/packages",
    "https://www.slt.lk/en/broadband/extragb",
    "https://www.slt.lk/en/broadband/Fibre-speed"
]
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CRAWL_DURATION = 60 * 20  # 10 minutes
MAX_CRAWL_DEPTH = 5

VISITED_URLS_FILE = "visited_urls.txt"

def load_visited_urls():
    if os.path.exists(VISITED_URLS_FILE):
        with open(VISITED_URLS_FILE, 'r') as f:
            return set(line.strip() for line in f.readlines())
    return set()

def save_visited_urls(urls: set):
    with open(VISITED_URLS_FILE, 'w') as f:
        for url in sorted(urls):
            f.write(url + '\n')

# --- Recursive Web Scraping with Time + Depth Limits ---
async def crawl_website(base_url: str, start_urls: list[str] = None) -> list[Document]:
    print(f"Starting to crawl website: {base_url}")
    start_time = time.time()
    
    urls_to_visit = deque([(url, 0) for url in start_urls] if start_urls else [(base_url, 0)])
    visited_urls = load_visited_urls()  # Load previous visited URLs
    documents = []

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context()
        page = await context.new_page()

        while urls_to_visit:
            elapsed_time = time.time() - start_time
            if elapsed_time > MAX_CRAWL_DURATION:
                print(f"Time limit of {MAX_CRAWL_DURATION / 60:.2f} minutes reached. Stopping crawl.")
                break

            url, depth = urls_to_visit.popleft()
            if url in visited_urls or depth > MAX_CRAWL_DEPTH:
                continue

            try:
                visited_urls.add(url)
                print(f"Navigating to (depth {depth}): {url}")
                await page.goto(url, wait_until="domcontentloaded")
                
                html_content = await page.content()
                doc = Document(
                    page_content=html_content,
                    metadata={"source": url, "title": await page.title()}
                )
                documents.append(doc)
                
                # Extract links and queue them with incremented depth
                links = await page.evaluate('''() => Array.from(document.querySelectorAll('a')).map(a => a.href)''')
                for link in links:
                    full_link = urljoin(url, link)
                    if urlparse(full_link).netloc == urlparse(base_url).netloc and full_link not in visited_urls:
                        urls_to_visit.append((full_link, depth + 1))

                print(f"Scraped: {url}")
            except Exception as e:
                print(f"Error scraping {url}: {e}")

        await browser.close()
        save_visited_urls(visited_urls)  # Save visited URLs to file
        print(f"Crawling finished. Scraped {len(documents)} documents.")
        return documents

# --- Text Splitting and Chunking with HTML Cleaning ---
def split_documents_into_chunks(documents: list[Document]) -> list[Document]:
    print("Splitting documents into chunks...")
    all_chunks = []

    def clean_text(html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove unwanted tags
        for tag in soup(["script", "style"]):
            tag.decompose()
        
        # Remove headers and footers by tag
        for tag in soup.find_all(["header", "footer"]):
            tag.decompose()
        
        # Remove common header/footer classes/IDs (site-specific tuning)
        selectors = [
            '.header', '#header', '.footer', '#footer',
            '.site-header', '.site-footer', '#siteHeader', '#siteFooter',
            '.main-header', '.main-footer'
        ]
        for selector in selectors:
            for element in soup.select(selector):
                element.decompose()

        # Extract text and clean whitespace
        text = soup.get_text()
        return re.sub(r'\s+', ' ', text).strip()

    for doc in documents:
        clean_content = clean_text(doc.page_content)
        clean_doc = Document(page_content=clean_content, metadata=doc.metadata)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        chunks = splitter.split_documents([clean_doc])
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks from clean text.")
    return all_chunks

# --- Embedding and Vector Database Storage ---
def create_vector_store(chunks: list[Document]):
    print("Creating embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print(f"Creating vector store in {CHROMA_PERSIST_DIR}...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    print("Vector database created successfully.")

# --- Main Execution ---
async def main():
    raw_docs = await crawl_website(WEBSITE_URL, start_urls=MUST_SCRAPE_URLS)
    
    if not raw_docs:
        print("No documents were scraped. Exiting.")
        return

    chunks = split_documents_into_chunks(raw_docs)
    create_vector_store(chunks)

if __name__ == "__main__":
    asyncio.run(main())

    # --- Example Retrieval ---
    print("\n--- Example Retrieval ---")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    
    query = "What is the main purpose of this website?"
    results = vector_store.similarity_search(query, k=3)
    
    print(f"Query: '{query}'")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source: {doc.metadata['source']}")
        print(f"Title: {doc.metadata.get('title', 'N/A')}")