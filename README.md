# SLT-Mobitel Assistant

This project is an **AI-powered chatbot assistant** designed to answer user queries about Sri Lanka Telecom (SLT) and Mobitel services. The assistant can provide information on broadband packages, PEO TV plans, connection charges, and even find nearby branch locations. It is built as a web-based application with a Python backend and a user-friendly HTML/CSS/JavaScript front end.

-----

### 🌟 Features

  * **Intelligent Q\&A**: Answers questions about SLT and Mobitel services, including broadband, PEO TV, and extra data packages.
  * **Hybrid LLM Integration**: Can be configured to use either a local Large Language Model (LLM) via **Ollama** or a cloud-based service like **Google Gemini**.
  * **Retrieval-Augmented Generation (RAG)**: Uses **LangChain** and **ChromaDB** as a vector store to retrieve information from scraped data, ensuring answers are grounded in specific package details and connection charges.
  * **Advanced Web Scraping**: Includes dedicated Python scripts to scrape dynamic and static content from the SLT website using both **Playwright** and **requests**.
  * **Location-Based Services**: Finds nearby SLT-Mobitel branches using the user's location, powered by `geopy`.

-----

### 📂 Project Structure

```
.
├── app.py                     # Main Flask application and chatbot logic
├── index.html                 # Front-end user interface for the chatbot
├── scrape_all_packages.py     # Script to scrape SLT package details
├── web_scraper.py             # Asynchronous web crawler to build the vector database
├── chroma_db/                 # Persistent vector store for general web content
└── packages_chroma_db/        # Persistent vector store for scraped packages
```

  * `app.py` is the primary entry point for the backend API.
  * `scrape_all_packages.py` and `web_scraper.py` are utility scripts used to prepare the necessary data before running the main application.

-----

### 🛠️ Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/thinulaH/SLT_AI_ChatBot
    cd SLT_AI_ChatBot
    ```

2.  **Install Python dependencies:**
    The project requires Python 3.7+ and several libraries.

    ```bash
    pip install Flask Flask-Cors geopy python-dotenv httpx google-generativeai langchain-community beautifulsoup4 requests playwright
    ```

    You also need to install the necessary browser binaries for Playwright:

    ```bash
    playwright install
    ```

3.  **Set up the LLM:**

      * **For Local LLM (Ollama)**: Ensure you have Ollama running locally with a compatible model.
      * **For Google Gemini**: Obtain an API key and pass it as an environment variable or modify `app.py` directly. The `gemini_api_key` is configured within the `SLTChatbot` class in `app.py`.

-----

### 🚀 Usage

#### 1\. Prepare the Data

First, you need to populate the vector databases with content from the SLT website. This is a crucial step before running the main application.

```bash
# Scrape general web content and build the main vector store
python web_scraper.py

# Scrape specific package information and build the packages vector store
python scrape_all_packages.py
```

These scripts will create the `chroma_db` and `packages_chroma_db` directories. A `data/branches.json` file is also required for branch location features.

#### 2\. Run the Application

Start the Flask server, which will launch the chatbot API.

```bash
python app.py
```

You should see log messages indicating the server is running and the vector stores have been loaded. The application runs on `localhost:5000` by default.

#### 3\. Access the Front End

Open the `index.html` file in your web browser. This is the front-end interface that will communicate with the Flask server you just started.

-----

### 💻 Tech Stack

  * **Backend**: Python, Flask
  * **Vector Database**: ChromaDB
  * **LLMs**: Ollama (local) and Google Gemini API
  * **Web Scraping**: Playwright, requests, BeautifulSoup4
  * **NLP/RAG**: LangChain, HuggingFaceEmbeddings
  * **Front End**: HTML, CSS, JavaScript

-----

### ⚙️ Configuration

Key configuration settings can be found at the top of the Python files:

  * **`app.py`**:
      * `use_local_llm`: Boolean flag to switch between Ollama and Google Gemini.
      * `gemini_api_key`: Your Google Gemini API key.
  * **`web_scraper.py`**:
      * `WEBSITE_URL`: The base URL for the scraper.
      * `CHROMA_PERSIST_DIR`: The path for the main vector store.
      * `CHUNK_SIZE`, `CHUNK_OVERLAP`: Settings for text splitting during RAG pipeline creation.
  * **`scrape_all_packages.py`**:
      * `PACKAGES_CHROMA_DIR`: The path for the packages vector store.
      * `urls`: A list of URLs to scrape, with methods specified for each.

-----

### ✉️ Contact / Author Info

  * **Authors**: Thinula Harischandra, Keshara Gunathilaka, Mevinu Gunaratne, Udantha Randil
  * **Contact**: thinula.haris@gmail.com
