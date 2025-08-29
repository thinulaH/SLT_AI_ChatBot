import re
import json
import os
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from datetime import datetime
import logging

# Langchain for vector store
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Ollama client (local LLM)
import httpx

# Google Gemini
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__)
CORS(app)

class SLTChatbot:
    def __init__(self, use_local_llm=True, gemini_api_key="AIzaSyCjS3Uj_ZdQX4TnSjx1CmCPMkLsc4sM0_4"):
        self.use_local_llm = use_local_llm
        self.gemini_api_key = gemini_api_key
        self.gemini_model = None
        self.vector_store = None
        self.packages_vector_store = None  # New: Dedicated packages vector store
        self.embeddings = None
        self.branches = []
        self.city_names = set()
        self.vector_store_path = None
        self.packages_vector_store_path = "./packages_chroma_db"  # New: Path to packages vector store
        self.initialize()
        self.sessions = {}  # Store user session histories

    def add_to_session(self, user_id, role, message, max_history=10):
        if user_id not in self.sessions:
            self.sessions[user_id] = []
        self.sessions[user_id].append({"role": role, "content": message})
        # Limit history length
        if len(self.sessions[user_id]) > max_history:
            self.sessions[user_id] = self.sessions[user_id][-max_history:]

    def get_session_history(self, user_id):
        return self.sessions.get(user_id, [])

    def initialize(self):
        logger.info("🔄 Initializing SLT Chatbot...")
        self.init_vector_db()
        self.load_branches()
        self.setup_llm()
        logger.info("✅ SLT Chatbot initialized successfully")

    def setup_llm(self):
        """Initialize the selected LLM (Local Ollama or Google Gemini)"""
        # Unchanged from original
        if self.use_local_llm:
            logger.info("🤖 Using Local LLM (Ollama)")
            try:
                response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Local LLM (Ollama) connection successful")
                else:
                    logger.warning("⚠️ Local LLM (Ollama) might not be running")
            except Exception as e:
                logger.warning(f"⚠️ Could not connect to local LLM: {e}")
        else:
            logger.info("🌟 Using Google Gemini")
            if not self.gemini_api_key:
                logger.error("❌ Gemini API key not provided")
                raise ValueError("Gemini API key is required when use_local_llm=False")
            
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
                logger.info("✅ Google Gemini initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Google Gemini: {e}")
                raise

    def init_vector_db(self):
        """Initialize both general and packages vector databases"""
        try:
            # Initialize general vector database
            db_paths = [
                "./chroma_db",
                "./slt_chroma_db",
                "./slt_vector_db",
                "../chroma_db"
            ]
            for db_path in db_paths:
                if Path(db_path).exists():
                    logger.info(f"🔍 Found general Chroma DB at: {db_path}")
                    self.vector_store_path = db_path
                    break

            if not self.vector_store_path:
                logger.warning("⚠️ No existing general Chroma DB found, creating new at ./chroma_db")
                self.vector_store_path = "./chroma_db"

            self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
            self.vector_store = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings
            )
            general_doc_count = self.vector_store._collection.count()
            logger.info(f"✅ General vector store loaded with {general_doc_count} documents")

            # Initialize packages vector database
            if Path(self.packages_vector_store_path).exists():
                logger.info(f"🔍 Found packages Chroma DB at: {self.packages_vector_store_path}")
                self.packages_vector_store = Chroma(
                    persist_directory=self.packages_vector_store_path,
                    embedding_function=self.embeddings
                )
                packages_doc_count = self.packages_vector_store._collection.count()
                logger.info(f"✅ Packages vector store loaded with {packages_doc_count} documents")
            else:
                logger.warning(f"⚠️ Packages vector store not found at {self.packages_vector_store_path}")
                self.packages_vector_store = None

        except Exception as e:
            logger.error(f"❌ Failed to initialize vector stores: {e}")
            self.vector_store = None
            self.packages_vector_store = None

    def load_branches(self):
        # Unchanged from original
        branch_files = [
            "data/branches.json",
            "./branches.json",
            "../data/branches.json"
        ]
        for file_path in branch_files:
            try:
                if os.path.exists(file_path):
                    with open(file_path, encoding="utf8") as f:
                        self.branches = json.load(f)
                        self.city_names = {branch["name"].lower() for branch in self.branches}
                        self.city_names.add("colombo")
                        self.city_names.add("kandy")
                        self.city_names.add("galle")
                        self.city_names.add("negombo")
                    logger.info(f"✅ Loaded {len(self.branches)} branches from {file_path}")
                    return
            except Exception as e:
                logger.warning(f"⚠️ Could not load {file_path}: {e}")
        logger.warning("⚠️ No branch data found. Location features will be limited.")
        self.branches = []

    def is_package_query(self, query):
        """Determine if the query is related to broadband packages, excluding PEO TV and extra GB queries"""
        package_keywords = [
            "package", "plan", "broadband", "internet", "data",
            "monthly", "unlimited", "gb" ]
        peo_tv_keywords = ["peo tv", "peotv", "tv", "television", "channel", "channels"]
        extra_gb_keywords = ["extra gb", "additional data", "add-on data", "more data", 
                            "extra data", "data booster", "data add-on", "additional gb"]
        query_lower = query.lower()
        
        # Check if the query contains PEO TV or extra GB-related keywords
        is_peo_tv_query = any(keyword in query_lower for keyword in peo_tv_keywords)
        is_extra_gb_query = any(keyword in query_lower for keyword in extra_gb_keywords)
        
        # Return True only for broadband package queries (not PEO TV or extra GB)
        return (any(keyword in query_lower for keyword in package_keywords) and 
                not is_peo_tv_query and 
                not is_extra_gb_query)

    def is_peo_tv_query(self, query):
        """Determine if the query is specifically about PEO TV"""
        peo_tv_keywords = ["peo tv", "peotv", "peo", "tv packages", "television packages", 
                          "tv plans", "television plans", "channels", "peo channels"]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in peo_tv_keywords)

    def is_non_broadband_query(self, query):
        """Determine if query is about services other than broadband packages"""
        return self.is_peo_tv_query(query) or any(keyword in query.lower() for keyword in [
            "extra gb", "additional data", "add-on data", "branch", "location", "office",
            "customer service", "bill", "payment", "support", "contact"
        ])

    def is_list_query(self, query):
        """Determine if the query is asking for a comprehensive list"""
        list_indicators = [
            "list", "all packages", "all plans", "show me", "what packages", "what plans",
            "available packages", "available plans", "package options", "plan options",
            "tell me about", "give me", "show all", "display all", "complete list",
            "full list", "entire list", "overview of", "summary of"
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in list_indicators)

    def preprocess_query(self, query):
        # Unchanged from original, but kept here for reference
        query = query.strip()
        if len(query.split()) < 3:
            query = query + " SLT broadband packages services"
        expansions = {
            "peo": "PEO TV television",
            "adsl": "ADSL broadband internet",
            "fiber": "fiber broadband internet",
            "wifi": "WiFi internet broadband",
            "tv": "television PEO TV"
        }
        query_lower = query.lower()
        for abbrev, expansion in expansions.items():
            if abbrev in query_lower:
                query += f" {expansion}"
        return query

    def find_relevant_chunks(self, query, top_n=3):
        """
        Search both vector databases with better logic to prevent irrelevant mixing.
        Prioritizes the correct database based on query type.
        """
        if not self.vector_store:
            logger.warning("⚠️ General vector store not initialized")
            return []

        processed_query = self.preprocess_query(query)
        logger.info(f"🔍 Searching for: {processed_query}")

        # Check if this is a list query - if so, get more results
        is_list_query = self.is_list_query(query)
        if is_list_query:
            top_n = max(top_n, 6)  # Increase results for list queries
            logger.info(f"📋 List query detected, expanding results to {top_n}")

        matched_chunks = []
        try:
            # Classify query types
            is_fiber_query = "fibre" in query.lower() or "fiber" in query.lower()
            is_package_query = self.is_package_query(query)
            is_peo_tv_query = self.is_peo_tv_query(query)
            is_non_broadband = self.is_non_broadband_query(query)

            if is_peo_tv_query:
                # PEO TV query: ONLY search general database, NO packages database
                logger.info("📺 PEO TV query detected, searching ONLY general database")
                general_results = self.vector_store.similarity_search_with_score(processed_query, k=top_n)
                for doc, score in general_results:
                    matched_chunks.append({
                        "title": doc.metadata.get("title", "Unknown Title"),
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                        "score": score,
                        "db_source": "general"
                    })
                logger.info(f"✅ Found {len(general_results)} PEO TV-specific chunks")

            elif is_fiber_query and self.packages_vector_store and not is_non_broadband:
                logger.info("🔍 Fibre broadband query detected, prioritizing packages vector store")
                
                # Get Fibre-specific documents from the packages DB
                search_k = top_n + 2 if is_list_query else top_n
                package_results = self.packages_vector_store.similarity_search_with_score(processed_query, k=search_k)
                for doc, score in package_results:
                    if "fibre" in doc.page_content.lower() or "fiber" in doc.page_content.lower():
                        matched_chunks.append({
                            "title": doc.metadata.get("title", "Unknown Title"),
                            "content": doc.page_content,
                            "source": "https://www.slt.lk/en/broadband/packages",
                            "score": score,
                            "db_source": "packages"
                        })
                
                # Get supplementary context from general DB only if needed and not enough results
                if len(matched_chunks) < 2:
                    logger.info("🔍 Getting minimal supplementary context from general database")
                    general_results = self.vector_store.similarity_search_with_score(processed_query, k=1)
                    for doc, score in general_results:
                        matched_chunks.append({
                            "title": doc.metadata.get("title", "Unknown Title"),
                            "content": doc.page_content,
                            "source": doc.metadata.get("source", ""),
                            "score": score,
                            "db_source": "general"
                        })

            elif is_package_query and self.packages_vector_store and not is_non_broadband:
                # Broadband package query: Prioritize packages DB with minimal general supplementation
                logger.info("🔍 Broadband package query detected, prioritizing packages database")
                
                # For list queries, get more package results, otherwise be conservative
                if is_list_query:
                    packages_slots = top_n  # Use most slots for packages in list queries
                    general_slots = 1 if len(matched_chunks) < 3 else 0  # Minimal general context
                else:
                    packages_slots = max(2, top_n - 1)  # Leave room for 1 general
                    general_slots = 1
                
                package_results = self.packages_vector_store.similarity_search_with_score(processed_query, k=packages_slots)
                for doc, score in package_results:
                    matched_chunks.append({
                        "title": doc.metadata.get("title", "Unknown Title"),
                        "content": doc.page_content,
                        "source": "https://www.slt.lk/en/broadband/packages",
                        "score": score,
                        "db_source": "packages"
                    })
                logger.info(f"✅ Found {len(package_results)} package-specific chunks")

                # Add minimal general context only if we have space and it's not a pure list query
                if general_slots > 0 and len(matched_chunks) < top_n and not is_list_query:
                    logger.info("🔍 Adding minimal general context")
                    general_results = self.vector_store.similarity_search_with_score(processed_query, k=general_slots)
                    for doc, score in general_results:
                        matched_chunks.append({
                            "title": doc.metadata.get("title", "Unknown Title"),
                            "content": doc.page_content,
                            "source": doc.metadata.get("source", ""),
                            "score": score,
                            "db_source": "general"
                        })

            else:
                # Non-package/Non-PEO query OR general query: Prioritize general DB
                logger.info("🔍 General/non-package query, searching general database primarily")
                
                general_results = self.vector_store.similarity_search_with_score(processed_query, k=top_n)
                for doc, score in general_results:
                    matched_chunks.append({
                        "title": doc.metadata.get("title", "Unknown Title"),
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                        "score": score,
                        "db_source": "general"
                    })
                logger.info(f"✅ Found {len(general_results)} general chunks")

                # Only add packages if we have very few results and it might be relevant
                if len(matched_chunks) < 2 and self.packages_vector_store and not is_non_broadband:
                    logger.info("🔍 Adding minimal package supplementation")
                    package_results = self.packages_vector_store.similarity_search_with_score(processed_query, k=1)
                    for doc, score in package_results:
                        matched_chunks.append({
                            "title": doc.metadata.get("title", "Unknown Title"),
                            "content": doc.page_content,
                            "source": "https://www.slt.lk/en/broadband/packages",
                            "score": score,
                            "db_source": "packages"
                        })

            # Deduplicate chunks based on content and sort by relevance
            seen_content = set()
            unique_chunks = []
            for chunk in matched_chunks:
                if chunk["content"] not in seen_content:
                    seen_content.add(chunk["content"])
                    unique_chunks.append(chunk)

            # Sort by similarity score (lower is better for better relevance)
            # But prioritize by database source for package queries
            if is_package_query and not is_non_broadband:
                # For package queries, prioritize packages DB results, then by score
                unique_chunks = sorted(unique_chunks, key=lambda x: (
                    0 if x["db_source"] == "packages" else 1,  # Packages first
                    x["score"]  # Then by similarity score
                ))
            else:
                # For non-package queries, just sort by score
                unique_chunks = sorted(unique_chunks, key=lambda x: x["score"])

            # For list queries, be more generous with results, but respect query type
            if is_list_query and (is_package_query or is_peo_tv_query):
                final_limit = min(len(unique_chunks), 8)
            else:
                final_limit = top_n
            
            unique_chunks = unique_chunks[:final_limit]

            # Remove internal metadata from final output
            for chunk in unique_chunks:
                chunk.pop("score", None)
                chunk.pop("db_source", None)

            logger.info(f"✅ Final count: {len(unique_chunks)} unique matching chunks")
            
            # Log the distribution for debugging
            if unique_chunks and matched_chunks:
                packages_count = sum(1 for chunk in matched_chunks if chunk.get("db_source") == "packages")
                general_count = sum(1 for chunk in matched_chunks if chunk.get("db_source") == "general")
                logger.info(f"📊 Result distribution: {packages_count} packages, {general_count} general")
            
            return unique_chunks

        except Exception as e:
            logger.error(f"❌ Error searching vector stores: {e}")
            return []

    def find_nearest_branches(self, user_coords, top_n=3):
        # Unchanged from original
        if not self.branches:
            return []
        distances = []
        for branch in self.branches:
            try:
                branch_coords = (branch["latitude"], branch["longitude"])
                dist = geodesic(user_coords, branch_coords).km
                distances.append((branch, dist))
            except (KeyError, ValueError):
                continue
        return sorted(distances, key=lambda x: x[1])[:top_n]

    def format_branch(self, branch, dist_km):
        # Unchanged from original
        lines = [f"📍 **{branch['name']}** – {dist_km:.1f} km away"]
        if branch.get("address"):
            lines.append(f"🏠 **Address:** {branch['address']}")
        if branch.get("phone"):
            lines.append(f"📞 **Phone:** {branch['phone']}")
        if branch.get("email"):
            lines.append(f"📧 **Email:** {branch['email']}")
        if branch.get("hours"):
            lines.append(f"🕒 **Hours:** {branch['hours']}")
        return "\n".join(lines)

    def handle_location_query(self, user_input, user_coords=None):
        # Unchanged from original
        try:
            geolocator = Nominatim(user_agent="slt-location-finder")
            
            if user_coords:
                logger.info(f"🌍 Using provided coordinates: {user_coords}")
                try:
                    location = geolocator.reverse(user_coords, language='en')
                    location_name = location.address if location else f"Coordinates: {user_coords[0]:.4f}, {user_coords[1]:.4f}"
                except:
                    location_name = f"Coordinates: {user_coords[0]:.4f}, {user_coords[1]:.4f}"
            
                nearest_branches = self.find_nearest_branches(user_coords)
                
                response_parts = [
                    f"📌 **Your Location:** {location_name}",
                    ""
                ]
                
                if nearest_branches:
                    response_parts.append("🏢 **Nearest SLT Branches:**")
                    response_parts.append("")
                    for branch, dist in nearest_branches:
                        response_parts.append(self.format_branch(branch, dist))
                        response_parts.append("")
                    response_parts.append("🔗 **More Info:** https://www.slt.lk/en/contact-us/branch-locator/our-locations/our-network")
                else:
                    response_parts.append("❌ No branch data available nearby.")
                    response_parts.append("🔗 **Find branches:** https://www.slt.lk/en/contact-us/branch-locator/our-locations/our-network")
                
                return "\n".join(response_parts)
            
            # Text-based location search
            location_query = f"{user_input}, Sri Lanka"
            location = geolocator.geocode(location_query, timeout=5)
            
            if not location:
                return "❌ Sorry, I couldn't find that location. Please try with a specific city or town name.\n\n**Examples:**\n• Colombo\n• Kandy\n• Galle\n• Negombo"
            
            user_coords = (location.latitude, location.longitude)
            nearest_branches = self.find_nearest_branches(user_coords)
            
            response_parts = [
                f"📌 **Location:** {location.address}",
                ""
            ]
            
            if nearest_branches:
                response_parts.append("🏢 **Nearest SLT Branches:**")
                response_parts.append("")
                for branch, dist in nearest_branches:
                    response_parts.append(self.format_branch(branch, dist))
                    response_parts.append("")
                response_parts.append("🔗 **More Info:** https://www.slt.lk/en/contact-us/branch-locator/our-locations/our-network")
            else:
                response_parts.append("❌ No branch data available nearby.")
                response_parts.append("🔗 **Find branches:** https://www.slt.lk/en/contact-us/branch-locator/our-locations/our-network")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"❌ Location query error: {e}")
            return f"❌ Sorry, I encountered an error processing your location request. Please try again or visit the SLT website."

    def _append_source_links(self, response_text, chunks):
        # Unchanged from original
        source_links = {chunk['source'] for chunk in chunks if chunk.get('source')}
        if not source_links:
            return response_text
        
        response_text += "\n\n🔗 **Sources:**\n"
        for link in source_links:
            response_text += f"- {link}\n"
        return response_text.strip()

    def query_llm(self, user_query, context_chunks, user_id):
        """Enhanced LLM query with better handling for list queries"""
        logger.info(f"🤖 Using {'Local LLM' if self.use_local_llm else 'Google Gemini'}")
        history = self.get_session_history(user_id)
        history_text = "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in history if h['role'] != "system"])
        
        # Check if this is a list query
        is_list_query = self.is_list_query(user_query)
        
        if self.use_local_llm:
            context_blocks = []
            for chunk in context_chunks:
                url = chunk['source']
                title = chunk.get('title', 'No Title')
                content = chunk['content'][:1500]
                block = f"Page: {url}\nTitle: {title}\nContent:\n{content}"
                context_blocks.append(block)
            full_context = "\n\n---\n\n".join(context_blocks)
            
            # Enhanced system prompt for list queries
            if is_list_query:
                system_prompt = (
                    "You are a helpful assistant answering questions about SLT (Sri Lanka Telecom) broadband, PEO TV, branches, and services.\n"
                    "The user is asking for a comprehensive list or overview. Provide detailed information about ALL the packages/options mentioned in the context.\n"
                    "Format your response as a complete list with clear details for each item.\n"
                    "Do not limit yourself to just 1-2 examples - include ALL relevant packages from the provided context.\n\n"
                    f"Conversation history:\n{history_text}\n\n"
                    f"User question: {user_query}\n\n"
                    f"Based on the following extracted content from the SLT website:\n\n{full_context}\n\n"
                    "Answer clearly and comprehensively. List ALL packages/plans mentioned in the context with their details. if ask only for names just give names only"
                    "Do not provide links in your answer, as they will be added automatically."
                )
            else:
                system_prompt = (
                    "You are a helpful assistant answering questions about SLT (Sri Lanka Telecom) broadband, PEO TV, branches, and services.\n\n"
                    f"Conversation history:\n{history_text}\n\n"
                    f"User question: {user_query}\n\n"
                    f"Based on the following extracted content from the SLT website:\n\n{full_context}\n\n"
                    "Answer clearly and helpfully. Do not provide links in your answer(if specially ask for a link just give the link), as they will be added automatically."
                )
            
            try:
                response = httpx.post(
                    "http://127.0.0.1:11434/api/chat",
                    json={
                        "model": "gemma3:4b",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_query}
                        ],
                        "stream": True
                    },
                    timeout=90 if is_list_query else 60  # More time for list queries
                )
                reply_text = "".join(json.loads(line)['message']['content'] for line in response.iter_lines() if line.strip())
            except Exception as e:
                logger.error(f"❌ Local LLM query failed: {e}")
                reply_text = "❌ I couldn't generate a detailed response. Please visit the SLT website for more info."
        else:
            context_blocks = []
            for chunk in context_chunks:
                url = chunk['source']
                title = chunk.get('title', 'No Title')
                content = chunk['content'][:1500]
                block = f"Page: {url}\nTitle: {title}\nContent:\n{content}"
                context_blocks.append(block)
            full_context = "\n\n---\n\n".join(context_blocks)
            
            # Enhanced prompt for Gemini with list queries
            if is_list_query:
                prompt = (
                    "You are a helpful assistant answering questions about SLT (Sri Lanka Telecom) broadband, PEO TV, branches, and services.\n"
                    "The user is asking for a comprehensive list or overview. Provide detailed information about ALL the packages/options mentioned in the context.\n"
                    "Format your response as a complete list with clear details for each item.\n"
                    "Do not limit yourself to just 1-2 examples - include ALL relevant packages from the provided context.\n\n"
                    f"Conversation history:\n{history_text}\n\n"
                    f"User question: {user_query}\n\n"
                    f"Based on the following extracted content from the SLT website:\n\n{full_context}\n\n"
                    "Answer clearly and comprehensively. List ALL packages/plans mentioned in the context with their details. "
                    "Do not provide links in your answer, as they will be added automatically."
                )
            else:
                prompt = (
                    "You are a helpful assistant answering questions about SLT (Sri Lanka Telecom) broadband, PEO TV, branches, and services.\n\n"
                    f"Conversation history:\n{history_text}\n\n"
                    f"User question: {user_query}\n\n"
                    f"Based on the following extracted content from the SLT website:\n\n{full_context}\n\n"
                    "Answer clearly and helpfully. Do not provide links in your answer, as they will be added automatically."
                )
            
            try:
                response = self.gemini_model.generate_content(prompt)
                reply_text = response.text.strip()
            except Exception as e:
                logger.error(f"❌ Gemini LLM query failed: {e}")
                reply_text = "❌ I couldn't generate a detailed response. Please visit the SLT website for more info."
        
        self.add_to_session(user_id, "assistant", reply_text)
        return self._append_source_links(reply_text, context_chunks)

    def generate_fallback_response(self, user_input, chunks):
        # Unchanged from original
        if not chunks:
            return "❌ I couldn't find specific information about that. Please visit https://www.slt.lk or call customer service at 1212."
        response_parts = ["📋 **Here's what I found:**", ""]
        for i, chunk in enumerate(chunks[:2], 1):
            response_parts.append(f"**{i}. {chunk['title']}**")
            snippet_lines = chunk['content'].split('\n')[:3]
            for line in snippet_lines:
                if line.strip():
                    response_parts.append(f"• {line.strip()}")
            response_parts.append("")
        
        return self._append_source_links("\n".join(response_parts), chunks)

    def switch_llm(self, use_local_llm, gemini_api_key=None):
        # Unchanged from original
        logger.info(f"🔄 Switching to {'Local LLM' if use_local_llm else 'Google Gemini'}")
        self.use_local_llm = use_local_llm
        if not use_local_llm and gemini_api_key:
            self.gemini_api_key = gemini_api_key
        self.setup_llm()

# Configuration
USE_LOCAL_LLM = os.environ.get("USE_LOCAL_LLM", "false").lower() == "true"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCjS3Uj_ZdQX4TnSjx1CmCPMkLsc4sM0_4")

# Initialize chatbot
chatbot = SLTChatbot(
    use_local_llm=USE_LOCAL_LLM,
    gemini_api_key=GEMINI_API_KEY
)

# === API Endpoints ===
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "✅ SLT Chatbot API is running",
        "timestamp": datetime.now().isoformat(),
        "branches_loaded": len(chatbot.branches),
        "vector_db_documents": (
            (chatbot.vector_store._collection.count() if chatbot.vector_store else 0) +
            (chatbot.packages_vector_store._collection.count() if chatbot.packages_vector_store else 0)
        ),
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_type": "Local LLM (Ollama)" if chatbot.use_local_llm else "Google Gemini",
        "gemini_configured": chatbot.gemini_api_key is not None
    })

@app.route("/switch-llm", methods=["POST"])
def switch_llm():
    try:
        data = request.get_json()
        use_local_llm = data.get("use_local_llm", True)
        gemini_api_key = data.get("gemini_api_key", None)
    
        if not use_local_llm and not gemini_api_key and not chatbot.gemini_api_key:
            return jsonify({
                "error": "❌ Gemini API key is required when switching to Google Gemini"
            }), 400
    
        chatbot.switch_llm(use_local_llm, gemini_api_key)
    
        return jsonify({
            "status": "✅ LLM switched successfully",
            "current_llm": "Local LLM (Ollama)" if chatbot.use_local_llm else "Google Gemini"
        })
    
    except Exception as e:
        logger.error(f"❌ Error switching LLM: {e}")
        return jsonify({
            "error": f"❌ Failed to switch LLM: {str(e)}"
        }), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "❌ No JSON data provided"}), 400
        user_input = data.get("message", "").strip()
        user_id = data.get("user_id", "default")
        if not user_input:
            return jsonify({"error": "❌ Empty message provided"}), 400

        logger.info(f"💬 User query: {user_input}")
        user_lower = user_input.lower()

        # Store user input in session
        chatbot.add_to_session(user_id, "user", user_input)

        # Handle casual replies first
        casual_replies = {
            "hello": "👋 Hello! I'm your SLT assistant. How can I help you today?",
            "hi": "Hi there! 😊 Ask me about SLT broadband packages, PEO TV, or branch locations.",
            "hey": "Hey! 👋 What would you like to know about SLT services?",
            "thanks": "🙏 You're welcome! Anything else I can help with?",
            "thank you": "Happy to help! 😊 Is there anything else about SLT services you'd like to know?",
            "bye": "👋 Goodbye! Have a great day!",
            "goodbye": "👋 Take care! Feel free to come back if you have more questions."
        }
        if user_lower in casual_replies:
            return jsonify({"reply": casual_replies[user_lower]})

        # Handle location queries
        location_keywords = ["branch", "location", "office", "near", "address", "where", "closest", "nearby"]
        user_words = user_lower.split()
        is_location_query = any(word in user_words for word in location_keywords)
        found_city = next((city for city in chatbot.city_names if city in user_lower), None)

        if is_location_query:
            if found_city:
                logger.info(f"📍 Specific city query detected: {found_city}")
                response = chatbot.handle_location_query(found_city)
                return jsonify({"reply": response})
            elif any(phrase in user_lower for phrase in ["near me", "my location", "closest", "nearby"]):
                return jsonify({
                    "reply": "📍 To find the nearest SLT branches, I need your location.\n\n**Options:**\n1️⃣ **Share your location** (most accurate)\n2️⃣ **Tell me your city/area** (e.g., 'Colombo', 'Kandy', 'Galle')\n\n🔒 *Your location data is only used to find nearby branches and is not stored.*",
                    "request_location": True
                })

        # Handle package and other queries using the appropriate vector store
        chunks = chatbot.find_relevant_chunks(user_input, top_n=3)
        if not chunks:
            return jsonify({
                "reply": "❌ I couldn't find specific information about that topic. \n\n🔗 **Try visiting:**\n- https://www.slt.lk/en/broadband/packages\n- https://www.slt.lk/en/peo-tv/packages\n- **Customer Service:** 1212"
            })

        llm_response = chatbot.query_llm(user_input, chunks, user_id)
        return jsonify({
            "reply": llm_response,
            "llm_used": "Local LLM (Ollama)" if chatbot.use_local_llm else "Google Gemini"
        })

    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        return jsonify({
            "error": "❌ Sorry, I encountered an error. Please try again.",
            "details": str(e) if app.debug else None
        }), 500

@app.route("/location", methods=["POST"])
def handle_location():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "❌ No JSON data provided"}), 400
    
        latitude = data.get("latitude")
        longitude = data.get("longitude")
    
        if not latitude or not longitude:
            return jsonify({"error": "❌ Latitude and longitude are required"}), 400
    
        try:
            lat = float(latitude)
            lng = float(longitude)
            user_coords = (lat, lng)
        except (ValueError, TypeError):
            return jsonify({"error": "❌ Invalid coordinates provided"}), 400
    
        if not (5.5 <= lat <= 10.0 and 79.0 <= lng <= 82.0):
            return jsonify({
                "error": "❌ Location appears to be outside Sri Lanka. Please check your coordinates or enter your city manually."
            }), 400
    
        logger.info(f"📍 Processing location request: {lat}, {lng}")
    
        response = chatbot.handle_location_query("", user_coords=user_coords)
        return jsonify({"reply": response})
    
    except Exception as e:
        logger.error(f"❌ Location processing error: {e}")
        return jsonify({
            "error": "❌ Sorry, I encountered an error processing your location. Please try entering your city manually.",
            "details": str(e) if app.debug else None
        }), 500

@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        top_n = data.get("top_n", 3)
        if not query:
            return jsonify({"error": "No query provided"}), 400
        chunks = chatbot.find_relevant_chunks(query, top_n=top_n)
        return jsonify({
            "query": query,
            "results": chunks,
            "total_found": len(chunks)
        })
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "❌ Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "❌ Internal server error"}), 500

# Start app
if __name__ == "__main__":
    logger.info("🚀 Starting SLT Chatbot API server (local mode)...")
    logger.info(f"📊 General vector store: {chatbot.vector_store._collection.count() if chatbot.vector_store else 0} documents loaded")
    logger.info(f"📊 Packages vector store: {chatbot.packages_vector_store._collection.count() if chatbot.packages_vector_store else 0} documents loaded")
    logger.info(f"🌍 Branch data: {len(chatbot.branches)} locations loaded")
    logger.info(f"🤖 LLM: {'Local LLM (Ollama)' if chatbot.use_local_llm else 'Google Gemini'}")
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 4321)),
        debug=os.environ.get("DEBUG", "true").lower() == "true"
    )