import streamlit as st
import os
import logging
from datetime import datetime
import re
import json
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
# Langchain for vector store
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# Google Gemini
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---- SLTChatbot class (retain all logic as before) ----

class SLTChatbot:
    def __init__(self, use_local_llm=True, gemini_api_key="AIzaSyCjS3Uj_ZdQX4TnSjx1CmCPMkLsc4sM0_4"):
        self.use_local_llm = use_local_llm
        self.gemini_api_key = gemini_api_key
        self.gemini_model = None
        self.vector_store = None
        self.packages_vector_store = None  # Dedicated packages vector store
        self.embeddings = None
        self.branches = []
        self.city_names = set()
        self.vector_store_path = None
        self.packages_vector_store_path = "./packages_chroma_db"
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
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                logger.info("✅ Google Gemini initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Google Gemini: {e}")
                raise

    def init_vector_db(self):
        """Initialize vector databases in in-memory mode to bypass SQLite issues"""
        try:
            # In-memory general vector store
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                model_kwargs={"device": "cpu"}
            )
            self.vector_store = Chroma(
                persist_directory=None,  # None → in-memory
                embedding_function=self.embeddings,
                collection_name="general_in_memory"
            )
            general_doc_count = self.vector_store._collection.count()
            logger.info(f"✅ General vector store initialized in-memory with {general_doc_count} documents")

            # Packages vector store in-memory
            if Path(self.packages_vector_store_path).exists():
                self.packages_vector_store = Chroma(
                    persist_directory=None,  # in-memory
                    embedding_function=self.embeddings,
                    collection_name="packages_in_memory"
                )
                packages_doc_count = self.packages_vector_store._collection.count()
                logger.info(f"✅ Packages vector store initialized in-memory with {packages_doc_count} documents")
            else:
                self.packages_vector_store = None
                logger.warning(f"⚠️ Packages vector store not found at {self.packages_vector_store_path}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize vector stores: {e}")
            self.vector_store = None
            self.packages_vector_store = None

    def load_branches(self):
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

    def classify_query(self, query):
        """Enhanced query classification for better routing"""
        query_lower = query.lower()
        
        classification = {
            "is_package_query": False,
            "is_fiber_query": False,
            "is_prepaid_query": False,
            "is_peo_tv_query": False,
            "is_extra_gb_query": False,
            "is_list_query": False,
            "is_price_query": False,
            "is_speed_query": False,
            "is_data_query": False,
            "wants_comparison": False,
            "query_type": "general"
        }
        
        # Package-related keywords
        package_keywords = ["package", "plan", "broadband", "internet", "data", 
                          "monthly", "unlimited", "gb", "tb", "subscription"]
        
        # Fiber-specific keywords
        fiber_keywords = ["fiber", "fibre", "ftth", "optical", "high speed"]
        
        # Prepaid keywords
        prepaid_keywords = ["prepaid", "reload", "top up", "topup", "no contract", 
                           "temporary", "flexible"]
        
        # PEO TV keywords
        peo_tv_keywords = ["peo tv", "peotv", "peo", "television", "tv channel", 
                          "tv package", "tv plan", "streaming"]
        
        # Extra GB keywords
        extra_gb_keywords = ["extra gb", "additional data", "more data", "data booster", 
                            "add on", "addon", "extra data"]
        
        # List query indicators
        list_indicators = ["list", "all packages", "all plans", "show me", "what packages",
                          "available packages", "available plans", "options", "tell me about",
                          "give me", "show all", "display", "overview", "summary"]
        
        # Price-related keywords
        price_keywords = ["price", "cost", "fee", "charge", "rental", "payment", 
                         "cheap", "affordable", "expensive", "under", "below", "less than"]
        
        # Speed-related keywords
        speed_keywords = ["speed", "mbps", "gbps", "fast", "slow", "bandwidth", "performance"]
        
        # Data-related keywords
        data_keywords = ["data", "gb", "tb", "unlimited", "limit", "quota", "allowance"]
        
        # Comparison keywords
        comparison_keywords = ["compare", "vs", "versus", "difference", "better", 
                              "which", "choose", "best", "recommend"]
        
        # Check for each classification
        classification["is_package_query"] = any(kw in query_lower for kw in package_keywords)
        classification["is_fiber_query"] = any(kw in query_lower for kw in fiber_keywords)
        classification["is_prepaid_query"] = any(kw in query_lower for kw in prepaid_keywords)
        classification["is_peo_tv_query"] = any(kw in query_lower for kw in peo_tv_keywords)
        classification["is_extra_gb_query"] = any(kw in query_lower for kw in extra_gb_keywords)
        classification["is_list_query"] = any(kw in query_lower for kw in list_indicators)
        classification["is_price_query"] = any(kw in query_lower for kw in price_keywords)
        classification["is_speed_query"] = any(kw in query_lower for kw in speed_keywords)
        classification["is_data_query"] = any(kw in query_lower for kw in data_keywords)
        classification["wants_comparison"] = any(kw in query_lower for kw in comparison_keywords)
        
        # Determine primary query type
        if classification["is_peo_tv_query"]:
            classification["query_type"] = "peo_tv"
        elif classification["is_extra_gb_query"]:
            classification["query_type"] = "extra_gb"
        elif classification["is_fiber_query"]:
            classification["query_type"] = "fiber_package"
        elif classification["is_prepaid_query"]:
            classification["query_type"] = "prepaid_package"
        elif classification["is_package_query"] or classification["is_data_query"] or classification["is_price_query"]:
            classification["query_type"] = "broadband_package"
        else:
            classification["query_type"] = "general"
        
        return classification

    def preprocess_query(self, query):
        """Enhanced query preprocessing"""
        query = query.strip()
        
        # Expand abbreviations and add context
        expansions = {
            "peo": "PEO TV television channels",
            "adsl": "ADSL broadband internet",
            "fiber": "fiber fibre broadband internet high speed FTTH",
            "fibre": "fiber fibre broadband internet high speed FTTH",
            "wifi": "WiFi wireless internet broadband",
            "tv": "television PEO TV channels",
            "4g": "4G LTE mobile broadband wireless",
            "lte": "LTE 4G mobile broadband wireless",
            "prepaid": "prepaid reload topup no contract flexible",
            "postpaid": "postpaid monthly contract subscription",
            "unlimited": "unlimited no limit unrestricted data",
            "gb": "GB gigabyte data allowance",
            "tb": "TB terabyte data allowance",
            "mbps": "Mbps megabit speed bandwidth"
        }
        
        query_lower = query.lower()
        expanded_query = query
        
        for abbrev, expansion in expansions.items():
            if abbrev in query_lower:
                expanded_query += f" {expansion}"
        
        # Add context for short queries
        if len(query.split()) < 3:
            if "package" in query_lower or "plan" in query_lower:
                expanded_query += " SLT broadband internet packages plans prices data"
            else:
                expanded_query += " SLT broadband services"
        
        return expanded_query

    def find_relevant_chunks(self, query, top_n=3):
        """Enhanced chunk retrieval with better package database utilization"""
        if not self.vector_store:
            logger.warning("⚠️ General vector store not initialized")
            return []

        # Classify the query
        classification = self.classify_query(query)
        processed_query = self.preprocess_query(query)
        
        logger.info(f"🔍 Query Classification: {classification['query_type']}")
        logger.info(f"🔍 Processed Query: {processed_query}")

        # Adjust top_n for list queries
        if classification["is_list_query"]:
            top_n = max(top_n, 8)
            logger.info(f"📋 List query detected, expanding results to {top_n}")

        matched_chunks = []
        
        try:
            # Route queries based on classification
            if classification["query_type"] == "peo_tv":
                # PEO TV: Only use general database
                logger.info("📺 PEO TV query - using ONLY general database")
                results = self.vector_store.similarity_search_with_score(processed_query, k=top_n)
                for doc, score in results:
                    matched_chunks.append({
                        "title": doc.metadata.get("title", "Unknown Title"),
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                        "score": score,
                        "db_source": "general"
                    })
                    
            elif classification["query_type"] in ["fiber_package", "prepaid_package", "broadband_package"]:
                # Package queries: Heavily prioritize packages database
                if self.packages_vector_store:
                    logger.info(f"📦 {classification['query_type']} - prioritizing packages database")
                    
                    # Get more results from packages DB for comprehensive coverage
                    package_k = top_n if classification["is_list_query"] else max(3, top_n - 1)
                    
                    # Build specific search query based on classification
                    if classification["is_fiber_query"]:
                        search_query = f"fiber fibre FTTH optical high speed {processed_query}"
                    elif classification["is_prepaid_query"]:
                        search_query = f"prepaid reload topup flexible {processed_query}"
                    else:
                        search_query = processed_query
                    
                    # Search packages database with metadata filtering if possible
                    package_results = self.packages_vector_store.similarity_search_with_score(
                        search_query, 
                        k=package_k * 2  # Get more results initially for filtering
                    )
                    
                    # Filter and rank results based on query type
                    for doc, score in package_results:
                        metadata = doc.metadata
                        
                        # Apply type-specific filtering
                        if classification["is_fiber_query"]:
                            # Prioritize fiber packages
                            connection_type = metadata.get("connection_type", "").lower()
                            if "fib" in connection_type or "fib" in doc.page_content.lower():
                                score *= 0.8  # Better score for fiber matches
                        
                        if classification["is_prepaid_query"]:
                            # Prioritize prepaid packages
                            package_type = metadata.get("package_type", "").lower()
                            if package_type == "prepaid":
                                score *= 0.8  # Better score for prepaid matches
                        
                        # Apply price filtering if price query
                        if classification["is_price_query"]:
                            # Extract price range from query if present
                            price_match = re.search(r'under\s+(\d+)|below\s+(\d+)|less\s+than\s+(\d+)', query.lower())
                            if price_match:
                                max_price = int(next(g for g in price_match.groups() if g))
                                monthly_price = metadata.get("monthly_price_num", float('inf'))
                                if monthly_price <= max_price:
                                    score *= 0.7  # Better score for matching price range
                        
                        matched_chunks.append({
                            "title": metadata.get("title", "Unknown Package"),
                            "content": doc.page_content,
                            "source": "https://www.slt.lk/en/broadband/packages",
                            "score": score,
                            "db_source": "packages",
                            "metadata": metadata
                        })
                    
                    # Sort by score and limit
                    matched_chunks = sorted(matched_chunks, key=lambda x: x["score"])[:package_k]
                    
                    # Add minimal general context only if needed
                    if len(matched_chunks) < 2 and not classification["is_list_query"]:
                        logger.info("🔍 Adding minimal general context")
                        general_results = self.vector_store.similarity_search_with_score(processed_query, k=1)
                        for doc, score in general_results:
                            matched_chunks.append({
                                "title": doc.metadata.get("title", "Unknown Title"),
                                "content": doc.page_content,
                                "source": doc.metadata.get("source", ""),
                                "score": score,
                                "db_source": "general"
                            })
                else:
                    # Fallback to general database if packages DB not available
                    logger.warning("⚠️ Packages database not available, using general database")
                    results = self.vector_store.similarity_search_with_score(processed_query, k=top_n)
                    for doc, score in results:
                        matched_chunks.append({
                            "title": doc.metadata.get("title", "Unknown Title"),
                            "content": doc.page_content,
                            "source": doc.metadata.get("source", ""),
                            "score": score,
                            "db_source": "general"
                        })
                        
            else:
                # General queries: Use general database primarily
                logger.info("🔍 General query - using general database")
                results = self.vector_store.similarity_search_with_score(processed_query, k=top_n)
                for doc, score in results:
                    matched_chunks.append({
                        "title": doc.metadata.get("title", "Unknown Title"),
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                        "score": score,
                        "db_source": "general"
                    })

            # Remove duplicates based on content
            seen_content = set()
            unique_chunks = []
            for chunk in matched_chunks:
                content_key = chunk["content"][:200]  # Use first 200 chars as key
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_chunks.append(chunk)

            # Sort by relevance
            unique_chunks = sorted(unique_chunks, key=lambda x: x["score"])
            
            # Apply final limit
            unique_chunks = unique_chunks[:top_n]
            
            # Clean up internal metadata
            for chunk in unique_chunks:
                chunk.pop("score", None)
                chunk.pop("db_source", None)
                chunk.pop("metadata", None)
            
            logger.info(f"✅ Returning {len(unique_chunks)} unique chunks")
            
            # Log distribution for debugging
            packages_count = sum(1 for c in matched_chunks if c.get("db_source") == "packages")
            general_count = sum(1 for c in matched_chunks if c.get("db_source") == "general")
            logger.info(f"📊 Source distribution: {packages_count} packages, {general_count} general")
            
            return unique_chunks

        except Exception as e:
            logger.error(f"❌ Error searching vector stores: {e}")
            return []

    def find_nearest_branches(self, user_coords, top_n=3):
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
        source_links = {chunk['source'] for chunk in chunks if chunk.get('source')}
        if not source_links:
            return response_text
        
        response_text += "\n\n🔗 **Sources:**\n"
        for link in source_links:
            response_text += f"- {link}\n"
        return response_text.strip()

    def query_llm(self, user_query, context_chunks, user_id):
        """Enhanced LLM query with better context handling"""
        classification = self.classify_query(user_query)
        logger.info(f"🤖 Using {'Local LLM' if self.use_local_llm else 'Google Gemini'}")
        
        history = self.get_session_history(user_id)
        history_text = "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in history if h['role'] != "system"])
        
        if self.use_local_llm:
            context_blocks = []
            for chunk in context_chunks:
                url = chunk['source']
                title = chunk.get('title', 'No Title')
                content = chunk['content'][:1500]
                block = f"Page: {url}\nTitle: {title}\nContent:\n{content}"
                context_blocks.append(block)
            full_context = "\n\n---\n\n".join(context_blocks)
            
            # Build system prompt based on query classification
            base_prompt = "You are a helpful assistant answering questions about SLT (Sri Lanka Telecom) broadband, PEO TV, branches, and services.\n"
            
            if classification["is_list_query"]:
                specific_prompt = (
                    "The user is asking for a comprehensive list or overview. "
                    "Provide detailed information about ALL the packages/options mentioned in the context. "
                    "Format your response as a complete list with clear details for each item. "
                    "Do not limit yourself to just 1-2 examples - include ALL relevant packages from the provided context.\n"
                )
            elif classification["wants_comparison"]:
                specific_prompt = (
                    "The user wants to compare packages or services. "
                    "Provide a clear comparison highlighting the key differences between the options. "
                    "Use the information from the context to make meaningful comparisons.\n"
                )
            elif classification["is_price_query"]:
                specific_prompt = (
                    "The user is asking about pricing. "
                    "Provide clear pricing information including monthly rentals, startup fees, and any other costs mentioned. "
                    "If there are multiple price points, list them clearly.\n"
                )
            else:
                specific_prompt = ""
            
            system_prompt = (
                f"{base_prompt}{specific_prompt}\n"
                f"Conversation history:\n{history_text}\n\n"
                f"User question: {user_query}\n\n"
                f"Based on the following extracted content from the SLT website:\n\n{full_context}\n\n"
                "Answer clearly and helpfully. If asked for names/titles only, provide just the names. "
                "Do not provide links in your answer unless specifically asked, as they will be added automatically."
            )
            
            try:
                response = httpx.post(
                    "http://127.0.0.1:11434/api/chat",
                    json={
                        "model": "gemma2:2b",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_query}
                        ],
                        "stream": True
                    },
                    timeout=90 if classification["is_list_query"] else 60
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
            
            # Build prompt based on query classification
            base_prompt = "You are a helpful assistant answering questions about SLT (Sri Lanka Telecom) broadband, PEO TV, branches, and services.\n"
            
            if classification["is_list_query"]:
                specific_prompt = (
                    "The user is asking for a comprehensive list or overview. "
                    "Provide detailed information about ALL the packages/options mentioned in the context. "
                    "Format your response as a complete list with clear details for each item.\n"
                )
            elif classification["wants_comparison"]:
                specific_prompt = (
                    "The user wants to compare packages or services. "
                    "Provide a clear comparison highlighting the key differences.\n"
                )
            elif classification["is_price_query"]:
                specific_prompt = (
                    "The user is asking about pricing. "
                    "Provide clear pricing information from the context.\n"
                )
            else:
                specific_prompt = ""
            
            prompt = (
                f"{base_prompt}{specific_prompt}\n"
                f"Conversation history:\n{history_text}\n\n"
                f"User question: {user_query}\n\n"
                f"Based on the following extracted content from the SLT website:\n\n{full_context}\n\n"
                "Answer clearly and helpfully. If asked for names/titles only, provide just the names. "
                "Do not provide links in your answer unless specifically asked, as they will be added automatically."
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
        """Generate fallback response when LLM fails"""
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
        """Switch between local LLM and Gemini"""
        logger.info(f"🔄 Switching to {'Local LLM' if use_local_llm else 'Google Gemini'}")
        self.use_local_llm = use_local_llm
        if not use_local_llm and gemini_api_key:
            self.gemini_api_key = gemini_api_key
        self.setup_llm()

# ---- Streamlit UI ----

# Force use Google Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCjS3Uj_ZdQX4TnSjx1CmCPMkLsc4sM0_4")

# Initialize chatbot with Gemini only
chatbot = SLTChatbot(
    use_local_llm=False,  # Force Gemini
    gemini_api_key=GEMINI_API_KEY
)

# Streamlit page settings
st.set_page_config(page_title="SLT AI Chatbot", page_icon="💬", layout="wide")
st.title("💬 SLT AI Chatbot")

# Sidebar for settings (debug only)
with st.sidebar:
    st.header("⚙️ Settings")
    debug_mode = st.checkbox("Debug mode", value=(os.environ.get("DEBUG", "true").lower() == "true"))

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask me something..."):
    user_id = "default"
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    chatbot.add_to_session(user_id, "user", prompt)

    # Casual replies
    casual_replies = {
        "hello": "👋 Hello! I'm your SLT assistant. How can I help you today?",
        "hi": "Hi there! Ask me about SLT broadband packages, PEO TV, or branch locations.",
        "hey": "Hey! What would you like to know about SLT services?",
        "thanks": "You're welcome! Anything else I can help with?",
        "thank you": "Happy to help! Is there anything else about SLT services you'd like to know?",
        "bye": "Goodbye! Have a great day!",
        "goodbye": "Take care! Feel free to come back if you have more questions."
    }
    
    user_lower = prompt.lower()
    if user_lower in casual_replies:
        response = casual_replies[user_lower]
        chatbot.add_to_session(user_id, "assistant", response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
    else:
        # Location queries
        location_keywords = ["branch", "location", "office", "near", "address", "where", "closest", "nearby"]
        user_words = user_lower.split()
        is_location_query = any(word in user_words for word in location_keywords)
        found_city = next((city for city in chatbot.city_names if city in user_lower), None)

        if is_location_query:
            if found_city:
                logger.info(f"📍 Specific city query detected: {found_city}")
                response = chatbot.handle_location_query(found_city)
            else:
                response = ("📍 To find the nearest SLT branches, please share your city or area (e.g., 'Colombo', 'Kandy', 'Galle').\n\n"
                            "🔒 *Your location data is only used to find nearby branches and is not stored.*")
            chatbot.add_to_session(user_id, "assistant", response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        else:
            # General/package queries
            chunks = chatbot.find_relevant_chunks(prompt, top_n=5)
            if not chunks:
                response = ("❌ I couldn't find specific information about that topic. \n\n"
                            "🔗 **Try visiting:**\n"
                            "- https://www.slt.lk/en/broadband/packages\n"
                            "- https://www.slt.lk/en/peo-tv/packages\n"
                            "- **Customer Service:** 1212")
                chatbot.add_to_session(user_id, "assistant", response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            else:
                llm_response = chatbot.query_llm(prompt, chunks, user_id)
                st.session_state.messages.append({"role": "assistant", "content": llm_response})
                with st.chat_message("assistant"):
                    st.markdown(llm_response)