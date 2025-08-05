"""
Ollama LLM Integration Module
Handles communication with local Ollama models
"""

import requests
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma3:4b"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = 30
        
    def query_ollama(self, prompt: str, **kwargs) -> str:
        """Query Ollama with a prompt and return the response"""
        try:
            # Prepare the request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.1),
                    "top_p": kwargs.get("top_p", 0.9),
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "stop": kwargs.get("stop", [])
                }
            }
            
            logger.info(f"🤖 Querying Ollama model: {self.model}")
            
            # Make the request
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                
                if answer:
                    logger.info("✅ Ollama response received")
                    return answer
                else:
                    logger.warning("⚠️ Empty response from Ollama")
                    return "I'm having trouble generating a response right now. Please try again."
            
            else:
                logger.error(f"❌ Ollama API error: {response.status_code} - {response.text}")
                return "Sorry, I'm having trouble connecting to the AI model. Please try again later."
        
        except requests.exceptions.ConnectionError:
            logger.error("❌ Cannot connect to Ollama. Is it running?")
            return "🤖 AI model is currently unavailable. Please make sure Ollama is running."
        
        except requests.exceptions.Timeout:
            logger.error("❌ Ollama request timed out")
            return "The AI model is taking too long to respond. Please try a simpler question."
        
        except Exception as e:
            logger.error(f"❌ Unexpected error querying Ollama: {e}")
            return f"An unexpected error occurred: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if Ollama is available and the model is loaded"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                
                if self.model in available_models:
                    logger.info(f"✅ Ollama model {self.model} is available")
                    return True
                else:
                    logger.warning(f"⚠️ Model {self.model} not found. Available: {available_models}")
                    return False
            else:
                return False
        except:
            return False
    
    def list_models(self) -> list:
        """List available models in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except:
            return []

# Global client instance
_ollama_client = OllamaClient()

def query_ollama(prompt: str, model: str = None, **kwargs) -> str:
    """
    Simple function to query Ollama
    
    Args:
        prompt: The prompt to send to the model
        model: Optional model name (uses default if not provided)
        **kwargs: Additional options like temperature, max_tokens, etc.
    
    Returns:
        str: The model's response
    """
    global _ollama_client
    
    # Use different model if specified
    if model and model != _ollama_client.model:
        client = OllamaClient(model=model)
        return client.query_ollama(prompt, **kwargs)
    
    return _ollama_client.query_ollama(prompt, **kwargs)

def check_ollama_status() -> Dict[str, Any]:
    """Check Ollama status and available models"""
    global _ollama_client
    
    return {
        "available": _ollama_client.is_available(),
        "current_model": _ollama_client.model,
        "available_models": _ollama_client.list_models(),
        "base_url": _ollama_client.base_url
    }

def set_ollama_model(model: str) -> bool:
    """Set the default Ollama model"""
    global _ollama_client
    
    old_model = _ollama_client.model
    _ollama_client.model = model
    
    if _ollama_client.is_available():
        logger.info(f"✅ Switched to model: {model}")
        return True
    else:
        # Revert if the model is not available
        _ollama_client.model = old_model
        logger.error(f"❌ Model {model} not available, reverted to {old_model}")
        return False

# Test function
def test_ollama():
    """Test Ollama connection and model"""
    print("🧪 Testing Ollama connection...")
    
    status = check_ollama_status()
    print(f"📊 Status: {status}")
    
    if status["available"]:
        test_prompt = "Hello! Please respond with a brief greeting."
        response = query_ollama(test_prompt)
        print(f"🤖 Test response: {response}")
        return True
    else:
        print("❌ Ollama is not available")
        return False

if __name__ == "__main__":
    test_ollama()