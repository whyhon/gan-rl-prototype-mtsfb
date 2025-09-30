"""
Ollama Client Interface for GAN-RL Red Teaming Framework
Handles communication with the local Ollama gemma3:270m model
"""

import requests
import json
import time
from typing import Optional, Dict, Any
import logging

class OllamaClient:
    """Client for interacting with Ollama local API"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma2:2b"):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
    def check_connection(self) -> bool:
        """Check if Ollama server is running and accessible"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException as e:
            self.logger.error(f"Connection check failed: {e}")
            return False
    
    def generate_response(self, prompt: str, temperature: float = 0.7, 
                         max_tokens: int = 150) -> Optional[Dict[str, Any]]:
        """
        Generate response from Ollama model
        
        Args:
            prompt: Input text prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stop": ["\n\n"]
                }
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "response": result.get("response", "").strip(),
                    "prompt": prompt,
                    "response_time": time.time() - start_time,
                    "model": self.model,
                    "temperature": temperature,
                    "success": True
                }
            else:
                self.logger.error(f"API request failed: {response.status_code}")
                return None
                
        except requests.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            return None
    
    def test_model(self) -> bool:
        """Test if the model is working with a simple prompt"""
        test_prompt = "Hello, how are you?"
        result = self.generate_response(test_prompt)
        return result is not None and result.get("success", False)

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    client = OllamaClient()
    
    if client.check_connection():
        print("Ollama server is running")
        
        if client.test_model():
            print("Model is responding correctly")
            
            # Test with sample prompt
            test_result = client.generate_response("What is the capital of France?")
            if test_result:
                print(f"Sample response: {test_result['response']}")
                print(f"Response time: {test_result['response_time']:.2f}s")
        else:
            print("✗ Model test failed")
    else:
        print("✗ Cannot connect to Ollama server")
        print("Make sure Ollama is running with: ollama serve")