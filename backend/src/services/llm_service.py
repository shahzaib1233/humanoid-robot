"""
LLM service for the RAG Chatbot application.
Handles interactions with OpenRouter API for response generation.
"""
import os
import requests
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class LLMRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False


class LLMResponse(BaseModel):
    content: str
    model: str
    tokens_used: int


class LLMService:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "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")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        self.default_model = "openchat/openchat-7b:free"  # Using a free model from OpenRouter

    def generate_response(self, prompt: str, context: Optional[str] = None) -> LLMResponse:
        """
        Generate a response from the LLM based on the prompt and optional context.

        Args:
            prompt: User's query
            context: Retrieved context to include in the response

        Returns:
            LLMResponse object with the generated content
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Construct the system message with context if available
        system_message = "You are a helpful assistant for the Humanoid Robots textbook. Answer questions based on the provided context and cite sources when possible."

        messages = [
            {"role": "system", "content": system_message}
        ]

        if context:
            messages.append({
                "role": "system",
                "content": f"Context for answering the question: {context}"
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        payload = {
            "model": self.default_model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500
        }

        try:
            response = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()

            data = response.json()
            content = data['choices'][0]['message']['content']
            tokens_used = data.get('usage', {}).get('total_tokens', 0)

            return LLMResponse(
                content=content,
                model=data['model'],
                tokens_used=tokens_used
            )
        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM API: {e}")
            # Provide more specific error information
            error_msg = f"LLM API Error: {str(e)}"
            if 'response' in locals():
                try:
                    error_response = response.json()
                    error_msg += f" | Status: {response.status_code} | Detail: {error_response.get('error', {}).get('message', 'No error details')}"
                except:
                    error_msg += f" | Status: {response.status_code} | Raw: {response.text[:200]}"
            return LLMResponse(
                content=error_msg,
                model="",
                tokens_used=0
            )

    def generate_response_stream(self, prompt: str, context: Optional[str] = None):
        """
        Generate a streaming response from the LLM.
        This returns a generator that yields response chunks.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Construct the system message with context if available
        system_message = "You are a helpful assistant for the Humanoid Robots textbook. Answer questions based on the provided context and cite sources when possible."

        messages = [
            {"role": "system", "content": system_message}
        ]

        if context:
            messages.append({
                "role": "system",
                "content": f"Context for answering the question: {context}"
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        payload = {
            "model": self.default_model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": True  # Enable streaming
        }

        try:
            # Make the request with streaming enabled
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                stream=True
            )
            response.raise_for_status()

            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data.strip() == '[DONE]':
                            break
                        try:
                            import json
                            chunk_data = json.loads(data)  # Parse the chunk safely
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                delta = chunk_data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    yield content
                        except json.JSONDecodeError:
                            continue  # Skip malformed chunks
        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM API for streaming: {e}")
            yield f"LLM API Error: {str(e)}"