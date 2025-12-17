"""
Multi-LLM service for the RAG Chatbot application.
Handles interactions with multiple LLM providers including Qwen and Google Gemini.
"""
import os
import requests
import json
from typing import Dict, Any, List, Optional, Generator
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


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


class MultiLLMService:
    def __init__(self):
        # Qwen API Configuration
        self.qwen_api_key = os.getenv("QWEN_API_KEY", "Wuhundv_V_2qQSrFU7bFQqaUto5PNohoHI39aFEiyCY7rlbvzFJxYr7RB3E5H3fNGI9Ww5Zyond1J0KFHGwnVw")
        self.qwen_base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation")

        # Google Gemini API Configuration
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyA15jFm76_K0gP2VtOYYPriunCXIuLIOmI")
        self.gemini_base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/models")

        # Default model selection
        self.default_model = os.getenv("DEFAULT_LLM_MODEL", "qwen")  # Can be 'qwen', 'gemini', or 'openrouter'

        # OpenRouter fallback (current implementation)
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "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")
        self.openrouter_base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    def _call_qwen_api(self, messages: List[Dict[str, str]], stream: bool = False, max_tokens: Optional[int] = None) -> LLMResponse:
        """
        Call Qwen API with the given messages.
        """
        headers = {
            "Authorization": f"Bearer {self.qwen_api_key}",
            "Content-Type": "application/json"
        }

        # Format messages for Qwen - system messages need to be handled differently
        formatted_messages = []
        system_message = ""

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                # Combine system messages
                system_message += f"{content}\n"
            else:
                formatted_messages.append({"role": role, "content": content})

        # If there's a system message, add it to the first user message or create one
        if system_message and formatted_messages and formatted_messages[0]["role"] == "user":
            formatted_messages[0]["content"] = f"{system_message}{formatted_messages[0]['content']}"
        elif system_message:
            formatted_messages.insert(0, {"role": "user", "content": system_message})

        payload = {
            "model": "qwen-max",  # Using qwen-max model
            "input": {
                "messages": formatted_messages
            },
            "parameters": {
                "temperature": 0.7,
                "max_tokens": max_tokens or 1000
            }
        }

        try:
            response = requests.post(self.qwen_base_url, json=payload, headers=headers)
            response.raise_for_status()

            data = response.json()

            # Extract content from Qwen response - check for different response formats
            if 'output' in data and 'choices' in data['output']:
                content = data['output']['choices'][0]['message']['content']
            elif 'output' in data and 'text' in data['output']:
                content = data['output']['text']
            else:
                content = "No response from Qwen API - unexpected response format"

            tokens_used = data.get('usage', {}).get('total_tokens', len(content.split()))

            return LLMResponse(
                content=content,
                model="qwen-max",
                tokens_used=tokens_used
            )
        except Exception as e:
            logger.error(f"Error calling Qwen API: {e}")
            error_msg = f"Qwen API Error: {str(e)}"
            if 'response' in locals():
                try:
                    error_response = response.json()
                    error_msg += f" | Detail: {error_response.get('error', {}).get('message', 'No error details')}"
                except:
                    error_msg += f" | Status: {response.status_code} | Raw: {response.text[:200]}"
            return LLMResponse(
                content=error_msg,
                model="qwen-max",
                tokens_used=0
            )

    def _call_gemini_api(self, messages: List[Dict[str, str]], stream: bool = False, max_tokens: Optional[int] = None) -> LLMResponse:
        """
        Call Google Gemini API with the given messages.
        """
        # Prepare the content for Gemini (system messages need to be handled differently)
        gemini_contents = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                # For system messages in Gemini, add to the first user message or create a user message
                if gemini_contents and gemini_contents[-1]["role"] == "user":
                    # Append to the last user message
                    gemini_contents[-1]["parts"][0]["text"] = f"{content}\n{gemini_contents[-1]['parts'][0]['text']}"
                else:
                    # Create a new user message with the system content
                    gemini_contents.append({
                        "role": "user",
                        "parts": [{"text": content}]
                    })
            else:
                # Convert assistant role to model role for Gemini
                gemini_role = "model" if role == "assistant" else role
                gemini_contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}]
                })

        # Construct the API URL for Gemini 2.5 Flash - using the correct endpoint format (available in free tier)
        model_name = "gemini-2.5-flash"  # Using gemini-2.5-flash which is available in free tier
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_api_key}"

        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": max_tokens or 1000
            }
        }

        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()

            data = response.json()

            # Extract content from Gemini response
            candidates = data.get('candidates', [])
            if candidates:
                content = candidates[0]['content']['parts'][0]['text']
                tokens_used = data.get('usageMetadata', {}).get('totalTokenCount', len(content.split()))
            else:
                content = "No response from Gemini API - no candidates returned"
                tokens_used = 0

            return LLMResponse(
                content=content,
                model="gemini-2.5-flash",
                tokens_used=tokens_used
            )
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            error_msg = f"Gemini API Error: {str(e)}"
            if 'response' in locals():
                try:
                    error_response = response.json()
                    error_msg += f" | Detail: {error_response.get('error', {}).get('message', 'No error details')}"
                except:
                    error_msg += f" | Status: {response.status_code} | Raw: {response.text[:200]}"
            return LLMResponse(
                content=error_msg,
                model="gemini-2.5-flash",
                tokens_used=0
            )

    def _call_openrouter_api(self, messages: List[Dict[str, str]], stream: bool = False, max_tokens: Optional[int] = None) -> LLMResponse:
        """
        Call OpenRouter API (current implementation) with the given messages.
        """
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "openchat/openchat-7b:free",  # Using the same free model as before
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": max_tokens or 500
        }

        if stream:
            payload["stream"] = True

        try:
            response = requests.post(f"{self.openrouter_base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()

            data = response.json()
            content = data['choices'][0]['message']['content']
            tokens_used = data.get('usage', {}).get('total_tokens', 0)

            return LLMResponse(
                content=content,
                model=data['model'],
                tokens_used=tokens_used
            )
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            error_msg = f"OpenRouter API Error: {str(e)}"
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

    def generate_response(self, prompt: str, context: Optional[str] = None, model: Optional[str] = None) -> LLMResponse:
        """
        Generate a response from the selected LLM based on the prompt and optional context.

        Args:
            prompt: User's query
            context: Retrieved context to include in the response
            model: Model to use ('qwen', 'gemini', 'openrouter', or None for default)

        Returns:
            LLMResponse object with the generated content
        """
        # Determine which model to use
        selected_model = model or self.default_model

        # Construct messages
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

        logger.info(f"Generating response using model: {selected_model}")

        if selected_model == "qwen":
            return self._call_qwen_api(messages)
        elif selected_model == "gemini":
            return self._call_gemini_api(messages)
        else:  # Default to OpenRouter
            return self._call_openrouter_api(messages)

    def generate_response_stream(self, prompt: str, context: Optional[str] = None, model: Optional[str] = None) -> Generator[str, None, None]:
        """
        Generate a streaming response from the selected LLM.
        This returns a generator that yields response chunks.
        Note: Streaming implementation varies by provider; some may not support true streaming.
        """
        # Determine which model to use
        selected_model = model or self.default_model

        # Construct messages
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

        logger.info(f"Generating streaming response using model: {selected_model}")

        # For now, we'll call the non-streaming API and yield the content in one chunk
        # In a real implementation, you'd need to implement proper streaming for each provider
        response = self.generate_response(prompt, context, selected_model)
        yield response.content