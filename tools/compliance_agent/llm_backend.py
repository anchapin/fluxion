"""
LLM Backend Interface for Code Compliance Agent

Provides a unified interface for different LLM backends:
- Mock: For testing without external dependencies
- Ollama: Local LLM using llama.cpp
- OpenAI: Cloud-based LLM (GPT-4, GPT-3.5)
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM backend."""
    content: str
    model: str
    usage: Dict[str, int]
    latency_ms: float
    raw_response: Optional[Dict[str, Any]] = None


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available and ready."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the backend."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class MockLLMBackend(LLMBackend):
    """
    Mock LLM backend for testing.
    
    Returns predefined responses based on the prompt content.
    """
    
    def __init__(
        self,
        response_delay_ms: float = 100.0,
        response_template: Optional[str] = None
    ):
        """
        Initialize the mock backend.
        
        Args:
            response_delay_ms: Simulated response delay in milliseconds
            response_template: Optional custom response template
        """
        self._response_delay_ms = response_delay_ms
        self._response_template = response_template
        self._call_count = 0
    
    @property
    def name(self) -> str:
        return "mock"
    
    def is_available(self) -> bool:
        return True
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """Generate a mock response."""
        self._call_count += 1
        
        # Simulate network delay
        time.sleep(self._response_delay_ms / 1000.0)
        
        # Extract the last user message
        last_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_message = msg.get("content", "")
                break
        
        # Generate a response based on the prompt content
        response = self._generate_response(last_message)
        
        return LLMResponse(
            content=response,
            model="mock-llm",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            latency_ms=self._response_delay_ms,
            raw_response={"mock": True, "call_number": self._call_count}
        )
    
    def _generate_response(self, prompt: str) -> str:
        """Generate a mock response based on the prompt content."""
        prompt_lower = prompt.lower()
        
        if "compliance" in prompt_lower or "ashrae" in prompt_lower or "iecc" in prompt_lower:
            return json.dumps({
                "compliance_checks": [
                    {
                        "rule_id": "5.1.1",
                        "status": "COMPLIANT",
                        "message": "Building envelope thermal resistance meets minimum R-value requirements.",
                        "details": {"wall_r_value": 15.0, "required": 13.0}
                    },
                    {
                        "rule_id": "5.1.2",
                        "status": "NON_COMPLIANT",
                        "message": "Window U-factor exceeds maximum allowed value.",
                        "details": {"window_u_factor": 0.50, "required_max": 0.40},
                        "recommendation": "Replace windows with high-performance glazing (U-factor <= 0.40)"
                    },
                    {
                        "rule_id": "6.1.1",
                        "status": "COMPLIANT",
                        "message": "HVAC equipment efficiency meets requirements.",
                        "details": {"cop": 3.5, "required_min": 3.0}
                    }
                ],
                "overall_status": "NEEDS_REVIEW",
                "summary": {
                    "total_checks": 3,
                    "compliant": 2,
                    "non_compliant": 1,
                    "needs_review": 0
                }
            })
        else:
            return json.dumps({
                "response": "This is a mock response from the compliance agent.",
                "message": "The model data has been analyzed successfully."
            })


class OllamaBackend(LLMBackend):
    """
    Ollama local LLM backend.
    
    Uses Ollama API to interact with local LLMs.
    """
    
    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        timeout: int = 120
    ):
        """
        Initialize the Ollama backend.
        
        Args:
            model: Model name to use (e.g., llama2, mistral, codellama)
            base_url: Base URL for Ollama API
            timeout: Request timeout in seconds
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = None
    
    @property
    def name(self) -> str:
        return f"ollama:{self._model}"
    
    def _get_client(self):
        """Get or create the HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.Client(timeout=self._timeout)
        return self._client
    
    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            client = self._get_client()
            response = client.get(f"{self._base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                return self._model.split(":")[0] in model_names
            return False
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            return False
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using Ollama."""
        import httpx
        
        # Convert messages to Ollama format
        ollama_messages = []
        system_message = ""
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_message = content
            else:
                ollama_messages.append({
                    "role": role,
                    "content": content
                })
        
        # Build the prompt
        if system_message:
            prompt = f"System: {system_message}\n\n"
        else:
            prompt = ""
        
        for msg in ollama_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        # Make the request
        start_time = time.time()
        
        try:
            client = self._get_client()
            response = client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=result.get("response", ""),
                model=self._model,
                usage={
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                },
                latency_ms=latency_ms,
                raw_response=result
            )
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama request failed: {e}")


class OpenAIBackend(LLMBackend):
    """
    OpenAI LLM backend.
    
    Uses OpenAI API to interact with GPT models.
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120
    ):
        """
        Initialize the OpenAI backend.
        
        Args:
            model: Model name to use (e.g., gpt-4, gpt-3.5-turbo)
            api_key: OpenAI API key (will use OPENAI_API_KEY env var if not provided)
            base_url: Custom base URL for OpenAI-compatible APIs
            timeout: Request timeout in seconds
        """
        import os
        
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or "https://api.openai.com/v1"
        self._timeout = timeout
        self._client = None
        
        if not self._api_key:
            logger.warning("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
    
    @property
    def name(self) -> str:
        return f"openai:{self._model}"
    
    def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    timeout=self._timeout
                )
            except ImportError:
                raise ImportError("openai package is required. Install with: pip install openai")
        return self._client
    
    def is_available(self) -> bool:
        """Check if OpenAI API is accessible."""
        if not self._api_key:
            return False
        try:
            client = self._get_client()
            # Try a simple request to check connectivity
            client.models.list()
            return True
        except Exception as e:
            logger.warning(f"OpenAI availability check failed: {e}")
            return False
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using OpenAI."""
        start_time = time.time()
        
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            choice = response.choices[0]
            content = choice.message.content or ""
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                },
                latency_ms=latency_ms,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else {}
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI request failed: {e}")


class LLMBackendFactory:
    """Factory for creating LLM backends."""
    
    _backends = {
        "mock": MockLLMBackend,
        "ollama": OllamaBackend,
        "openai": OpenAIBackend,
    }
    
    @classmethod
    def create(
        cls,
        backend: str,
        **kwargs
    ) -> LLMBackend:
        """
        Create an LLM backend instance.
        
        Args:
            backend: Backend name ("mock", "ollama", or "openai")
            **kwargs: Additional arguments for the backend
            
        Returns:
            LLMBackend instance
            
        Raises:
            ValueError: If backend name is not recognized
        """
        backend_lower = backend.lower()
        
        if backend_lower not in cls._backends:
            raise ValueError(
                f"Unknown backend: {backend}. "
                f"Available backends: {', '.join(cls._backends.keys())}"
            )
        
        return cls._backends[backend_lower](**kwargs)
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """List available backend names."""
        return list(cls._backends.keys())


def create_backend(
    backend: Union[str, LLMBackend],
    **kwargs
) -> LLMBackend:
    """
    Create an LLM backend from a string or existing instance.
    
    Args:
        backend: Backend name or existing LLMBackend instance
        **kwargs: Additional arguments for the backend
        
    Returns:
        LLMBackend instance
    """
    if isinstance(backend, LLMBackend):
        return backend
    
    if isinstance(backend, str):
        return LLMBackendFactory.create(backend, **kwargs)
    
    raise TypeError(f"Expected str or LLMBackend, got {type(backend)}")
