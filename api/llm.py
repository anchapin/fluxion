"""
Local LLM Service for Fluxion

Provides a local LLM interface for facility managers using llama.cpp.
Enables natural language queries to the Fluxion API with function calling support.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import httpx
from llama_cpp import Llama
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Fluxion API function definitions for tool calling
FLUXION_API_FUNCTIONS = [
    {
        "name": "evaluate_population",
        "description": "Evaluate a population of building design configurations. Use this when the user wants to evaluate or compare building designs. Each parameter vector should contain: window_u_value (W/m²K, range: 0.1-5.0), heating_setpoint (°C, range: 15-25), cooling_setpoint (°C, range: 22-32). Returns EUI values (kWh/m²/year) for each candidate.",
        "parameters": {
            "type": "object",
            "properties": {
                "population": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3
                    },
                    "description": "List of parameter vectors [window_u_value, heating_setpoint, cooling_setpoint]"
                },
                "use_surrogates": {
                    "type": "boolean",
                    "description": "Use AI surrogates for faster evaluation (recommended)",
                    "default": True
                }
            },
            "required": ["population"]
        }
    },
    {
        "name": "get_model_status",
        "description": "Get the current status of the surrogate model (loaded/unloaded)",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "load_surrogate",
        "description": "Load a surrogate model for fast building energy prediction",
        "parameters": {
            "type": "object",
            "properties": {
                "model_path": {
                    "type": "string",
                    "description": "Path to the ONNX model file"
                }
            },
            "required": ["model_path"]
        }
    }
]


class LLMQueryRequest(BaseModel):
    """Request model for LLM queries."""
    query: str = Field(..., description="Natural language query from the user")
    model_path: Optional[str] = Field(None, description="Path to GGUF model (optional, uses default if not provided)")
    temperature: float = Field(0.7, description="Sampling temperature (0.0-2.0)")
    max_tokens: int = Field(2048, description="Maximum tokens to generate")
    context_window: int = Field(8192, description="Context window size")


class LLMQueryResponse(BaseModel):
    """Response model for LLM queries."""
    response: str = Field(..., description="Natural language response from the LLM")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="JSON tool calls made by the LLM")
    tool_results: Optional[List[Dict[str, Any]]] = Field(None, description="Results from executed tool calls")
    model: str = Field(..., description="Model used for inference")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class LLMPool:
    """
    Manages LLM instances for the Fluxion API.
    Supports multiple models and provides function calling capabilities.
    """
    
    def __init__(self):
        self._models: Dict[str, Llama] = {}
        self._default_model_path: Optional[str] = None
        self._api_functions = FLUXION_API_FUNCTIONS
    
    @property
    def functions_schema(self) -> str:
        """Get the function calling schema as a formatted string."""
        return json.dumps(self._api_functions, indent=2)
    
    def set_default_model(self, model_path: str):
        """Set the default model path."""
        self._default_model_path = model_path
        logger.info(f"Default model set to: {model_path}")
    
    def get_model(self, model_path: Optional[str] = None) -> Llama:
        """Get or load an LLM instance."""
        path = model_path or self._default_model_path
        
        if not path:
            raise ValueError("No model path provided and no default model set")
        
        # Return cached model if available
        if path in self._models:
            return self._models[path]
        
        # Load new model
        logger.info(f"Loading LLM from: {path}")
        
        # Check if file exists
        model_file = Path(path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        llm = Llama(
            model_path=str(path),
            n_ctx=8192,
            n_threads=4,
            n_gpu_layers=0,  # Set to >0 if GPU acceleration is available
            verbose=False
        )
        
        self._models[path] = llm
        logger.info(f"LLM loaded successfully: {path}")
        
        return llm
    
    def is_model_loaded(self, model_path: Optional[str] = None) -> bool:
        """Check if a model is loaded."""
        path = model_path or self._default_model_path
        return path in self._models
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the LLM pool."""
        return {
            "loaded": bool(self._models),
            "model_path": self._default_model_path,
            "loaded_models": list(self._models.keys())
        }
    
    def unload_model(self, model_path: Optional[str] = None):
        """Unload a model to free memory."""
        path = model_path or self._default_model_path
        
        if path and path in self._models:
            del self._models[path]
            logger.info(f"Model unloaded: {path}")
    
    def create_function_calling_prompt(
        self, 
        user_query: str, 
        include_function_results: bool = False,
        function_results: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Create a prompt for function calling.
        
        Returns a message list suitable for chat completion.
        """
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful assistant for facility managers working with building energy simulation. "
                "You have access to the Fluxion API functions to evaluate building designs and get model status. "
                "When the user asks about building designs, energy efficiency, or wants to evaluate configurations, "
                "you should use the provided functions to get accurate data.\n\n"
                f"Available functions:\n{self.functions_schema}\n\n"
                "IMPORTANT: When you need to call a function, respond with a JSON object in this exact format:\n"
                "```json\n"
                "{\n"
                '  "name": "function_name",\n'
                '  "arguments": { ... }\n'
                "}\n"
                "```\n"
                "Do not include any other text in your response when making function calls. "
                "After receiving function results, summarize the data in plain English for the facility manager."
            )
        }
        
        user_message = {
            "role": "user",
            "content": user_query
        }
        
        messages = [system_message]
        
        if include_function_results and function_results:
            assistant_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {
                            "name": "evaluate_population",
                            "arguments": json.dumps(user_query)
                        }
                    }
                ]
            }
            messages.append(assistant_message)
            
            tool_result_message = {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": function_results
            }
            messages.append(tool_result_message)
        else:
            messages.append(user_message)
        
        return messages


# Global LLM pool instance
llm_pool = LLMPool()


def initialize_llm(
    model_path: str,
    context_window: int = 8192,
    n_threads: int = 4,
    n_gpu_layers: int = 0
) -> Llama:
    """
    Initialize the default LLM model.
    
    Args:
        model_path: Path to the GGUF model file
        context_window: Context window size
        n_threads: Number of CPU threads
        n_gpu_layers: Number of GPU layers (0 for CPU only)
    
    Returns:
        Loaded Llama instance
    """
    llm_pool.set_default_model(model_path)
    return llm_pool.get_model(model_path)


# Convenience wrapper functions for FastAPI endpoints
def init_llm(model_path: str, context_window: int = 8192, n_threads: int = 4, n_gpu_layers: int = 0) -> Dict[str, Any]:
    """Initialize the LLM and return status."""
    try:
        llm = initialize_llm(model_path, context_window, n_threads, n_gpu_layers)
        return {
            "status": "success",
            "message": f"LLM initialized from {model_path}",
            "model_path": model_path
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


def unload_llm() -> Dict[str, Any]:
    """Unload the LLM from memory."""
    try:
        llm_pool.unload_model()
        return {
            "status": "success",
            "message": "LLM unloaded"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


def get_llm_status() -> Dict[str, Any]:
    """Get current LLM status."""
    status = llm_pool.get_status()
    return {
        "model_loaded": status["loaded"],
        "model_path": status.get("model_path"),
        "available_functions": FLUXION_API_FUNCTIONS
    }


def query_with_function_calling(
    query: str,
    model_path: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    execute_functions: bool = True,
    api_client=None
) -> Dict[str, Any]:
    """
    Process a natural language query with function calling support.
    
    Args:
        query: User's natural language query
        model_path: Optional path to GGUF model
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        execute_functions: Whether to execute function calls
        api_client: Optional HTTP client for executing API calls
    
    Returns:
        Dictionary with response, tool_calls, and tool_results
    """
    import time
    import re
    
    llm = llm_pool.get_model(model_path)
    
    # Create messages for chat completion
    messages = llm_pool.create_function_calling_prompt(query)
    
    # First pass: Get LLM response with potential function calls
    start_time = time.time()
    
    response = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        functions=FLUXION_API_FUNCTIONS,
        function_call="auto"
    )
    
    inference_time = (time.time() - start_time) * 1000
    
    # Extract the assistant's response
    assistant_message = response["choices"][0]["message"]
    
    tool_calls = None
    tool_results = None
    final_response = None
    
    # Check if the LLM made function calls (native function_call)
    if assistant_message.get("function_call"):
        tool_calls = [assistant_message["function_call"]]
    else:
        # Try to parse function call from text response
        content = assistant_message.get("content", "")
        
        # Look for JSON in the response that matches function call format
        # Pattern 1: {"name": "function_name", "arguments": {...}}
        # Pattern 2: {"name": "function_name", "arguments": {...}} inside code blocks
        json_patterns = [
            r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{[\s\S]*?\})\s*\}',
            r'```json\s*\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{[\s\S]*?\})\s*\}\s*```',
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, content)
            if match:
                func_name = match.group(1)
                try:
                    # Extract just the arguments portion
                    args_str = match.group(2)
                    func_args = json.loads(args_str)
                    tool_calls = [{
                        "name": func_name,
                        "arguments": json.dumps(func_args)
                    }]
                    break
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse function arguments: {e}")
                    continue
    
    # Execute function calls if detected
    if tool_calls and execute_functions and api_client:
        # Execute the function call(s)
        tool_results = []
        
        for tool_call in tool_calls:
            func_name = tool_call["name"]
            try:
                func_args = json.loads(tool_call["arguments"])
            except (json.JSONDecodeError, KeyError):
                func_args = {}
            
            try:
                # Execute the function via API client
                result = execute_api_call(api_client, func_name, func_args)
                tool_results.append({
                    "function": func_name,
                    "arguments": func_args,
                    "result": result
                })
            except Exception as e:
                tool_results.append({
                    "function": func_name,
                    "arguments": func_args,
                    "error": str(e)
                })
        
        # Second pass: Get summary from LLM with function results
        # Use a simpler approach - append function results to a new query
        if tool_results:
            function_results_str = json.dumps(tool_results, indent=2)
            
            # Create a new completion request with results
            summary_messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that explains Fluxion API results in plain English. The user asked a question, and the function results are provided below. Provide a clear, concise summary."
                },
                {
                    "role": "user",
                    "content": f"Original question: {query}\n\nFunction results:\n{function_results_str}\n\nPlease summarize these results in plain English."
                }
            ]
            
            # Get final summary from LLM
            summary_response = llm.create_chat_completion(
                messages=summary_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            final_response = summary_response["choices"][0]["message"].get("content", "")
    
    if not final_response:
        final_response = assistant_message.get("content", "")
    
    return {
        "response": final_response or "No response generated",
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "inference_time_ms": inference_time
    }


def execute_api_call(client, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a Fluxion API call.
    
    Args:
        client: HTTP client (httpx or similar)
        function_name: Name of the function to call
        arguments: Arguments for the function
    
    Returns:
        API response as dictionary
    """
    import httpx
    
    if function_name == "evaluate_population":
        response = client.post(
            "/evaluate",
            json={
                "population": arguments.get("population", []),
                "use_surrogates": arguments.get("use_surrogates", True)
            }
        )
        response.raise_for_status()
        return response.json()
    
    elif function_name == "get_model_status":
        response = client.get("/status")
        response.raise_for_status()
        return response.json()
    
    elif function_name == "load_surrogate":
        response = client.post(
            "/load-surrogate",
            json={"model_path": arguments.get("model_path")}
        )
        response.raise_for_status()
        return response.json()
    
    else:
        raise ValueError(f"Unknown function: {function_name}")
