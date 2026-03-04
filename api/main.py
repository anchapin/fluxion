"""
Fluxion REST API Server

A production-ready REST API server for evaluating building design populations remotely.
Built with FastAPI for high-performance async operations.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import json
import httpx
from pathlib import Path

# Import LLM module for local LLM support
from api.llm import (
    init_llm,
    unload_llm,
    get_llm_status,
    query_with_function_calling,
    llm_pool
)

# Import monitoring module for real-time monitoring and BAS integration
from api.monitoring import router as monitoring_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fluxion API",
    description="REST API for building energy simulation and optimization",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include monitoring router for real-time monitoring and BAS integration
app.include_router(monitoring_router)


# Data Models
class PopulationEvaluationRequest(BaseModel):
    """Request model for population evaluation."""
    population: List[List[float]] = Field(
        ...,
        description="List of parameter vectors. Each vector represents one design candidate. "
                   "Required parameters: [window_u_value, heating_setpoint, cooling_setpoint]"
    )
    use_surrogates: bool = Field(
        default=True,
        description="If true, use AI surrogates for faster evaluation"
    )


class PopulationEvaluationResponse(BaseModel):
    """Response model for population evaluation."""
    results: List[float] = Field(
        ...,
        description="List of EUI values (kWh/m²/year) for each candidate"
    )
    num_evaluated: int = Field(..., description="Number of configurations evaluated")
    evaluation_time_ms: float = Field(..., description="Evaluation time in milliseconds")


class ModelStatus(BaseModel):
    """Model loading status."""
    surrogate_loaded: bool
    surrogate_path: Optional[str] = None
    model_info: Optional[dict] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


class QuantizationRequest(BaseModel):
    """Request model for model quantization."""
    model_path: str = Field(..., description="Path to the input ONNX model")
    output_path: str = Field(..., description="Path to save the quantized model")
    quantization_type: str = Field(
        default="int8",
        description="Quantization type: int8, fp16, or dynamic"
    )


class QuantizationResponse(BaseModel):
    """Response model for quantization."""
    status: str
    original_size_kb: float
    quantized_size_kb: float
    reduction_percent: float
    output_path: str


# In-memory state (in production, use a proper state manager)
class AppState:
    def __init__(self):
        self.surrogate_loaded = False
        self.surrogate_path = None
        self.model_info = None


state = AppState()


# Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - returns API health status."""
    return HealthResponse(status="healthy", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring."""
    return HealthResponse(status="healthy", version="0.1.0")


@app.get("/status", response_model=ModelStatus)
async def get_status():
    """Get current model status."""
    return ModelStatus(
        surrogate_loaded=state.surrogate_loaded,
        surrogate_path=state.surrogate_path,
        model_info=state.model_info
    )


@app.post("/load-surrogate")
async def load_surrogate(request: dict):
    """
    Load an ONNX surrogate model.
    
    Request body:
        model_path: str - Path to the ONNX model file
    """
    model_path = request.get("model_path")
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path is required")
    
    try:
        # In a real implementation, this would load the model via PyO3 bindings
        # For now, we just track the state
        state.surrogate_loaded = True
        state.surrogate_path = model_path
        state.model_info = {
            "path": model_path,
            "loaded": True
        }
        
        logger.info(f"Loaded surrogate model from {model_path}")
        return {"status": "success", "message": f"Model loaded from {model_path}"}
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=PopulationEvaluationResponse)
async def evaluate_population(request: PopulationEvaluationRequest):
    """
    Evaluate a population of building design configurations.
    
    This is the core endpoint for optimization workflows. It accepts a list of
    parameter vectors and returns fitness values (EUI) for each candidate.
    
    Parameters:
        - population: List of [u_value, heating_setpoint, cooling_setpoint]
        - use_surrogates: Use AI surrogates (fast) vs physics (accurate)
    """
    import time
    
    if not request.population:
        raise HTTPException(status_code=400, detail="Population cannot be empty")
    
    start_time = time.time()
    
    try:
        # Import the Fluxion Rust bindings (when available)
        # For now, return mock results to demonstrate the API
        results = []
        
        for params in request.population:
            # Validate parameters
            if len(params) < 3:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Each parameter vector must have at least 3 elements, got {len(params)}"
                )
            
            u_value, heating, cooling = params[0], params[1], params[2]
            
            # Basic validation
            if not (0.1 <= u_value <= 5.0):
                results.append(float('nan'))
                continue
            if not (15.0 <= heating <= 25.0):
                results.append(float('nan'))
                continue
            if not (22.0 <= cooling <= 32.0):
                results.append(float('nan'))
                continue
            if heating >= cooling:
                results.append(float('nan'))
                continue
            
            # Mock calculation (replace with actual Fluxion call)
            # EUI = base_load + u_value_effect + heating_effect + cooling_effect
            base_eui = 100.0
            u_value_penalty = (u_value - 1.0) * 20.0
            heating_effect = (20.0 - heating) * 5.0 if heating < 20 else 0
            cooling_effect = (cooling - 27.0) * 5.0 if cooling > 27 else 0
            
            eui = base_eui + u_value_penalty + heating_effect + cooling_effect
            results.append(eui)
        
        evaluation_time = (time.time() - start_time) * 1000
        
        return PopulationEvaluationResponse(
            results=results,
            num_evaluated=len(results),
            evaluation_time_ms=evaluation_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload-surrogate")
async def unload_surrogate():
    """Unload the current surrogate model to free memory."""
    state.surrogate_loaded = False
    state.surrogate_path = None
    state.model_info = None
    return {"status": "success", "message": "Surrogate model unloaded"}


@app.post("/quantize", response_model=QuantizationResponse)
async def quantize_model(request: QuantizationRequest):
    """
    Quantize an ONNX model for optimized inference.
    
    Applies INT8 dynamic quantization to reduce model size by ~4x and speed up
    CPU inference for commercial building edge devices.
    
    Parameters:
        - model_path: Path to the input ONNX model
        - output_path: Path to save the quantized model
        - quantization_type: Type of quantization (int8, fp16, dynamic)
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import os
    
    input_path = request.model_path
    output_path = request.output_path
    quant_type = request.quantization_type.lower()
    
    # Validate input file exists
    if not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail=f"Input model not found: {input_path}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Apply quantization based on type
        if quant_type == "int8":
            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                weight_type=QuantType.QInt8,
            )
        elif quant_type == "fp16":
            # FP16 quantization requires onnxconverter-common
            from onnxruntime.quantization import quantize_static, QuantType
            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                weight_type=QuantType.QFLOAT8,
            )
        else:
            # Default to INT8 dynamic quantization
            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                weight_type=QuantType.QInt8,
            )
        
        # Get file sizes
        original_size = os.path.getsize(input_path)
        quantized_size = os.path.getsize(output_path)
        
        reduction = (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
        
        logger.info(f"Quantized {input_path} -> {output_path}: {original_size/1024:.1f}KB -> {quantized_size/1024:.1f}KB ({reduction:.1f}% reduction)")
        
        return QuantizationResponse(
            status="success",
            original_size_kb=original_size / 1024,
            quantized_size_kb=quantized_size / 1024,
            reduction_percent=reduction,
            output_path=output_path
        )
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quantization failed: {str(e)}")


# LLM Endpoints
@app.post("/llm/init")
async def llm_init(request: dict):
    """Initialize the local LLM with a GGUF model."""
    model_path = request.get("model_path")
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path is required")
    
    try:
        result = init_llm(model_path)
        return result
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm/status")
async def llm_status():
    """Get current LLM status."""
    try:
        return get_llm_status()
    except Exception as e:
        logger.error(f"Failed to get LLM status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/query")
async def llm_query(request: dict):
    """Query the LLM with natural language."""
    query = request.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    
    model_path = request.get("model_path")
    temperature = request.get("temperature", 0.7)
    max_tokens = request.get("max_tokens", 2048)
    
    try:
        # Get model path from pool if not specified
        if not model_path:
            status = get_llm_status()
            model_path = status.get("model_path")
        
        result = query_with_function_calling(
            query=query,
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            execute_functions=True,
            api_client=httpx.Client(base_url="http://localhost:8000", timeout=60.0)
        )
        
        return {
            "response": result.get("response", ""),
            "tool_calls": result.get("tool_calls"),
            "tool_results": result.get("tool_results"),
            "model": model_path,
            "inference_time_ms": result.get("inference_time_ms", 0)
        }
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/unload")
async def llm_unload():
    """Unload the current LLM to free memory."""
    try:
        result = unload_llm()
        return result
    except Exception as e:
        logger.error(f"Failed to unload LLM: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
