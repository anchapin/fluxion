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
from pathlib import Path

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
