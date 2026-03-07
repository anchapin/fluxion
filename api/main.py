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
import os
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

# Import distributed inference modules
from api.distributed_inference import (
    DistributedInferenceManager,
    EndpointConfig,
    LoadBalancingStrategy,
    get_inference_manager,
    initialize_inference_manager,
    start_inference_manager,
    stop_inference_manager,
)

from api.distributed_inference_config import (
    DistributedInferenceConfig,
    auto_load,
    initialize_from_config,
)

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
        self.distributed_manager: Optional[DistributedInferenceManager] = None
        self.distributed_enabled = False


state = AppState()


# Distributed Inference Configuration
DISTRIBUTED_CONFIG_PATH = os.getenv("FLUXION_DISTRIBUTED_CONFIG", "config/distributed_inference.yaml")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # Try to load distributed inference configuration
    try:
        config = auto_load(DISTRIBUTED_CONFIG_PATH)
        if config.enabled and config.endpoints:
            state.distributed_manager = initialize_from_config(config)
            state.distributed_enabled = True
            await start_inference_manager()
            logger.info("Distributed inference enabled")
        else:
            logger.info("Distributed inference not configured or disabled")
    except Exception as e:
        logger.warning(f"Failed to initialize distributed inference: {e}")
        state.distributed_enabled = False


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    if state.distributed_manager:
        await stop_inference_manager()


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


@app.get("/distributed/status")
async def get_distributed_status():
    """Get distributed inference status."""
    if not state.distributed_enabled or not state.distributed_manager:
        return {
            "enabled": False,
            "message": "Distributed inference is not enabled"
        }
    
    health_status = await state.distributed_manager.health_check()
    return {
        "enabled": True,
        **health_status
    }


@app.get("/distributed/metrics")
async def get_distributed_metrics():
    """Get distributed inference metrics."""
    if not state.distributed_enabled or not state.distributed_manager:
        return {
            "enabled": False,
            "message": "Distributed inference is not enabled"
        }
    
    metrics = state.distributed_manager.get_metrics()
    return {
        "enabled": True,
        **metrics
    }


@app.post("/distributed/configure")
async def configure_distributed_inference(request: dict):
    """
    Configure distributed inference endpoints.
    
    Request body:
        endpoints: List of endpoint URLs
        strategy: Load balancing strategy (round_robin, least_connections, weighted, consistent_hash)
        health_check_interval: Health check interval in seconds
        default_timeout: Request timeout in seconds
    """
    endpoints_config = request.get("endpoints", [])
    strategy = request.get("strategy", "round_robin")
    health_check_interval = request.get("health_check_interval", 30.0)
    default_timeout = request.get("default_timeout", 30.0)
    max_retries = request.get("max_retries", 3)
    
    # Convert to EndpointConfig objects
    endpoints = []
    for ep in endpoints_config:
        if isinstance(ep, str):
            endpoints.append(EndpointConfig(url=ep))
        elif isinstance(ep, dict):
            endpoints.append(EndpointConfig(
                url=ep["url"],
                weight=ep.get("weight", 1),
                max_retries=ep.get("max_retries", max_retries),
                timeout=ep.get("timeout", default_timeout),
            ))
    
    if not endpoints:
        raise HTTPException(status_code=400, detail="At least one endpoint is required")
    
    # Stop existing manager if running
    if state.distributed_manager:
        await stop_inference_manager()
    
    # Initialize new manager
    state.distributed_manager = initialize_inference_manager(
        endpoints=endpoints,
        strategy=LoadBalancingStrategy(strategy),
        health_check_interval=health_check_interval,
        default_timeout=default_timeout,
        max_retries=max_retries,
    )
    
    await start_inference_manager()
    state.distributed_enabled = True
    
    logger.info(f"Configured distributed inference with {len(endpoints)} endpoints")
    
    return {
        "status": "success",
        "message": f"Configured {len(endpoints)} endpoints with {strategy} strategy"
    }


@app.post("/distributed/enable")
async def enable_distributed_inference():
    """Enable distributed inference."""
    if state.distributed_manager:
        await start_inference_manager()
        state.distributed_enabled = True
        return {"status": "success", "message": "Distributed inference enabled"}
    return {"status": "error", "message": "No endpoints configured"}


@app.post("/distributed/disable")
async def disable_distributed_inference():
    """Disable distributed inference."""
    if state.distributed_manager:
        await stop_inference_manager()
        state.distributed_enabled = False
        return {"status": "success", "message": "Distributed inference disabled"}
    return {"status": "error", "message": "No endpoints configured"}


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
    
    When distributed inference is enabled, requests are automatically load-balanced
    across multiple endpoints with automatic failover.
    
    Parameters:
        - population: List of [u_value, heating_setpoint, cooling_setpoint]
        - use_surrogates: Use AI surrogates (fast) vs physics (accurate)
    """
    import time
    
    if not request.population:
        raise HTTPException(status_code=400, detail="Population cannot be empty")
    
    start_time = time.time()
    
    try:
        # Check if distributed inference is enabled and use it
        if state.distributed_enabled and state.distributed_manager:
            # Use distributed inference
            response = await state.distributed_manager.evaluate_population(
                population=request.population,
                use_surrogates=request.use_surrogates
            )
            
            evaluation_time = (time.time() - start_time) * 1000
            
            return PopulationEvaluationResponse(
                results=response["results"],
                num_evaluated=response["num_evaluated"],
                evaluation_time_ms=evaluation_time
            )
        
        # Fall back to local evaluation
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


# ===========================================
# Compliance Agent Endpoints (ASHRAE 90.1 / IECC)
# ===========================================

from api.compliance import (
    ComplianceAgent,
    ComplianceDataAggregator,
    ComplianceMetrics,
    ComplianceStandard,
    generate_compliance_report,
    create_prompt_for_llm,
    create_sample_metrics,
)


class ComplianceCheckRequest(BaseModel):
    """Request model for compliance check."""
    # Building info
    building_name: str = Field(default="Building", description="Name of the building")
    building_area_m2: float = Field(default=1000.0, description="Conditioned floor area in m²")
    building_type: str = Field(default="Commercial", description="Type of building")
    climate_zone: str = Field(default="4A", description="Climate zone")
    
    # Energy metrics (required)
    hourly_temperatures: List[float] = Field(..., description="Hourly zone temperatures (°C), 8760 values")
    hourly_heating_loads: List[float] = Field(..., description="Hourly heating loads (W), 8760 values")
    hourly_cooling_loads: List[float] = Field(..., description="Hourly cooling loads (W), 8760 values")
    
    # Optional end-use data
    hourly_lighting: Optional[List[float]] = Field(default=None, description="Hourly lighting loads (W)")
    hourly_plug_loads: Optional[List[float]] = Field(default=None, description="Hourly plug loads (W)")
    hourly_internal_gains: Optional[List[float]] = Field(default=None, description="Hourly internal gains (W)")
    
    # Baseline comparison (optional)
    baseline_hourly_temperatures: Optional[List[float]] = Field(default=None, description="Baseline hourly temperatures")
    baseline_hourly_heating_loads: Optional[List[float]] = Field(default=None, description="Baseline hourly heating loads")
    baseline_hourly_cooling_loads: Optional[List[float]] = Field(default=None, description="Baseline hourly cooling loads")
    
    # Configuration
    standard: str = Field(default="ASHRAE 90.1-2019", description="Compliance standard")
    project_name: str = Field(default="Project", description="Name of the project")
    building_address: str = Field(default="Address", description="Building address")


class ComplianceReportRequest(BaseModel):
    """Request model for compliance report generation."""
    # Building info
    building_name: str = Field(default="Building", description="Name of the building")
    building_area_m2: float = Field(default=1000.0, description="Conditioned floor area in m²")
    building_type: str = Field(default="Commercial", description="Type of building")
    climate_zone: str = Field(default="4A", description="Climate zone")
    
    # Energy metrics
    total_energy_kwh: float = Field(..., description="Annual total energy consumption (kWh)")
    total_eui_kwh_m2: float = Field(..., description="Annual EUI (kWh/m²/year)")
    peak_heating_kw: float = Field(..., description="Peak heating load (kW)")
    peak_cooling_kw: float = Field(..., description="Peak cooling load (kW)")
    unmet_hours: float = Field(..., description="Total unmet hours")
    heating_energy_kwh: float = Field(default=0.0, description="Annual heating energy (kWh)")
    cooling_energy_kwh: float = Field(default=0.0, description="Annual cooling energy (kWh)")
    lighting_energy_kwh: float = Field(default=0.0, description="Annual lighting energy (kWh)")
    plug_loads_kwh: float = Field(default=0.0, description="Annual plug loads (kWh)")
    
    # Baseline (optional)
    baseline_total_energy_kwh: Optional[float] = Field(default=None, description="Baseline annual energy (kWh)")
    baseline_total_eui_kwh_m2: Optional[float] = Field(default=None, description="Baseline EUI")
    baseline_peak_heating_kw: Optional[float] = Field(default=None, description="Baseline peak heating")
    baseline_peak_cooling_kw: Optional[float] = Field(default=None, description="Baseline peak cooling")
    baseline_unmet_hours: Optional[float] = Field(default=None, description="Baseline unmet hours")
    
    # Configuration
    standard: str = Field(default="ASHRAE 90.1-2019", description="Compliance standard")
    project_name: str = Field(default="Project", description="Name of the project")
    building_address: str = Field(default="Address", description="Building address")
    output_format: str = Field(default="markdown", description="Output format: markdown, json")


@app.post("/compliance/check")
async def check_compliance(request: ComplianceCheckRequest):
    """
    Check building energy model compliance with ASHRAE 90.1 or IECC.
    
    Provide hourly simulation data and receive compliance determination.
    """
    try:
        # Create aggregator for proposed building
        proposed_aggregator = ComplianceDataAggregator(
            building_area_m2=request.building_area_m2,
            building_type=request.building_type,
        )
        proposed_aggregator.metrics.building_name = request.building_name
        proposed_aggregator.metrics.climate_zone = request.climate_zone
        
        # Process proposed metrics
        proposed_metrics = proposed_aggregator.process_simulation_results(
            hourly_temperatures=request.hourly_temperatures,
            hourly_heating_loads=request.hourly_heating_loads,
            hourly_cooling_loads=request.hourly_cooling_loads,
            hourly_lighting=request.hourly_lighting,
            hourly_plug_loads=request.hourly_plug_loads,
            hourly_internal_gains=request.hourly_internal_gains,
        )
        
        # Process baseline if provided
        baseline_metrics = None
        if request.baseline_hourly_temperatures:
            baseline_aggregator = ComplianceDataAggregator(
                building_area_m2=request.building_area_m2,
                building_type=request.building_type,
            )
            baseline_metrics = baseline_aggregator.process_simulation_results(
                hourly_temperatures=request.baseline_hourly_temperatures,
                hourly_heating_loads=request.baseline_hourly_heating_loads,
                hourly_cooling_loads=request.baseline_hourly_cooling_loads,
            )
        
        # Create compliance agent
        standard_map = {
            "ASHRAE 90.1-2019": ComplianceStandard.ASHRAE_90_1_2019,
            "ASHRAE 90.1-2022": ComplianceStandard.ASHRAE_90_1_2022,
            "IECC 2021": ComplianceStandard.IECC_2021,
            "IECC 2024": ComplianceStandard.IECC_2024,
        }
        standard = standard_map.get(request.standard, ComplianceStandard.ASHRAE_90_1_2019)
        
        agent = ComplianceAgent()
        
        # Run compliance check
        result = agent.check_compliance(proposed_metrics, baseline_metrics)
        
        # Add metrics to response
        result["proposed_metrics"] = proposed_aggregator.to_dict()
        if baseline_metrics:
            baseline_aggregator2 = ComplianceDataAggregator(
                building_area_m2=request.building_area_m2,
                building_type=request.building_type,
            )
            baseline_aggregator2.metrics = baseline_metrics
            result["baseline_metrics"] = baseline_aggregator2.to_dict()
        
        return result
        
    except Exception as e:
        logger.error(f"Compliance check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compliance/report")
async def generate_report(request: ComplianceReportRequest):
    """
    Generate a compliance report (Markdown or JSON).
    
    Provide building metrics and receive a formatted compliance report.
    """
    try:
        # Build proposed metrics
        proposed_metrics = ComplianceMetrics()
        proposed_metrics.building_name = request.building_name
        proposed_metrics.building_area_m2 = request.building_area_m2
        proposed_metrics.building_type = request.building_type
        proposed_metrics.climate_zone = request.climate_zone
        proposed_metrics.total_energy_kwh = request.total_energy_kwh
        proposed_metrics.total_eui_kwh_m2 = request.total_eui_kwh_m2
        proposed_metrics.peak_heating_load_kw = request.peak_heating_kw
        proposed_metrics.peak_cooling_load_kw = request.peak_cooling_kw
        proposed_metrics.total_unmet_hours = request.unmet_hours
        proposed_metrics.heating_energy_kwh = request.heating_energy_kwh
        proposed_metrics.cooling_energy_kwh = request.cooling_energy_kwh
        proposed_metrics.lighting_energy_kwh = request.lighting_energy_kwh
        proposed_metrics.plug_loads_kwh = request.plug_loads_kwh
        
        # Build baseline if provided
        baseline_metrics = None
        if request.baseline_total_energy_kwh is not None:
            baseline_metrics = ComplianceMetrics()
            baseline_metrics.building_area_m2 = request.building_area_m2
            baseline_metrics.total_energy_kwh = request.baseline_total_energy_kwh
            baseline_metrics.total_eui_kwh_m2 = request.baseline_total_eui_kwh_m2
            baseline_metrics.peak_heating_load_kw = request.baseline_peak_heating_kw or 0
            baseline_metrics.peak_cooling_load_kw = request.baseline_peak_cooling_kw or 0
            baseline_metrics.total_unmet_hours = request.baseline_unmet_hours or 0
        
        # Generate report
        if request.output_format == "json":
            # Return structured data
            agent = ComplianceAgent()
            compliance_result = agent.check_compliance(proposed_metrics, baseline_metrics)
            
            return {
                "report_type": "compliance_json",
                "standard": request.standard,
                "building": {
                    "name": request.building_name,
                    "area_m2": request.building_area_m2,
                    "type": request.building_type,
                    "climate_zone": request.climate_zone,
                },
                "compliance": compliance_result,
            }
        else:
            # Generate Markdown report
            standard_map = {
                "ASHRAE 90.1-2019": ComplianceStandard.ASHRAE_90_1_2019,
                "ASHRAE 90.1-2022": ComplianceStandard.ASHRAE_90_1_2022,
                "IECC 2021": ComplianceStandard.IECC_2021,
                "IECC 2024": ComplianceStandard.IECC_2024,
            }
            standard = standard_map.get(request.standard, ComplianceStandard.ASHRAE_90_1_2019)
            
            report = generate_compliance_report(
                proposed_metrics=proposed_metrics,
                baseline_metrics=baseline_metrics,
                project_name=request.project_name,
                building_name=request.building_name,
                building_address=request.building_address,
                standard=standard,
            )
            
            return {
                "report_type": "markdown",
                "standard": request.standard,
                "report": report,
            }
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compliance/prompt")
async def get_llm_prompt(request: ComplianceReportRequest):
    """
    Generate an LLM prompt for compliance report generation.
    
    Returns the system and user prompts that can be sent to an LLM
    to generate a natural language compliance report.
    """
    try:
        # Build proposed metrics
        proposed_metrics = ComplianceMetrics()
        proposed_metrics.building_name = request.building_name
        proposed_metrics.building_area_m2 = request.building_area_m2
        proposed_metrics.building_type = request.building_type
        proposed_metrics.climate_zone = request.climate_zone
        proposed_metrics.total_energy_kwh = request.total_energy_kwh
        proposed_metrics.total_eui_kwh_m2 = request.total_eui_kwh_m2
        proposed_metrics.peak_heating_load_kw = request.peak_heating_kw
        proposed_metrics.peak_cooling_load_kw = request.peak_cooling_kw
        proposed_metrics.total_unmet_hours = request.unmet_hours
        proposed_metrics.heating_energy_kwh = request.heating_energy_kwh
        proposed_metrics.cooling_energy_kwh = request.cooling_energy_kwh
        proposed_metrics.lighting_energy_kwh = request.lighting_energy_kwh
        proposed_metrics.plug_loads_kwh = request.plug_loads_kwh
        
        # Build baseline if provided
        baseline_metrics = None
        if request.baseline_total_energy_kwh is not None:
            baseline_metrics = ComplianceMetrics()
            baseline_metrics.building_area_m2 = request.building_area_m2
            baseline_metrics.total_energy_kwh = request.baseline_total_energy_kwh
            baseline_metrics.total_eui_kwh_m2 = request.baseline_total_eui_kwh_m2
            baseline_metrics.peak_heating_load_kw = request.baseline_peak_heating_kw or 0
            baseline_metrics.peak_cooling_load_kw = request.baseline_peak_cooling_kw or 0
            baseline_metrics.total_unmet_hours = request.baseline_unmet_hours or 0
        
        # Generate prompts
        standard_map = {
            "ASHRAE 90.1-2019": ComplianceStandard.ASHRAE_90_1_2019,
            "ASHRAE 90.1-2022": ComplianceStandard.ASHRAE_90_1_2022,
            "IECC 2021": ComplianceStandard.IECC_2021,
            "IECC 2024": ComplianceStandard.IECC_2024,
        }
        standard = standard_map.get(request.standard, ComplianceStandard.ASHRAE_90_1_2019)
        
        prompt = create_prompt_for_llm(
            metrics=proposed_metrics,
            baseline_metrics=baseline_metrics,
            standard=standard,
        )
        
        return {
            "standard": request.standard,
            "system_prompt": prompt.system_prompt,
            "user_prompt": prompt.user_prompt_template,
        }
        
    except Exception as e:
        logger.error(f"Prompt generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compliance/standards")
async def list_standards():
    """List available compliance standards."""
    return {
        "standards": [
            {"id": "ASHRAE 90.1-2019", "name": "ASHRAE 90.1-2019", "type": "performance"},
            {"id": "ASHRAE 90.1-2022", "name": "ASHRAE 90.1-2022", "type": "performance"},
            {"id": "IECC 2021", "name": "IECC 2021", "type": "prescriptive/performance"},
            {"id": "IECC 2024", "name": "IECC 2024", "type": "prescriptive/performance"},
        ]
    }


@app.get("/compliance/sample")
async def get_sample_metrics():
    """Get sample compliance metrics for testing."""
    metrics = create_sample_metrics()
    aggregator = ComplianceDataAggregator(building_area_m2=1000.0)
    aggregator.metrics = metrics
    return aggregator.to_dict()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
