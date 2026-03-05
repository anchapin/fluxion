"""
Configuration Management for Distributed Inference

Provides configuration loading and validation for distributed inference endpoints.
Supports loading from YAML, JSON, or environment variables.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from api.distributed_inference import (
    DistributedInferenceManager,
    EndpointConfig,
    LoadBalancingStrategy,
    initialize_inference_manager,
)

logger = logging.getLogger(__name__)


@dataclass
class DistributedInferenceConfig:
    """Configuration for distributed inference system."""
    endpoints: List[EndpointConfig] = field(default_factory=list)
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    health_check_interval: float = 30.0
    default_timeout: float = 30.0
    max_retries: int = 3
    enabled: bool = True


def load_endpoint_config(config: Dict[str, Any]) -> EndpointConfig:
    """Load endpoint configuration from dictionary."""
    return EndpointConfig(
        url=config["url"],
        weight=config.get("weight", 1),
        max_retries=config.get("max_retries", 3),
        timeout=config.get("timeout", 30.0),
        health_check_interval=config.get("health_check_interval", 30.0),
        failure_threshold=config.get("failure_threshold", 3),
        recovery_threshold=config.get("recovery_threshold", 2),
    )


def load_from_dict(config: Dict[str, Any]) -> DistributedInferenceConfig:
    """Load configuration from dictionary."""
    endpoints = [load_endpoint_config(ep) for ep in config.get("endpoints", [])]
    
    strategy = LoadBalancingStrategy(config.get("strategy", "round_robin"))
    
    return DistributedInferenceConfig(
        endpoints=endpoints,
        strategy=strategy,
        health_check_interval=config.get("health_check_interval", 30.0),
        default_timeout=config.get("default_timeout", 30.0),
        max_retries=config.get("max_retries", 3),
        enabled=config.get("enabled", True),
    )


def load_from_yaml(file_path: Union[str, Path]) -> DistributedInferenceConfig:
    """Load configuration from YAML file."""
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return load_from_dict(config)


def load_from_json(file_path: Union[str, Path]) -> DistributedInferenceConfig:
    """Load configuration from JSON file."""
    import json
    with open(file_path, "r") as f:
        config = json.load(f)
    return load_from_dict(config)


def load_from_env() -> DistributedInferenceConfig:
    """
    Load configuration from environment variables.
    
    Environment variables:
    - FLUXION_DISTRIBUTED_ENABLED: Enable distributed inference (default: true)
    - FLUXION_DISTRIBUTED_STRATEGY: Load balancing strategy
    - FLUXION_ENDPOINTS: Comma-separated list of endpoint URLs
    - FLUXION_ENDPOINT_WEIGHTS: Comma-separated weights (optional)
    - FLUXION_HEALTH_CHECK_INTERVAL: Health check interval in seconds
    - FLUXION_DEFAULT_TIMEOUT: Default request timeout
    - FLUXION_MAX_RETRIES: Maximum retries per request
    """
    enabled = os.getenv("FLUXION_DISTRIBUTED_ENABLED", "true").lower() == "true"
    
    strategy = LoadBalancingStrategy(
        os.getenv("FLUXION_DISTRIBUTED_STRATEGY", "round_robin")
    )
    
    endpoints_str = os.getenv("FLUXION_ENDPOINTS", "")
    endpoints = []
    
    if endpoints_str:
        urls = [url.strip() for url in endpoints_str.split(",")]
        weights_str = os.getenv("FLUXION_ENDPOINT_WEIGHTS", "")
        
        if weights_str:
            weights = [int(w.strip()) for w in weights_str.split(",")]
        else:
            weights = [1] * len(urls)
        
        for url, weight in zip(urls, weights):
            endpoints.append(EndpointConfig(url=url, weight=weight))
    
    health_check_interval = float(os.getenv("FLUXION_HEALTH_CHECK_INTERVAL", "30.0"))
    default_timeout = float(os.getenv("FLUXION_DEFAULT_TIMEOUT", "30.0"))
    max_retries = int(os.getenv("FLUXION_MAX_RETRIES", "3"))
    
    return DistributedInferenceConfig(
        endpoints=endpoints,
        strategy=strategy,
        health_check_interval=health_check_interval,
        default_timeout=default_timeout,
        max_retries=max_retries,
        enabled=enabled,
    )


def auto_load(
    config_path: Optional[Union[str, Path]] = None,
    config_format: Optional[str] = None,
) -> DistributedInferenceConfig:
    """
    Automatically load configuration from file or environment.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        config_format: Format of config file ('yaml' or 'json'), auto-detected if None
    
    Returns:
        DistributedInferenceConfig: Loaded configuration
    """
    # Try loading from file first
    if config_path:
        path = Path(config_path)
        
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}")
        else:
            if config_format == "yaml" or path.suffix in (".yaml", ".yml"):
                return load_from_yaml(path)
            elif config_format == "json" or path.suffix == ".json":
                return load_from_json(path)
            else:
                # Auto-detect based on extension
                if path.suffix in (".yaml", ".yml"):
                    return load_from_yaml(path)
                elif path.suffix == ".json":
                    return load_from_json(path)
    
    # Fall back to environment variables
    return load_from_env()


def initialize_from_config(
    config: DistributedInferenceConfig,
) -> Optional[DistributedInferenceManager]:
    """
    Initialize the distributed inference manager from configuration.
    
    Args:
        config: Distributed inference configuration
    
    Returns:
        DistributedInferenceManager or None if no endpoints configured
    """
    if not config.enabled:
        logger.info("Distributed inference is disabled")
        return None
    
    if not config.endpoints:
        logger.warning("No endpoints configured for distributed inference")
        return None
    
    manager = initialize_inference_manager(
        endpoints=config.endpoints,
        strategy=config.strategy,
        health_check_interval=config.health_check_interval,
        default_timeout=config.default_timeout,
        max_retries=config.max_retries,
    )
    
    logger.info(
        f"Initialized distributed inference with {len(config.endpoints)} endpoints, "
        f"strategy: {config.strategy.value}"
    )
    
    return manager
