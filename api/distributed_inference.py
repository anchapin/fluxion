"""
Distributed Inference Architecture for AI Surrogates

This module provides a distributed inference system with:
- Multiple inference endpoints management
- Load balancing strategies (round-robin, least connections, weighted)
- Health checking and automatic failover
- Connection pooling and request distribution

Designed for high-availability and scalability in production environments.
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(str, Enum):
    """Supported load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    CONSISTENT_HASH = "consistent_hash"


class EndpointStatus(str, Enum):
    """Endpoint health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class EndpointConfig:
    """Configuration for a single inference endpoint."""
    url: str
    weight: int = 1  # Weight for weighted load balancing
    max_retries: int = 3
    timeout: float = 30.0
    health_check_interval: float = 30.0  # seconds
    failure_threshold: int = 3  # consecutive failures before marking unhealthy
    recovery_threshold: int = 2  # consecutive successes before marking healthy


@dataclass
class Endpoint:
    """Represents a single inference endpoint with health tracking."""
    config: EndpointConfig
    status: EndpointStatus = EndpointStatus.UNKNOWN
    current_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_health_check: float = 0.0
    average_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    
    @property
    def base_url(self) -> str:
        """Extract base URL from full URL."""
        parsed = urlparse(self.config.url)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    @property
    def is_healthy(self) -> bool:
        """Check if endpoint is healthy."""
        return self.status in (EndpointStatus.HEALTHY, EndpointStatus.DEGRADED)
    
    def update_response_time(self, response_time: float):
        """Update response time metrics."""
        self.response_times.append(response_time)
        # Keep last 100 response times
        if len(self.response_times) > 100:
            self.response_times.pop(0)
        # Calculate moving average
        self.average_response_time = sum(self.response_times) / len(self.response_times)


class HealthCheckProtocol(Protocol):
    """Protocol for custom health check implementations."""
    async def check_health(self, endpoint: Endpoint) -> bool:
        """Check if endpoint is healthy."""
        ...


class DefaultHealthCheck:
    """Default health check implementation using /health endpoint."""
    
    def __init__(self, health_path: str = "/health"):
        self.health_path = health_path
    
    async def check_health(self, endpoint: Endpoint) -> bool:
        """Check endpoint health via HTTP request."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{endpoint.base_url}{self.health_path}")
                return response.status_code == 200
        except Exception:
            return False


class LoadBalancer(ABC):
    """Abstract base class for load balancers."""
    
    @abstractmethod
    def select_endpoint(self, endpoints: List[Endpoint]) -> Optional[Endpoint]:
        """Select an endpoint based on the load balancing strategy."""
        pass
    
    @abstractmethod
    def record_success(self, endpoint: Endpoint):
        """Record a successful request."""
        pass
    
    @abstractmethod
    def record_failure(self, endpoint: Endpoint):
        """Record a failed request."""
        pass


class RoundRobinLoadBalancer(LoadBalancer):
    """Round-robin load balancing strategy."""
    
    def __init__(self):
        self._current_index: Dict[str, int] = {}  # key -> index
    
    def select_endpoint(self, endpoints: List[Endpoint]) -> Optional[Endpoint]:
        """Select next endpoint in round-robin fashion."""
        healthy = [e for e in endpoints if e.is_healthy]
        if not healthy:
            return None
        
        # Get or create counter for this endpoint list
        key = id(endpoints)
        if key not in self._current_index:
            self._current_index[key] = 0
        
        # Select endpoint and advance index
        index = self._current_index[key] % len(healthy)
        selected = healthy[index]
        self._current_index[key] = index + 1
        
        return selected
    
    def record_success(self, endpoint: Endpoint):
        """Record successful request."""
        pass
    
    def record_failure(self, endpoint: Endpoint):
        """Record failed request."""
        pass


class LeastConnectionsLoadBalancer(LoadBalancer):
    """Least connections load balancing strategy."""
    
    def select_endpoint(self, endpoints: List[Endpoint]) -> Optional[Endpoint]:
        """Select endpoint with fewest active connections."""
        healthy = [e for e in endpoints if e.is_healthy]
        if not healthy:
            return None
        
        return min(healthy, key=lambda e: e.current_connections)
    
    def record_success(self, endpoint: Endpoint):
        """Record successful request."""
        endpoint.current_connections = max(0, endpoint.current_connections - 1)
    
    def record_failure(self, endpoint: Endpoint):
        """Record failed request."""
        endpoint.current_connections = max(0, endpoint.current_connections - 1)


class WeightedLoadBalancer(LoadBalancer):
    """Weighted load balancing strategy."""
    
    def __init__(self):
        self._current_index: Dict[str, int] = {}
    
    def select_endpoint(self, endpoints: List[Endpoint]) -> Optional[Endpoint]:
        """Select endpoint based on weight and round-robin."""
        healthy = [e for e in endpoints if e.is_healthy]
        if not healthy:
            return None
        
        # Build weighted list
        weighted_list = []
        for endpoint in healthy:
            weighted_list.extend([endpoint] * endpoint.config.weight)
        
        if not weighted_list:
            return None
        
        # Get or create counter
        key = id(endpoints)
        if key not in self._current_index:
            self._current_index[key] = 0
        
        # Select endpoint and advance index
        index = self._current_index[key] % len(weighted_list)
        selected = weighted_list[index]
        self._current_index[key] = index + 1
        
        return selected
    
    def record_success(self, endpoint: Endpoint):
        """Record successful request."""
        pass
    
    def record_failure(self, endpoint: Endpoint):
        """Record failed request."""
        pass


class ConsistentHashLoadBalancer(LoadBalancer):
    """Consistent hash load balancing for request affinity."""
    
    def __init__(self, virtual_nodes: int = 150):
        self._hash_ring: Dict[int, Endpoint] = {}
        self._sorted_keys: List[int] = []
        self._virtual_nodes = virtual_nodes
    
    def _hash(self, key: str) -> int:
        """Hash function for consistent hashing."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def _build_ring(self, endpoints: List[Endpoint]):
        """Build the hash ring from endpoints."""
        self._hash_ring.clear()
        self._sorted_keys.clear()
        
        for endpoint in endpoints:
            if not endpoint.is_healthy:
                continue
            # Add virtual nodes for better distribution
            for i in range(self._virtual_nodes):
                hash_key = self._hash(f"{endpoint.config.url}:{i}")
                self._hash_ring[hash_key] = endpoint
        
        self._sorted_keys = sorted(self._hash_ring.keys())
    
    def select_endpoint(self, endpoints: List[Endpoint], request_key: str = "") -> Optional[Endpoint]:
        """Select endpoint using consistent hashing."""
        self._build_ring(endpoints)
        
        if not self._sorted_keys:
            return None
        
        # Hash the request key
        if request_key:
            hash_value = self._hash(request_key)
        else:
            hash_value = self._hash(str(time.time()))
        
        # Find the first endpoint in the ring greater than or equal to hash
        for key in self._sorted_keys:
            if key >= hash_value:
                return self._hash_ring[key]
        
        # Wrap around to first endpoint
        return self._hash_ring[self._sorted_keys[0]]
    
    def record_success(self, endpoint: Endpoint):
        """Record successful request."""
        pass
    
    def record_failure(self, endpoint: Endpoint):
        """Record failed request."""
        pass


class DistributedInferenceManager:
    """
    Main class for managing distributed inference across multiple endpoints.
    
    Features:
    - Multiple inference endpoints
    - Configurable load balancing strategies
    - Automatic health checking
    - Automatic failover on endpoint failure
    - Request retry with exponential backoff
    - Metrics collection
    """
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        health_check: Optional[HealthCheckProtocol] = None,
        health_check_interval: float = 30.0,
        default_timeout: float = 30.0,
        max_retries: int = 3
    ):
        self._endpoints: Dict[str, Endpoint] = {}
        self._strategy = strategy
        self._load_balancer = self._create_load_balancer(strategy)
        self._health_check = health_check or DefaultHealthCheck()
        self._health_check_interval = health_check_interval
        self._default_timeout = default_timeout
        self._max_retries = max_retries
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Metrics
        self.total_requests = 0
        self.total_failures = 0
    
    def _create_load_balancer(self, strategy: LoadBalancingStrategy) -> LoadBalancer:
        """Create a load balancer based on strategy."""
        balancers = {
            LoadBalancingStrategy.ROUND_ROBIN: RoundRobinLoadBalancer,
            LoadBalancingStrategy.LEAST_CONNECTIONS: LeastConnectionsLoadBalancer,
            LoadBalancingStrategy.WEIGHTED: WeightedLoadBalancer,
            LoadBalancingStrategy.CONSISTENT_HASH: ConsistentHashLoadBalancer,
        }
        return balancers[strategy]()
    
    async def start(self):
        """Start the distributed inference manager."""
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Distributed inference manager started")
    
    async def stop(self):
        """Stop the distributed inference manager."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Distributed inference manager stopped")
    
    async def _health_check_loop(self):
        """Periodic health check loop."""
        while self._running:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._check_all_endpoints()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _check_all_endpoints(self):
        """Check health of all endpoints."""
        async with self._lock:
            for endpoint in self._endpoints.values():
                await self._check_endpoint_health(endpoint)
    
    async def _check_endpoint_health(self, endpoint: Endpoint):
        """Check health of a single endpoint."""
        is_healthy = await self._health_check.check_health(endpoint)
        endpoint.last_health_check = time.time()
        
        if is_healthy:
            endpoint.consecutive_successes += 1
            endpoint.consecutive_failures = 0
            
            if (endpoint.consecutive_successes >= endpoint.config.recovery_threshold 
                and endpoint.status == EndpointStatus.UNHEALTHY):
                endpoint.status = EndpointStatus.DEGRADED
                logger.info(f"Endpoint {endpoint.config.url} recovered to degraded state")
            
            if endpoint.status == EndpointStatus.UNKNOWN:
                endpoint.status = EndpointStatus.HEALTHY
                logger.info(f"Endpoint {endpoint.config.url} is now healthy")
        else:
            endpoint.consecutive_failures += 1
            endpoint.consecutive_successes = 0
            
            if endpoint.consecutive_failures >= endpoint.config.failure_threshold:
                if endpoint.status != EndpointStatus.UNHEALTHY:
                    endpoint.status = EndpointStatus.UNHEALTHY
                    logger.warning(f"Endpoint {endpoint.config.url} marked as unhealthy")
    
    def add_endpoint(self, config: EndpointConfig):
        """Add an inference endpoint."""
        endpoint = Endpoint(config=config)
        self._endpoints[config.url] = endpoint
        logger.info(f"Added endpoint: {config.url} (weight: {config.weight})")
    
    def remove_endpoint(self, url: str):
        """Remove an inference endpoint."""
        if url in self._endpoints:
            del self._endpoints[url]
            logger.info(f"Removed endpoint: {url}")
    
    def get_endpoint(self, url: str) -> Optional[Endpoint]:
        """Get an endpoint by URL."""
        return self._endpoints.get(url)
    
    def list_endpoints(self) -> List[Endpoint]:
        """List all endpoints."""
        return list(self._endpoints.values())
    
    def list_healthy_endpoints(self) -> List[Endpoint]:
        """List all healthy endpoints."""
        return [e for e in self._endpoints.values() if e.is_healthy]
    
    async def _select_endpoint(self) -> Optional[Endpoint]:
        """Select an endpoint using the load balancer."""
        healthy = self.list_healthy_endpoints()
        return self._load_balancer.select_endpoint(healthy)
    
    async def _execute_request(
        self,
        endpoint: Endpoint,
        method: str,
        path: str,
        **kwargs
    ) -> httpx.Response:
        """Execute a request to an endpoint with retry logic."""
        url = f"{endpoint.config.url.rstrip('/')}{path}"
        
        async with self._lock:
            endpoint.current_connections += 1
        
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=endpoint.config.timeout) as client:
                response = await client.request(method, url, **kwargs)
            
            response_time = time.time() - start_time
            endpoint.update_response_time(response_time)
            
            async with self._lock:
                endpoint.total_requests += 1
                endpoint.successful_requests += 1
                endpoint.current_connections = max(0, endpoint.current_connections - 1)
            
            self._load_balancer.record_success(endpoint)
            
            return response
            
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            async with self._lock:
                endpoint.failed_requests += 1
                endpoint.current_connections = max(0, endpoint.current_connections - 1)
                endpoint.consecutive_failures += 1
            
            self._load_balancer.record_failure(endpoint)
            
            # Check if should mark as unhealthy
            if endpoint.consecutive_failures >= endpoint.config.failure_threshold:
                async with self._lock:
                    endpoint.status = EndpointStatus.UNHEALTHY
            
            raise
    
    async def request(
        self,
        method: str,
        path: str,
        retries: Optional[int] = None,
        request_key: str = "",  # For consistent hash
        **kwargs
    ) -> httpx.Response:
        """
        Make a distributed request with automatic failover.
        
        Args:
            method: HTTP method
            path: Request path
            retries: Number of retries (default: max_retries)
            request_key: Key for consistent hash routing
            **kwargs: Additional arguments passed to httpx request
        
        Returns:
            httpx.Response: Response from the endpoint
        
        Raises:
            httpx.HTTPError: If all endpoints fail
        """
        max_retries = retries if retries is not None else self._max_retries
        last_error = None
        
        for attempt in range(max_retries + 1):
            endpoint = await self._select_endpoint()
            
            if endpoint is None:
                if attempt == max_retries:
                    raise httpx.HTTPError("No healthy endpoints available")
                await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                continue
            
            self.total_requests += 1
            
            try:
                response = await self._execute_request(endpoint, method, path, **kwargs)
                return response
                
            except Exception as e:
                last_error = e
                self.total_failures += 1
                logger.warning(f"Request to {endpoint.config.url} failed (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries:
                    # Exponential backoff
                    await asyncio.sleep(0.1 * (2 ** attempt))
        
        raise httpx.HTTPError(f"All endpoints failed after {max_retries + 1} attempts: {last_error}")
    
    async def get(self, path: str, **kwargs) -> httpx.Response:
        """Make a GET request."""
        return await self.request("GET", path, **kwargs)
    
    async def post(self, path: str, **kwargs) -> httpx.Response:
        """Make a POST request."""
        return await self.request("POST", path, **kwargs)
    
    async def evaluate_population(
        self,
        population: List[List[float]],
        use_surrogates: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a population of building designs using distributed inference.
        
        Args:
            population: List of parameter vectors
            use_surrogates: Use AI surrogates for evaluation
        
        Returns:
            Dictionary with results and metadata
        """
        response = await self.post(
            "/evaluate",
            json={
                "population": population,
                "use_surrogates": use_surrogates
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def health_check(self) -> Dict[str, Any]:
        """Get health status of all endpoints."""
        endpoints_status = []
        
        for endpoint in self.list_endpoints():
            endpoints_status.append({
                "url": endpoint.config.url,
                "status": endpoint.status.value,
                "current_connections": endpoint.current_connections,
                "total_requests": endpoint.total_requests,
                "successful_requests": endpoint.successful_requests,
                "failed_requests": endpoint.failed_requests,
                "average_response_time_ms": endpoint.average_response_time * 1000,
                "last_health_check": endpoint.last_health_check
            })
        
        healthy_count = len([e for e in self._endpoints.values() if e.is_healthy])
        
        return {
            "total_endpoints": len(self._endpoints),
            "healthy_endpoints": healthy_count,
            "strategy": self._strategy.value,
            "endpoints": endpoints_status
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get overall metrics."""
        return {
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "success_rate": (
                (self.total_requests - self.total_failures) / self.total_requests * 100
                if self.total_requests > 0 else 0
            ),
            "endpoints": {
                url: {
                    "status": ep.status.value,
                    "total_requests": ep.total_requests,
                    "successful_requests": ep.successful_requests,
                    "failed_requests": ep.failed_requests,
                    "average_response_time_ms": ep.average_response_time * 1000
                }
                for url, ep in self._endpoints.items()
            }
        }


# Global instance
_inference_manager: Optional[DistributedInferenceManager] = None


def get_inference_manager() -> Optional[DistributedInferenceManager]:
    """Get the global inference manager instance."""
    return _inference_manager


def initialize_inference_manager(
    endpoints: List[EndpointConfig],
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
    health_check_interval: float = 30.0,
    default_timeout: float = 30.0,
    max_retries: int = 3
) -> DistributedInferenceManager:
    """
    Initialize the global inference manager.
    
    Args:
        endpoints: List of endpoint configurations
        strategy: Load balancing strategy
        health_check_interval: Health check interval in seconds
        default_timeout: Default request timeout
        max_retries: Maximum number of retries per request
    
    Returns:
        DistributedInferenceManager: The initialized manager
    """
    global _inference_manager
    
    _inference_manager = DistributedInferenceManager(
        strategy=strategy,
        health_check_interval=health_check_interval,
        default_timeout=default_timeout,
        max_retries=max_retries
    )
    
    for config in endpoints:
        _inference_manager.add_endpoint(config)
    
    return _inference_manager


async def start_inference_manager():
    """Start the global inference manager."""
    global _inference_manager
    if _inference_manager:
        await _inference_manager.start()


async def stop_inference_manager():
    """Stop the global inference manager."""
    global _inference_manager
    if _inference_manager:
        await _inference_manager.stop()
