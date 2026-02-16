# Fluxion REST API Server

A production-ready REST API server for evaluating building design populations remotely using Fluxion.

## Features

- **Population Evaluation**: Evaluate thousands of building configurations in parallel
- **Surrogate Model Support**: Load ONNX surrogate models for fast inference
- **Health Monitoring**: Built-in health checks and status endpoints
- **CORS Enabled**: Easy integration with web frontends

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install from PyPI (when available)
pip install fluxion-api
```

### Running the Server

```bash
# Start the API server
python -m api.main

# Or with uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build and run with Docker
docker build -t fluxion-api .
docker run -p 8000:8000 fluxion-api
```

## API Endpoints

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/` | GET | Root endpoint, returns health status |
| `/health` | GET | Health check for monitoring |
| `/status` | GET | Get current model status |
| `/load-surrogate` | POST | Load an ONNX surrogate model |
| `/unload-surrogate` | POST | Unload the current model |
| `/evaluate` | POST | Evaluate population of designs |

## Example Usage

### Evaluate Population

```python
import requests

# Define population of building designs
population = [
    [1.5, 20.0, 27.0],  # [u_value, heating_setpoint, cooling_setpoint]
    [2.0, 21.0, 26.0],
    [1.0, 19.0, 28.0],
]

response = requests.post("http://localhost:8000/evaluate", json={
    "population": population,
    "use_surrogates": True
})

results = response.json()
print(f"Evaluated {results['num_evaluated']} designs in {results['evaluation_time_ms']:.2f}ms")
print(f"Results (EUI): {results['results']}")
```

### Using curl

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "population": [[1.5, 20.0, 27.0], [2.0, 21.0, 26.0]],
    "use_surrogates": true
  }'
```

## API Specification

The full OpenAPI specification is available in `openapi.yaml`. You can use it to:

- Generate client SDKs
- Validate API responses
- Auto-generate documentation

```bash
# Generate Python client
openapi-generator generate -i openapi.yaml -g python -o client

# Generate TypeScript client
openapi-generator generate -i openapi.yaml -g typescript-axios -o client
```

## Parameter Format

Each parameter vector should contain:

| Index | Parameter | Unit | Valid Range |
|-------|-----------|------|-------------|
| 0 | Window U-value | W/m²K | 0.1 - 5.0 |
| 1 | Heating Setpoint | °C | 15 - 25 |
| 2 | Cooling Setpoint | °C | 22 - 32 |

Note: Heating setpoint must be less than cooling setpoint.

## Response Format

```json
{
  "results": [105.5, 98.2, 112.3],
  "num_evaluated": 3,
  "evaluation_time_ms": 150.5
}
```

- `results`: List of EUI values (kWh/m²/year) for each candidate
- `num_evaluated`: Number of configurations evaluated
- `evaluation_time_ms`: Evaluation time in milliseconds

## License

MIT License - see [LICENSE](../LICENSE)
