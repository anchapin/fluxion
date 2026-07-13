# Performance Validation User Guide

User guide for Fluxion's performance validation capabilities — run and interpret performance tests.
BEM engineers and developers — validate models, optimize simulations, ensure compliance.
Covers: CLI commands (fluxion performance validate/report/integrated), key metrics (timestep <50ms).
Companion to documentation/performance.md (benchmarks, optimization targets, targets table).
Status: Active — integrated with CI via performance validation framework.
Action: Run `fluxion performance integrated` before submitting PRs affecting simulation speed.

## Introduction

This guide helps users understand and utilize Fluxion's performance validation capabilities. Whether you're validating building energy models, optimizing simulations, or ensuring compliance with performance standards, this guide provides practical information.

## Getting Started

### Prerequisites

- Fluxion installed and working
- Basic understanding of building energy modeling
- Familiarity with command-line interfaces

### Quick Start

```bash
# Run performance validation
fluxion performance validate

# Generate performance report
fluxion performance report --output my_report.json

# Run integrated validation
fluxion performance integrated
```

## Performance Validation Concepts

### Key Metrics

| Metric | Description | Target Value |
|--------|-------------|---------------|
| Timestep Duration | Time to compute one simulation timestep | <50ms |
| Memory Usage | Memory allocated during simulation | <10MB |
| Solver Iterations | Number of solver iterations per timestep | <20 |
| Throughput | Timesteps processed per second | >20 |

### Validation Levels

1. **Basic Validation**: Checks individual performance metrics
2. **Comparative Validation**: Compares different configurations
3. **Integrated Validation**: Combines standard and performance validation
4. **Final Validation**: Comprehensive validation with recommendations

## Using the CLI

### Basic Commands

```bash
# Show performance command help
fluxion performance --help

# Run performance benchmarks
fluxion performance benchmark

# Validate performance
fluxion performance validate

# Generate report
fluxion performance report
```

### Advanced Usage

```bash
# Validate with custom threshold
fluxion performance validate --threshold 10.0

# Compare with baseline
fluxion performance validate --baseline baseline.json

# Run specific scenario
fluxion performance benchmark --scenario multi-zone

# Integrated validation with detailed output
fluxion performance integrated --detailed
```

### ASHRAE 140 Validation

```bash
# Validate ASHRAE 140 Case 900 performance
fluxion performance ashrae140 --case 900

# Save ASHRAE 140 report
fluxion performance ashrae140 --case 900 --output ashrae_report.json
```

## Programmatic Usage

### Rust API

Add to your Cargo.toml:
```toml
[dependencies]
fluxion = "1.2"
```

Basic example:
```rust
use fluxion::validation::performance::PerformanceValidator;
use fluxion::thermal::ThermalModelConfig;

fn validate_performance() -> Result<(), Box<dyn std::error::Error>> {
    // Create thermal model
    let config = ThermalModelConfig::standard();
    let model = fluxion::thermal::ThermalModel::new(config);

    // Create validator
    let validator = PerformanceValidator::new(model);

    // Run validation
    let report = validator.validate_performance()?;

    println!("Performance: {:.2}ms/timestep", report.metrics.timestep_duration_ms);

    Ok(())
}
```

### Python API (coming soon)

```python
# Future Python API
import fluxion

model = fluxion.ThermalModel.standard()
validator = fluxion.PerformanceValidator(model)
report = validator.validate()
print(f"Performance: {report.timestep_ms:.2f}ms")
```

## Performance Optimization

### Optimization Techniques

#### Solver Optimization

```rust
use fluxion::thermal::SolverConfig;

let mut config = SolverConfig::default();
config.set_tolerance(1e-6);  // More precise convergence
config.enable_warm_start(true); // Use previous solution as initial guess
```

#### Memory Optimization

```rust
use fluxion::validation::performance::MemoryOptimizer;

let mut optimizer = MemoryOptimizer::new();
optimizer.enable_caching(true); // Cache material properties
optimizer.set_pool_size(100); // Object pooling
```

#### Parallel Processing

```rust
use fluxion::validation::ParallelValidator;

let validator = ParallelValidator::new(4); // 4 parallel workers
let reports = validator.validate_multiple(configs);
```

### Common Optimization Scenarios

| Scenario | Optimization Technique | Expected Improvement |
|----------|------------------------|---------------------|
| Single-zone | Solver tuning | 10-20% |
| Multi-zone (10) | Parallel processing | 30-50% |
| High-mass | Memory caching | 15-25% |
| Peak load | Adaptive convergence | 20-30% |

## Performance Monitoring

### CI/CD Integration

Add performance validation to your GitHub Actions:

```yaml
- name: Performance Validation
  run: fluxion performance validate

- name: Performance Regression Check
  run: fluxion performance validate --baseline baseline.json
```

### Monitoring Performance Trends

```bash
# Generate historical report
fluxion performance report --historical

# Track performance over time
fluxion performance trend --days 30
```

## Troubleshooting

### Performance Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Slow timestep | >50ms duration | Check solver configuration, reduce zone count |
| High memory | >10MB usage | Enable caching, review material properties |
| Many iterations | >20 iterations | Adjust solver tolerance, improve initial guess |
| Regression | Performance worse than baseline | Review recent changes, check configuration |

### Debugging Commands

```bash
# Verbose performance output
RUST_LOG=debug fluxion performance validate

# Performance profiling
fluxion performance benchmark --profile

# Detailed report
fluxion performance report --detailed
```

## Best Practices

### Validation Workflow

1. **Establish Baseline**: Run initial performance validation
2. **Make Changes**: Modify configuration or code
3. **Validate Changes**: Run performance validation again
4. **Compare Results**: Use comparative analysis
5. **Document Findings**: Record performance improvements

### Performance Targets

- **Single-zone**: <40ms timestep, <8MB memory
- **Multi-zone (10)**: <80ms timestep, <15MB memory
- **High-mass**: <60ms timestep, <12MB memory
- **ASHRAE 140**: <100ms timestep for compliance

### Configuration Management

```bash
# Save configuration
fluxion config save --name high-performance

# Load configuration
fluxion config load high-performance

# Compare configurations
fluxion performance compare config1 config2
```

## Examples

See `examples/performance_example.rs` for comprehensive examples including:

- Basic performance validation
- Configuration comparison
- Integrated validation
- Performance reporting
- ASHRAE 140 compliance checking

Run the examples:

```bash
cd examples
cargo run --example performance_example
```

## Advanced Topics

### Custom Performance Metrics

Extend performance validation with custom metrics:

```rust
use fluxion::validation::performance::CustomMetric;

let custom_metric = CustomMetric::new(
    "custom_metric".to_string(),
    || {
        // Your custom measurement logic
        42.0
    }
);

validator.add_custom_metric(custom_metric);
```

### Performance Profiles

Create and use performance profiles:

```bash
# Create profile
fluxion profile create --name fast-simulation --solver optimized --memory-cache true

# Use profile
fluxion simulate --profile fast-simulation

# Validate with profile
fluxion performance validate --profile fast-simulation
```

### Batch Validation

Validate multiple configurations in batch:

```bash
# Validate all standard configurations
fluxion performance batch --configs standard,high-mass,peak-load

# Generate batch report
fluxion performance batch --configs standard,high-mass --output batch_report.json
```

## Support

### Getting Help

```bash
# Show help for specific command
fluxion performance validate --help

# Show version information
fluxion --version

# Check for updates
fluxion update check
```

### Reporting Issues

When reporting performance issues, please include:

- Fluxion version (`fluxion --version`)
- Configuration details
- Performance report (JSON format)
- System information (OS, CPU, memory)
- Steps to reproduce

### Performance Data

Performance reports are stored in JSON format and contain:

```json
{
  "timestamp": "2026-04-08T00:00:00Z",
  "version": "1.2.0",
  "metrics": {
    "timestep_duration_ms": 35.2,
    "memory_usage_bytes": 7500000,
    "iterations_per_timestep": 14
  },
  "status": "PASS"
}
```

## Appendix

### Glossary

- **Timestep**: One iteration of the simulation
- **Solver Iteration**: One step in the numerical solver
- **Convergence**: When the solver reaches a stable solution
- **Baseline**: Reference performance measurement
- **Regression**: Performance degradation from baseline
- **Throughput**: Number of timesteps processed per second

### Performance Formulas

- **Improvement %**: `(baseline - current) / baseline × 100`
- **Throughput**: `1000 / timestep_duration_ms`
- **Memory Efficiency**: `timestep_duration_ms / memory_usage_bytes`

### Further Reading

- [Fluxion Documentation](https://fluxion.example.com/docs)
- [ASHRAE 140 Standard](https://www.ashrae.org/standards/140)
- [Building Energy Modeling Guide](https://example.com/bem-guide)