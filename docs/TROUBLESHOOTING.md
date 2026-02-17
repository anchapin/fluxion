# Fluxion Troubleshooting Guide

## Common Issues

### Installation Issues

#### "maturin not found" error

```bash
# Install maturin first
pip install maturin
```

#### Rust compilation errors

```bash
# Ensure Rust is installed
rustup update
cargo build
```

### Runtime Issues

#### ONNX model loading fails

**Error:** `RuntimeError: Failed to load ONNX model`

**Solution:**
1. Verify the model file exists and is valid ONNX format
2. Check ONNX Runtime is installed: `pip install onnxruntime`
3. Try with a simpler model first

#### Out of memory errors

**Error:** `OutOfMemoryError` during batch evaluation

**Solutions:**
1. Reduce batch size
2. Use CPU fallback: `onnxruntime.set_default_execution_providers(['CPUExecutionProvider'])`
3. Free up GPU memory

### Physics Simulation Issues

#### Unrealistic energy values

**Check:**
- Verify building parameters are in valid ranges
- Ensure weather data is loaded correctly
- Check units (SI units expected)

#### Simulation diverges

**Error:** `NaN` or infinite values in results

**Solutions:**
1. Reduce timestep size
2. Check for invalid parameter combinations
3. Verify boundary conditions

### Performance Issues

#### Slow inference

**Solutions:**
1. Use release build: `cargo build --release`
2. Enable GPU: `onnxruntime.set_default_execution_providers(['CUDAExecutionProvider'])`
3. Increase batch size

#### Low GPU utilization

**Solutions:**
1. Ensure ONNX Runtime GPU support is installed
2. Use larger batches
3. Check GPU is available: `nvidia-smi`

## Debugging Tips

### Enable logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check system info

```python
import fluxion
print(fluxion.system_info())
```

### Validate configuration

```bash
fluxion validate config.json
```

## Getting Help

- GitHub Issues: https://github.com/anchapin/fluxion/issues
- Discord: https://discord.gg/fluxion
- Email: support@fluxion.org
