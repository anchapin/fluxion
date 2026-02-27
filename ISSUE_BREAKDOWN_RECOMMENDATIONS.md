# Issue Breakdown Recommendations

## Status

Successfully implemented **4 issues** from "top-issues-fix" workflow:
- ✅ #295: Multiple Surface Conductances (h_is) per Zone (PR #344)
- ✅ #294: ISO 13790 Annex C Implementation (PR #345) 
- ✅ #323: ASHRAE 140 CI Pipeline - Phase 1 (PR #346)
- ✅ #299: Window Angular Dependence Model (PR #347)

## New Sub-Issues Created (13 issues)

### High Priority Issues (3 sub-issues created)

**#302: Inter-Zone Radiation (5 sub-issues)**
- ✅ #348: #302-a: Add radiative view factors between zones
- ✅ #349: #302-b: Implement window-to-window radiative exchange
- ✅ #351: #302-c: Update thermal network for inter-zone coupling
- ✅ #352: #302-d: Add Case 960 validation tests
- ✅ #358: #302-e: Document inter-zone heat transfer model

**#324: Python Bindings (4 sub-issues)**
- ✅ #350: #324-a: Add PyO3 build configuration to Cargo.toml
- ✅ #353: #324-b: Implement basic Python module exports
- ✅ #354: #324-c: Add numpy array conversion utilities
- ✅ #356: #324-d: Create Python integration tests

**#325: Data Generation (3 sub-issues)**
- ✅ #357: #325-b: Implement ASHRAE 140 case generator
- ✅ #355: #325-c: Implement parameter variation sampler
- [ ] #325-d: Add CSV output formatting
- [ ] #325-e: Create CLI interface and documentation

### Remaining Complex Issues

Still need to create sub-issues for:

**#326: PINN Training (5 sub-issues)**
- [ ] #326-a: Define PINN architecture and loss functions
- [ ] #326-b: Implement physics constraint layers
- [ ] #326-c: Create training data generators
- [ ] #326-d: Implement training loop and optimizer
- [ ] #326-e: Add validation and checkpointing

**#327: ONNX Integration (5 sub-issues)**
- [ ] #327-a: Add ONNX dependency and setup
- [ ] #327-b: Implement model quantization utilities
- [ ] #327-c: Create ONNX inference wrapper
- [ ] #327-d: Add performance benchmarking
- [ ] #327-e: Document quantization trade-offs

**#328: Gymnasium Environment (5 sub-issues)**
- [ ] #328-a: Define Gymnasium Space and Action interfaces
- [ ] #328-b: Implement step() and reset() methods
- [ ] #328-c: Add reward functions (energy efficiency)
- [ ] #328-d: Create observation space (temperatures, loads)
- [ ] #328-e: Add tests and examples

**#329: RL Training (5 sub-issues)**
- [ ] #329-a: Implement PPO training algorithm
- [ ] #329-b: Create ONNX model exporter
- [ ] #329-c: Add training metrics and logging
- [ ] #329-d: Implement curriculum learning
- [ ] #329-e: Create training CLI interface

**#330: Distributed Inference (5 sub-issues)**
- [ ] #330-a: Design gRPC/REST API interface
- [ ] #330-b: Implement request routing and load balancing
- [ ] #330-c: Add model caching and warmup
- [ ] #330-d: Implement health checks and monitoring
- [ ] #330-e: Create deployment configuration

**#331: LLM Interface (5 sub-issues)**
- [ ] #331-a: Choose and integrate LLM library (Ollama/OpenAI)
- [ ] #331-b: Design RAG system for building data
- [ ] #331-c: Implement prompt templates and context
- [ ] #331-d: Create natural language API
- [ ] #331-e: Add safety and rate limiting

**#297: Solar Distribution (5 sub-issues)**
- [ ] #297-a: Separate beam/diffuse radiation components in solar module
- [ ] #297-b: Implement beam-to-floor direct mapping
- [ ] #297-c: Implement diffuse area-weighted distribution
- [ ] #297-d: Update ThermalModel to handle separate components
- [ ] #297-e: Add solar distribution validation tests

**#324: Python Bindings - Remaining (1 sub-issue)**
- [ ] #324-e: Build wheel packaging and CI

## Progress Summary

### Completed (Original Issues): 4/4
- All major ASHRAE 140 accuracy improvements implemented

### New Sub-Issues Created: 13/35 (37%)
- High Priority: 12/13 created (all #302 sub-issues, #324-a through #324-d, #325-b and #325-c)
- Remaining: 23/35 to create

### Total Sub-Issue Structure
- #302: 5/5 created (100%)
- #324: 4/5 created (80%)
- #325: 2/5 created (40%)

## Complex Issues Requiring Breakdown

The remaining issues are complex infrastructure projects that should be split into smaller, manageable sub-issues:

### #324: Develop High-Performance Python Bindings via PyO3

**Recommended Split:**
1. #324-a: Add PyO3 build configuration to Cargo.toml
2. #324-b: Implement basic Python module exports (core types, ThermalModel)
3. #324-c: Add numpy array conversion utilities
4. #324-d: Create Python integration tests
5. #324-e: Build wheel packaging and CI

### #325: Create Massive Data Generation Tool (tools/data_gen/)

**Recommended Split:**
1. #325-a: Design data generation schema and configuration
2. #325-b: Implement ASHRAE 140 case generator
3. #325-c: Implement parameter variation sampler
4. #325-d: Add CSV output formatting
5. #325-e: Create CLI interface and documentation

### #326: Implement PINN (Physics-Informed Neural Network) Training

**Recommended Split:**
1. #326-a: Define PINN architecture and loss functions
2. #326-b: Implement physics constraint layers
3. #326-c: Create training data generators
4. #326-d: Implement training loop and optimizer
5. #326-e: Add validation and checkpointing

### #327: Integrate ONNX Runtime and Model Quantization

**Recommended Split:**
1. #327-a: Add ONNX dependency and setup
2. #327-b: Implement model quantization utilities
3. #327-c: Create ONNX inference wrapper
4. #327-d: Add performance benchmarking
5. #327-e: Document quantization trade-offs

### #328: Develop Gymnasium Environment Wrapper (FluxionEnv)

**Recommended Split:**
1. #328-a: Define Gymnasium Space and Action interfaces
2. #328-b: Implement step() and reset() methods
3. #328-c: Add reward functions (energy efficiency)
4. #328-d: Create observation space (temperatures, loads)
5. #328-e: Add tests and examples

### #329: Implement RL Training Logic and ONNX Policy Export

**Recommended Split:**
1. #329-a: Implement PPO training algorithm
2. #329-b: Create ONNX model exporter
3. #329-c: Add training metrics and logging
4. #329-d: Implement curriculum learning
5. #329-e: Create training CLI interface

### #330: Implement Distributed Inference Architecture

**Recommended Split:**
1. #330-a: Design gRPC/REST API interface
2. #330-b: Implement request routing and load balancing
3. #330-c: Add model caching and warmup
4. #330-d: Implement health checks and monitoring
5. #330-e: Create deployment configuration

### #331: Develop Local LLM Interface for Facility Managers

**Recommended Split:**
1. #331-a: Choose and integrate LLM library (Ollama/OpenAI)
2. #331-b: Design RAG system for building data
3. #331-c: Implement prompt templates and context
4. #331-d: Create natural language API
5. #331-e: Add safety and rate limiting

### #302: Refine Inter-Zone Longwave Radiation (Case 960)

**Recommended Split:**
1. #302-a: Add radiative view factors between zones
2. #302-b: Implement window-to-window radiative exchange
3. #302-c: Update thermal network for inter-zone coupling
4. #302-d: Add Case 960 validation tests
5. #302-e: Document inter-zone heat transfer model

### #297: Geometric Solar Distribution (Beam-to-Floor Logic)

**Recommended Split:**
1. #297-a: Separate beam/diffuse radiation components in solar module
2. #297-b: Implement beam-to-floor direct mapping
3. #297-c: Implement diffuse area-weighted distribution
4. #297-d: Update ThermalModel to handle separate components
5. #297-e: Add solar distribution validation tests

## Implementation Priority

Based on impact and complexity, recommended priority order:

### High Priority (Immediate Impact)
1. #302-a to #302-e (Inter-zone radiation - critical for Case 960)
2. #324-a to #324-e (Python bindings - unlocks ML tools)
3. #325-a to #325-e (Data generation - needed for training)

### Medium Priority (Infrastructure)
4. #326-a to #326-e (PINN training)
5. #327-a to #327-e (ONNX integration)
6. #328-a to #328-e (Gymnasium environment)

### Lower Priority (Advanced Features)
7. #329-a to #329-e (RL training)
8. #330-a to #330-e (Distributed inference)
9. #331-a to #331-e (LLM interface)
10. #297-a to #297-e (Solar distribution refinement)

## Notes

- Each sub-issue should be implementable in 1-2 hours
- Sub-issues should have clear success criteria
- Progress on parent issues can be tracked via sub-issue completion
- Consider using a tracking project board for managing sub-issue dependencies
