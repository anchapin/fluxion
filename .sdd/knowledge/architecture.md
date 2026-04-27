# Architecture Overview

## .
**gsd_mistral_adapter.py**: GSD-Mistral Vibe Adapter Layer

## .githooks
**perf-baseline.py**: Hook: Performance baseline check and comparison
**rust-doc-check.py**: Hook: Validate Rust doc comments on public API

## .github/scripts
**comment_pr.py**: (no docstring)

## api
**distributed_inference.py**: Distributed Inference Architecture for AI Surrogates
**distributed_inference_config.py**: Configuration Management for Distributed Inference
**main.py**: Fluxion REST API Server
**monitoring.py**: Real-time Monitoring and BAS Integration Module

## api/tests
**__init__.py**: (no docstring)
**test_distributed_inference.py**: Tests for Distributed Inference Architecture

## examples
**risk_aware_optimization.py**: Risk-Aware Optimization Example
**run_model.py**: Simple example showing how to use the `Model` API.
**run_oracle.py**: Example showing how to use `BatchOracle` to evaluate a small population.
**validate_surrogate.py**: Validation example: compare surrogate vs analytical predictions.

## scripts
**compare_ff_profiles.py**: (no docstring)
**compare_peak_profiles.py**: (no docstring)
**display_gate_status.py**: Display gate status from gate_status.json file.
**generate_diagnostic_data.py**: Generate synthetic reference data for ASHRAE 140 Cases 195-470 (Diagnostic Validation)
**generate_reference_data.py**: Generate synthetic reference data for ASHRAE 140 Cases 800-810 (HVAC Equipment)
**generate_scorecard.py**: Fluxion Release Scorecard Generator
**release_gate_checker.py**: Release Gate Checker for Fluxion
**sync_planning.py**: Fluxion Planning Sync Tool

## src/python
**hvac.py**: (no docstring)

## tests
**compare_fluxion_energyplus.py**: Compare Fluxion simulation results with EnergyPlus reference data.
**conftest.py**: (no docstring)
**parse_energyplus_eso.py**: Parse EnergyPlus .eso output files and extract reference data for unit tests.
**test_conductance_calculations.py**: Test stubs for 5R1C conductance calculation tests.
**test_examples.py**: (no docstring)
**test_python_bindings.py**: (no docstring)

## tests/ashrae_140_diagnostics
**__init__.py**: ASHRAE 140 Diagnostic Tests.
**test_envelope_heat_transfer.py**: Envelope heat transfer diagnostic tests for ASHRAE 140 cases.
**test_infiltration_loss.py**: Infiltration heat loss diagnostic tests for ASHRAE 140 cases.
**test_internal_gains.py**: Internal gains diagnostic tests for ASHRAE 140 cases.
**test_solar_heat_gain.py**: Solar heat gain diagnostic tests for ASHRAE 140 cases.
**test_thermal_mass.py**: Thermal mass diagnostic tests for ASHRAE 140 cases.

## tests/ashrae_140_input_validation
**__init__.py**: ASHRAE 140 Input Validation Tests.
**test_constructions.py**: Construction and material validation tests for ASHRAE 140 cases.
**test_geometry.py**: Geometry validation tests for ASHRAE 140 cases.
**test_hvac.py**: HVAC and thermostat validation tests for ASHRAE 140 cases.
**test_infiltration.py**: Infiltration validation tests for ASHRAE 140 cases.
**test_internal_gains.py**: Internal gains validation tests for ASHRAE 140 cases.
**test_weather.py**: Weather and location validation tests for ASHRAE 140 cases.

## tests/ashrae_140_output_validation
**__init__.py**: ASHRAE 140 Output Validation Tests.
**test_annual_energy.py**: Annual energy comparison tests for ASHRAE 140 cases.
**test_hourly_temps.py**: Hourly temperature comparison tests for ASHRAE 140 cases.
**test_monthly_energy.py**: Monthly energy comparison tests for ASHRAE 140 cases.
**test_peak_loads.py**: Peak load comparison tests for ASHRAE 140 cases.

## tests/integration
**test_numpy_arrays.py**: NumPy array validation tests for PyO3 bindings

## tests/python
**test_hvac_bindings.py**: Python tests for HVAC bindings

## tools
**__init__.py**: (no docstring)
**ashrae_140_reference.py**: ASHRAE 140 Reference Data Module
**ashrae_140_test_harness.py**: ASHRAE 140 Test Harness.
**benchmark_batch_inference.py**: Phase 6: Batch Inference Benchmark
**benchmark_inference.py**: (no docstring)
**benchmark_throughput.py**: (no docstring)
**benchmark_throughput_gpu.py**: (no docstring)
**data_collection.py**: Data Collection Tool for Fluxion.
**ep_oracle.py**: EnergyPlus Test Oracle for Physics Validation
**generate_dummy_surrogate.py**: Generate a trivial ONNX surrogate model for examples/demo.
**generate_ep_reference.py**: Generate EnergyPlus reference data for ASHRAE 140 cases.
**generate_regression_tests.py**: Generate Regression Tests from EnergyPlus Reference Data
**geometry_extraction.py**: Automated Geometry Ingestion Pipeline (PDF/CAD-to-BEM) via Vision-Language Models.
**integrate_ml_surrogate_pipeline.py**: Bridge script for ASHRAE 140 validation training data collection and ONNX model training.
**multi_case_optimization.py**: Multi-Case Optimization Module
**optimization.py**: Optimization Algorithms Module
**parameter_validation.py**: Parameter Validation Module
**performance_metrics.py**: Performance Metrics Module
**physics_informed_loss.py**: Physics-Informed Loss Functions
**train_ensemble_surrogates.py**: Phase 7: Ensemble Training for Multiple Surrogate Models
**train_ml_surrogate.py**: ML Surrogate Training Pipeline for Fluxion
**train_pinn.py**: PINN (Physics-Informed Neural Network) Training Pipeline

## tools/compliance_agent
**__init__.py**: Code Compliance Agent for Building Energy Modeling
**agent.py**: Code Compliance Agent for Building Energy Modeling
**demo.py**: Demo Script for Code Compliance Agent
**llm_backend.py**: LLM Backend Interface for Code Compliance Agent

## tools/compliance_agent/tests
**__init__.py**: (no docstring)
**test_compliance_agent.py**: Tests for the Code Compliance Agent

## tools/data_gen
**__init__.py**: Data Generation Tool for Fluxion Surrogate Models.
**ashrae_140_generator.py**: ASHRAE 140 Case Generator for Training Data Generation.
**create_dummy_onnx.py**: (no docstring)
**geometry.py**: Geometry generation utilities for OpenStudio.
**main.py**: CLI Entry point for the Data Generation Tool.
**monte_carlo.py**: Massive Data Generation Tool for AI Surrogate Training.
**sampler.py**: Parameter variation sampler for training data generation.
**simulation.py**: Simulation execution engine.
**test_ashrae_140_generator.py**: Tests for ASHRAE 140 case generator.
**test_monte_carlo.py**: Tests for Monte Carlo Data Generation Tool.
**weather.py**: Weather file management utilities.

## tools/tests
**test_ashrae_140_reference.py**: Test suite for ASHRAE 140 reference data module.
**test_data_collection.py**: (no docstring)
**test_data_gen.py**: (no docstring)
**test_ensemble_training.py**: Tests for Ensemble Training
**test_gymnasium_env.py**: Tests for Gymnasium Environment Wrapper (FluxionEnv)
**test_multi_case_optimization.py**: Test suite for multi-case optimization module.
**test_online_learning.py**: Tests for Online Learning Framework
**test_optimization.py**: Test suite for optimization algorithms.
**test_parameter_validation.py**: Test suite for parameter validation module.
**test_performance_metrics.py**: Test suite for performance metrics module.
**test_physics_informed_loss.py**: Tests for Physics-Informed Loss Functions
**test_train_integration.py**: (no docstring)
