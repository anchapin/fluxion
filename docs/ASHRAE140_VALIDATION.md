# ASHRAE Standard 140 Validation

This document provides an overview of the ASHRAE Standard 140 validation suite implemented in Fluxion.

## Introduction

ASHRAE Standard 140, "Standard Method of Test for the Evaluation of Building Energy Analysis Computer Programs," is the industry-standard for validating building energy simulation software. It consists of a series of carefully specified test cases that test various aspects of a simulation engine's physics and logic.

## Test Cases

Fluxion implements the following ASHRAE 140 test cases:

### 600 Series (Low Mass)
- **Case 600**: Baseline low-mass building.
- **Case 610**: South shading (1m overhang).
- **Case 620**: East/West window orientation.
- **Case 630**: East/West shading (overhang + fins).
- **Case 640**: Thermostat setback (nighttime).
- **Case 650**: Night ventilation.
- **Case 600FF**: Free-floating temperatures.
- **Case 650FF**: Free-floating with night ventilation.

### 900 Series (High Mass)
- **Case 900**: Baseline high-mass (concrete) building.
- **Case 910**: South shading.
- **Case 920**: East/West window orientation.
- **Case 930**: East/West shading.
- **Case 940**: Thermostat setback.
- **Case 950**: Night ventilation.
- **Case 900FF**: Free-floating temperatures.
- **Case 950FF**: Free-floating with night ventilation.

### Special Cases
- **Case 960**: Multi-zone sunspace coupling.
- **Case 195**: Solid conduction through envelope.

## Validation Process

Validation is performed by comparing Fluxion's results (Annual Heating, Annual Cooling, Peak Loads) against reference results from established programs like EnergyPlus, ESP-r, and TRNSYS.

Results are considered passing if they fall within the specified reference ranges or within a 5% tolerance band of the reference midpoint.

## Usage

You can run the validation suite using the Fluxion CLI:

```bash
fluxion validate --all
```

Or specific cases:

```bash
fluxion validate --case 600
```
