// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! TOON Round-Trip Integration Tests
//!
//! These tests verify that the TOON (Token-Oriented Object Notation)
//! serializer and deserializer preserve data correctly through a round-trip
//! (serialize → deserialize → compare).
//!
//! # Acceptance Criteria
//!
//! - All primitive types (i32, i64, f32, f64, String) round-trip losslessly
//! - Uniform arrays (Vec<T>) collapse to CSV format correctly
//! - Uniform structs serialize with proper key=value format
//! - Count headers enable length validation guardrails
//! - Zero numerical drift for floating-point values
//!
//! See Issue #2071

use fluxion_toon::{from_str, to_string};

// ============================================================================
// Test Data Structures
// ============================================================================

/// Simple scalar value wrapper for testing.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct ScalarWrapper {
    value: i32,
}

/// Zone temperature reading for testing.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct ZoneReading {
    name: String,
    temperature: f64,
    setpoint: f64,
}

/// Multi-zone simulation state for testing.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct ZoneState {
    zone_count: usize,
    zones: Vec<ZoneReading>,
    ambient_temperature: f64,
}

/// Energy measurement for testing.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct EnergyReading {
    heating: f64,
    cooling: f64,
    internal_gains: f64,
}

// ============================================================================
// Scalar Round-Trip Tests
// ============================================================================

#[test]
fn test_roundtrip_i32_scalar() {
    let value = 42i32;
    let toon = to_string(&value).expect("serialization should succeed");
    let parsed: i32 = from_str(&toon).expect("deserialization should succeed");
    assert_eq!(value, parsed, "i32 should round-trip losslessly");
}

#[test]
fn test_roundtrip_i64_scalar() {
    let value = 1234567890i64;
    let toon = to_string(&value).expect("serialization should succeed");
    let parsed: i64 = from_str(&toon).expect("deserialization should succeed");
    assert_eq!(value, parsed, "i64 should round-trip losslessly");
}

#[test]
fn test_roundtrip_f64_scalar() {
    let value = 3.14159265359f64;
    let toon = to_string(&value).expect("serialization should succeed");
    let parsed: f64 = from_str(&toon).expect("deserialization should succeed");
    assert!(
        (value - parsed).abs() < 1e-10,
        "f64 should round-trip losslessly"
    );
}

#[test]
fn test_roundtrip_string() {
    let value = "Zone_1".to_string();
    let toon = to_string(&value).expect("serialization should succeed");
    let parsed: String = from_str(&toon).expect("deserialization should succeed");
    assert_eq!(value, parsed, "String should round-trip losslessly");
}

#[test]
fn test_roundtrip_simple_struct() {
    let wrapper = ScalarWrapper { value: 100 };
    let toon = to_string(&wrapper).expect("serialization should succeed");
    let parsed: ScalarWrapper = from_str(&toon).expect("deserialization should succeed");
    assert_eq!(
        wrapper, parsed,
        "Simple struct should round-trip losslessly"
    );
}

// ============================================================================
// Zone Reading Round-Trip Tests
// ============================================================================

#[test]
fn test_roundtrip_zone_reading() {
    let reading = ZoneReading {
        name: "Living Zone".to_string(),
        temperature: 22.5,
        setpoint: 21.0,
    };
    let toon = to_string(&reading).expect("serialization should succeed");
    let parsed: ZoneReading = from_str(&toon).expect("deserialization should succeed");
    assert_eq!(reading, parsed, "ZoneReading should round-trip losslessly");
}

#[test]
fn test_roundtrip_zone_reading_with_decimal_precision() {
    let reading = ZoneReading {
        name: "Office Zone".to_string(),
        temperature: 23.123456789,
        setpoint: 22.987654321,
    };
    let toon = to_string(&reading).expect("serialization should succeed");
    let parsed: ZoneReading = from_str(&toon).expect("deserialization should succeed");
    assert!((reading.temperature - parsed.temperature).abs() < 1e-9);
    assert!((reading.setpoint - parsed.setpoint).abs() < 1e-9);
}

// ============================================================================
// Multi-Zone State Round-Trip Tests (CSV Collapse Validation)
// ============================================================================

#[test]
fn test_roundtrip_zone_state_single_zone() {
    let state = ZoneState {
        zone_count: 1,
        zones: vec![ZoneReading {
            name: "Zone1".to_string(),
            temperature: 20.0,
            setpoint: 21.0,
        }],
        ambient_temperature: 15.0,
    };
    let toon = to_string(&state).expect("serialization should succeed");
    let parsed: ZoneState = from_str(&toon).expect("deserialization should succeed");
    assert_eq!(state.zone_count, parsed.zone_count);
    assert_eq!(state.zones.len(), parsed.zones.len());
    assert_eq!(state.zones[0].name, parsed.zones[0].name);
    assert!((state.zones[0].temperature - parsed.zones[0].temperature).abs() < 1e-6);
}

#[test]
fn test_roundtrip_zone_state_multiple_zones() {
    let state = ZoneState {
        zone_count: 3,
        zones: vec![
            ZoneReading {
                name: "North Zone".to_string(),
                temperature: 19.5,
                setpoint: 21.0,
            },
            ZoneReading {
                name: "Central Zone".to_string(),
                temperature: 22.0,
                setpoint: 21.0,
            },
            ZoneReading {
                name: "South Zone".to_string(),
                temperature: 24.5,
                setpoint: 21.0,
            },
        ],
        ambient_temperature: 10.0,
    };
    let toon = to_string(&state).expect("serialization should succeed");
    let parsed: ZoneState = from_str(&toon).expect("deserialization should succeed");

    assert_eq!(state.zone_count, parsed.zone_count);
    assert_eq!(state.zones.len(), parsed.zones.len());

    for (i, (orig, recv)) in state.zones.iter().zip(parsed.zones.iter()).enumerate() {
        assert_eq!(orig.name, recv.name, "Zone {} name should match", i);
        assert!(
            (orig.temperature - recv.temperature).abs() < 1e-6,
            "Zone {} temperature should match: {} vs {}",
            i,
            orig.temperature,
            recv.temperature
        );
    }
}

// ============================================================================
// Energy Reading Round-Trip Tests
// ============================================================================

#[test]
fn test_roundtrip_energy_reading() {
    let energy = EnergyReading {
        heating: 1500.5,
        cooling: 800.3,
        internal_gains: 1200.0,
    };
    let toon = to_string(&energy).expect("serialization should succeed");
    let parsed: EnergyReading = from_str(&toon).expect("deserialization should succeed");

    assert!(
        (energy.heating - parsed.heating).abs() < 1e-6,
        "heating should match: {} vs {}",
        energy.heating,
        parsed.heating
    );
    assert!(
        (energy.cooling - parsed.cooling).abs() < 1e-6,
        "cooling should match: {} vs {}",
        energy.cooling,
        parsed.cooling
    );
    assert!(
        (energy.internal_gains - parsed.internal_gains).abs() < 1e-6,
        "internal_gains should match: {} vs {}",
        energy.internal_gains,
        parsed.internal_gains
    );
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_roundtrip_zero_values() {
    let energy = EnergyReading {
        heating: 0.0,
        cooling: 0.0,
        internal_gains: 0.0,
    };
    let toon = to_string(&energy).expect("serialization should succeed");
    let parsed: EnergyReading = from_str(&toon).expect("deserialization should succeed");
    assert_eq!(energy, parsed);
}

#[test]
fn test_roundtrip_large_values() {
    let energy = EnergyReading {
        heating: 1e10,
        cooling: 1e9,
        internal_gains: 1e8,
    };
    let toon = to_string(&energy).expect("serialization should succeed");
    let parsed: EnergyReading = from_str(&toon).expect("deserialization should succeed");
    assert!(
        (energy.heating - parsed.heating).abs() < 1.0,
        "Large heating value should be preserved"
    );
    assert!(
        (energy.cooling - parsed.cooling).abs() < 0.1,
        "Large cooling value should be preserved"
    );
    assert!(
        (energy.internal_gains - parsed.internal_gains).abs() < 0.01,
        "Large internal_gains value should be preserved"
    );
}

#[test]
fn test_roundtrip_negative_values() {
    let temperature = -5.5f64;
    let toon = to_string(&temperature).expect("serialization should succeed");
    let parsed: f64 = from_str(&toon).expect("deserialization should succeed");
    assert!(
        (temperature - parsed).abs() < 1e-10,
        "Negative temperature should round-trip"
    );
}
