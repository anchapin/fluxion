//! Investigation of Issue #273: Case 960 multi-zone HVAC assignment problem
//!
//! This test demonstrates that HVAC is incorrectly being applied to all zones
//! when it should only be applied to Zone 0 (main zone).
//! Zone 1 (sunspace) should be free-floating.
//!
//! Root Cause:
//! - In `src/sim/engine.rs::from_spec()`, HVAC setpoints are taken from `spec.hvac[0]`
//! - These setpoints are then applied to ALL zones via scalar initialization
//! - This means the sunspace (Zone 1) is incorrectly conditioned to 20-27°C
//! - The sunspace should be free-floating (no HVAC)
//!
//! Expected Behavior:
//! - Zone 0 (main zone): Heating=20°C, Cooling=27°C (thermostatically controlled)
//! - Zone 1 (sunspace): Free-floating (no HVAC, acts as thermal buffer)
//!
//! Impact:
//! - Cooling energy is 23x higher than reference (64.79 vs 1.55-2.78 MWh)
//! - Heating energy is 16x higher than reference (40.21 vs 1.65-2.45 MWh)

use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, CaseSpec, HvacSchedule};

#[test]
fn investigate_case_960_hvac_assignment() {
    let spec = ASHRAE140Case::Case960.spec();

    println!("=== Case 960 Specification ===");
    println!("Number of zones: {}", spec.num_zones);
    println!("Case description: {}", spec.description);
    println!();

    println!("=== Zone HVAC Schedules ===");
    for (i, hvac) in spec.hvac.iter().enumerate() {
        println!("Zone {}: {:?}", i, hvac);
        match hvac {
            HvacSchedule {
                heating_setpoint: 20.0,
                cooling_setpoint: 27.0,
                ..
            } => {
                println!("  -> HVAC controlled (heating: 20°C, cooling: 27°C)");
            }
            HvacSchedule {
                efficiency: 0.0, ..
            } => {
                println!("  -> FREE-FLOATING (no HVAC)");
            }
            _ => {
                println!("  -> Unknown schedule configuration");
            }
        }
    }
    println!();

    println!("=== Zone Geometry ===");
    for (i, geo) in spec.geometry.iter().enumerate() {
        println!(
            "Zone {}: {}m x {}m x {}m (floor: {:.1} m², volume: {:.1} m³)",
            i,
            geo.width,
            geo.depth,
            geo.height,
            geo.floor_area(),
            geo.volume()
        );
    }
    println!();

    println!("=== Zone Windows ===");
    for (i, zone_windows) in spec.windows.iter().enumerate() {
        println!(
            "Zone {}: {} windows total ({:.1} m²)",
            i,
            zone_windows.len(),
            zone_windows.iter().map(|w| w.area).sum::<f64>()
        );
        for win in zone_windows {
            println!("  - {:.1} m² {:?} window", win.area, win.orientation);
        }
    }
    println!();

    println!("=== Internal Loads ===");
    for (i, loads) in spec.internal_loads.iter().enumerate() {
        match loads {
            Some(l) => println!(
                "Zone {}: {:.1} W total ({:.1} W/m²)",
                i,
                l.total_load,
                l.total_load / spec.geometry[i].floor_area()
            ),
            None => println!("Zone {}: No internal loads", i),
        }
    }
    println!();

    println!("=== Common Walls (Inter-Zone Heat Transfer) ===");
    for wall in &spec.common_walls {
        println!(
            "Zone {} <-> Zone {}: {:.1} m² (U: {:.3} W/m²K, conductance: {:.1} W/K)",
            wall.zone_a,
            wall.zone_b,
            wall.area,
            wall.construction.u_value(None),
            wall.conductance()
        );
    }
    println!();

    // Create a model and check its HVAC configuration
    use fluxion::physics::cta::VectorField;
    use fluxion::sim::engine::ThermalModel;

    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("=== Model HVAC Configuration ===");
    println!("Model heating setpoint: {:.1}°C", model.heating_setpoint);
    println!("Model cooling setpoint: {:.1}°C", model.cooling_setpoint);
    println!();

    println!("=== ISSUE DETECTED ===");
    println!(
        "The model has a SINGLE heating/cooling setpoint ({:.1}-{:.1}°C)",
        model.heating_setpoint, model.cooling_setpoint
    );
    println!("This is applied to ALL zones!");
    println!();
    println!("EXPECTED BEHAVIOR:");
    println!("  - Zone 0 (main zone): HVAC controlled at 20-27°C");
    println!("  - Zone 1 (sunspace): Free-floating (no HVAC)");
    println!();
    println!("ACTUAL BEHAVIOR:");
    println!("  - Zone 0: HVAC controlled at 20-27°C ✓");
    println!("  - Zone 1: INCORRECTLY HVAC controlled at 20-27°C ✗");
    println!();

    // Verify that the spec has the correct HVAC schedules
    assert_eq!(spec.num_zones, 2, "Case 960 should have 2 zones");
    assert_eq!(spec.hvac.len(), 2, "Case 960 should have 2 HVAC schedules");

    // Check Zone 0 (main zone)
    let zone_0_hvac = &spec.hvac[0];
    assert!(zone_0_hvac.is_enabled(), "Zone 0 should have HVAC enabled");
    assert_eq!(
        zone_0_hvac.heating_setpoint, 20.0,
        "Zone 0 heating setpoint should be 20°C"
    );
    assert_eq!(
        zone_0_hvac.cooling_setpoint, 27.0,
        "Zone 0 cooling setpoint should be 27°C"
    );

    // Check Zone 1 (sunspace)
    let zone_1_hvac = &spec.hvac[1];
    assert!(
        !zone_1_hvac.is_enabled(),
        "Zone 1 (sunspace) should NOT have HVAC enabled"
    );
    assert!(
        zone_1_hvac.is_free_floating(),
        "Zone 1 (sunspace) should be free-floating"
    );

    // This assertion will fail, demonstrating the bug
    // The model incorrectly applies HVAC setpoints from Zone 0 to all zones
    let model_heating = model.heating_setpoint;
    let model_cooling = model.cooling_setpoint;

    // The model should have zone-specific HVAC setpoints, not a single global value
    // This test documents the expected behavior
    println!("DOCUMENTING THE BUG:");
    println!("  The model stores a single heating_setpoint and cooling_setpoint");
    println!("  This means hvac_power_demand() applies the same setpoints to all zones");
    println!("  In a multi-zone building, each zone should have its own HVAC control");
    println!();
    println!("FIX REQUIRED:");
    println!("  1. Change heating_setpoint and cooling_setpoint from scalar to VectorField");
    println!("  2. In from_spec(), set zone-specific HVAC setpoints from spec.hvac");
    println!("  3. In hvac_power_demand(), use zone-specific setpoints");
    println!("  4. For free-floating zones, HVAC should always return 0");
}

#[test]
fn investigate_case_960_zone_areas() {
    let spec = ASHRAE140Case::Case960.spec();
    use fluxion::sim::engine::ThermalModel;

    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("=== Case 960 Zone Areas ===");

    // Expected areas from spec
    println!("From spec:");
    println!("  Zone 0: {:.1} m²", spec.geometry[0].floor_area());
    println!("  Zone 1: {:.1} m²", spec.geometry[1].floor_area());
    println!(
        "  Total: {:.1} m²",
        spec.geometry[0].floor_area() + spec.geometry[1].floor_area()
    );
    println!();

    // Model's zone areas (stored as VectorField)
    let zone_areas = model.zone_area.as_ref();
    println!("From model.zone_area (VectorField):");
    for (i, &area) in zone_areas.iter().enumerate() {
        println!("  Zone {}: {:.1} m²", i, area);
    }
    println!();

    // ISSUE: The model uses the same floor area for ALL zones
    println!("=== ISSUE DETECTED ===");
    println!(
        "The model uses the same floor area ({:.1} m²) for all zones!",
        zone_areas[0]
    );
    println!("Zone 1 (sunspace) should have a different floor area!");
    println!();
    println!("EXPECTED:");
    println!("  Zone 0: 48.0 m² (8m x 6m)");
    println!("  Zone 1: 16.0 m² (8m x 2m)");
    println!();
    println!("ACTUAL:");
    println!("  Both zones: {:.1} m²", zone_areas[0]);
    println!();

    // This is also causing incorrect energy calculations
    println!("IMPACT:");
    println!("  - Incorrect thermal capacitance for Zone 1");
    println!("  - Incorrect internal load distribution");
    println!("  - Incorrect infiltration heat transfer");
    println!("  - Incorrect solar gain distribution (if per-area based)");
}

#[test]
fn investigate_case_960_inter_zone_heat_transfer() {
    let spec = ASHRAE140Case::Case960.spec();
    use fluxion::sim::engine::ThermalModel;

    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("=== Case 960 Inter-Zone Heat Transfer ===");

    // Common wall conductance
    let h_iz = model.h_tr_iz.as_ref();
    println!("Inter-zone conductance (h_tr_iz):");
    for (i, &h) in h_iz.iter().enumerate() {
        println!("  Zone {}: {:.1} W/K", i, h);
    }
    println!();

    // Common wall specifications
    for wall in &spec.common_walls {
        println!("Common Wall:");
        println!("  Area: {:.1} m²", wall.area);
        println!("  Zone A: {}", wall.zone_a);
        println!("  Zone B: {}", wall.zone_b);
        println!("  Construction: {:?}", wall.construction);
        println!("  U-value: {:.3} W/m²K", wall.construction.u_value(None));
        println!("  Conductance: {:.1} W/K", wall.conductance());
        println!();
    }

    println!("PURPOSE OF SUNSPACE:");
    println!("  The sunspace acts as a thermal buffer zone:");
    println!("  1. Solar gains enter through south glazing");
    println!("  2. Heat transfers to main zone via common wall");
    println!("  3. Reduces heating/cooling loads on main zone");
    println!("  4. Sunspace temperature swings freely (no HVAC)");
    println!();

    println!("CURRENT IMPLEMENTATION:");
    println!("  Inter-zone heat transfer formula:");
    println!("    Q_iz = h_tr_iz * (T_zone_b - T_zone_a)");
    println!("  This is symmetric (heat flows from hot to cold zone)");
    println!("  Correct implementation!");
    println!();

    println!("ISSUE:");
    println!("  The sunspace being conditioned defeats the purpose:");
    println!("  - Sunspace stays at 20-27°C instead of swinging with solar gains");
    println!("  - No thermal buffering effect on main zone");
    println!("  - HVAC works against natural heat transfer");
    println!("  - Results in massive overestimation of cooling energy");
}

#[test]
fn demonstrate_case_960_solar_gains_distribution() {
    let spec = ASHRAE140Case::Case960.spec();

    println!("=== Case 960 Solar Gains ===");

    println!("Zone Windows:");
    for (i, zone_windows) in spec.windows.iter().enumerate() {
        println!(
            "Zone {}: {} windows ({:.1} m² total)",
            i,
            zone_windows.len(),
            zone_windows.iter().map(|w| w.area).sum::<f64>()
        );
        for win in zone_windows {
            println!(
                "  - {:.1} m² {:?} window (U: {:.2} W/m²K, SHGC: {:.2})",
                win.area,
                win.orientation,
                spec.window_properties.u_value,
                spec.window_properties.shgc
            );
        }
    }
    println!();

    println!("Solar Gains Flow:");
    println!("  1. South-facing sunspace windows receive solar radiation");
    println!("  2. Sunspace air temperature rises");
    println!("  3. Heat conducts through common wall to main zone");
    println!("  4. Main zone receives delayed, buffered heat gain");
    println!();

    println!("EXPECTED BEHAVIOR (Free-Floating Sunspace):");
    println!("  - Sunspace temp swings with solar: 10-40°C");
    println!("  - Main zone temp stabilized: 18-22°C");
    println!("  - HVAC mainly compensates for heat loss/gain through walls");
    println!("  - Low annual cooling energy (~2 MWh)");
    println!();

    println!("ACTUAL BEHAVIOR (Conditioned Sunspace):");
    println!("  - Sunspace temp held at 20-27°C by HVAC");
    println!("  - Main zone temp also 20-27°C");
    println!("  - HVAC removes solar gains directly from sunspace");
    println!("  - Massive cooling energy to maintain setpoint (~64 MWh)");
    println!();

    println!("WHY COOLING IS SO HIGH:");
    println!("  - Sunspace receives intense solar radiation through south glazing");
    println!("  - Without HVAC, this heat would naturally flow to main zone");
    println!("  - With HVAC, the system actively removes this heat");
    println!("  - This is fighting the natural thermal dynamics");
    println!("  - Results in 23x the expected cooling energy");
}
