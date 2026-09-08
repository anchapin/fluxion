mod tests {
    //! Inline unit tests for the deterministic physics helpers in
    //! `thermal_model_core` (Issue #2535).
    //!
    //! These tests target functions whose result can be checked without running
    //! a full 8760-step simulation: the diurnal daily-cycle generator, the
    //! ISO 13790 half-insulation resistance calculators, the energy-conservation
    //! validator, the cumulative-energy / peak-power bucket accessors, and the
    //! solar-position cache. Reference values are computed in Python (RULES.md
    //! constraint #0) and reproduced here as `approx_eq` checks.
    use crate::sim::construction::{
        Construction, ConstructionLayer, SurfaceType as SimSurfaceType,
    };
    use crate::sim::thermal_model_core::*;

    const TOL: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn is_finite(x: f64) -> bool {
        x.is_finite()
    }

    // ---------------------------------------------------------------------
    // get_daily_cycle: deterministic diurnal sine wave, sin(h/24 * 2π − π/2)
    // ---------------------------------------------------------------------
    #[test]
    fn test_get_daily_cycle_has_24_entries_all_finite() {
        let cycle = get_daily_cycle();
        assert_eq!(cycle.len(), 24);
        assert!(cycle.iter().all(|&v| is_finite(v)));
    }

    #[test]
    fn test_get_daily_cycle_extrema_and_zero_crossings() {
        // Reference (Python): val[h] = sin(h/24 * 2π − π/2)
        //   h=0  -> sin(-π/2)   = -1.0  (minimum)
        //   h=6  -> sin(0)       =  0.0  (zero crossing, rising)
        //   h=12 -> sin(π/2)     =  1.0  (maximum)
        //   h=18 -> sin(π)       ≈  0.0  (zero crossing, falling; |sin π| ≤ 2e-16 in f64)
        let cycle = get_daily_cycle();
        assert!(approx_eq(cycle[0], -1.0, TOL));
        assert!(approx_eq(cycle[6], 0.0, TOL));
        assert!(approx_eq(cycle[12], 1.0, TOL));
        assert!(approx_eq(cycle[18], 0.0, 1e-15));
    }

    #[test]
    fn test_get_daily_cycle_bounded_by_unit_amplitude() {
        // Pure sine wave with amplitude 1 — no entry may exceed ±1.
        let cycle = get_daily_cycle();
        for (h, &v) in cycle.iter().enumerate() {
            assert!(
                v.abs() <= 1.0 + TOL,
                "cycle[{h}] = {v} exceeds unit amplitude"
            );
        }
    }

    #[test]
    fn test_get_daily_cycle_anti_periodic_over_12h() {
        // sin((h+12)/24·2π − π/2) = sin(h/24·2π − π/2 + π) = −val[h].
        let cycle = get_daily_cycle();
        for h in 0..12 {
            assert!(
                approx_eq(cycle[h] + cycle[h + 12], 0.0, 1e-15),
                "anti-periodicity broken at h={h}: {} vs {}",
                cycle[h],
                cycle[h + 12]
            );
        }
    }

    #[test]
    fn test_get_daily_cycle_idempotent_pointer() {
        // OnceLock initialisation: two calls must return the same allocation.
        let a = get_daily_cycle() as *const [f64; 24];
        let b = get_daily_cycle() as *const [f64; 24];
        assert_eq!(a, b);
    }

    // ---------------------------------------------------------------------
    // DoorGeometry
    // ---------------------------------------------------------------------
    #[test]
    fn test_door_geometry_new_stores_fields() {
        let d = DoorGeometry::new(2.1, 1.9);
        assert!(approx_eq(d.height, 2.1, TOL));
        assert!(approx_eq(d.area, 1.9, TOL));
    }

    #[test]
    fn test_door_geometry_default_is_zero() {
        let d = DoorGeometry::default();
        assert!(approx_eq(d.height, 0.0, TOL));
        assert!(approx_eq(d.area, 0.0, TOL));
    }

    // ---------------------------------------------------------------------
    // compute_r_interior_to_mass / compute_r_exterior_to_mass
    // ISO 13790 Annex C "half-insulation" rule.
    // ---------------------------------------------------------------------
    /// Build an ASHRAE 140 Case 600-style 3-layer wall:
    /// interior plasterboard (R=0.075) | fiberglass (R=1.65) | wood siding (R≈0.0643).
    /// Fiberglass at index 1 is the dominant insulation layer.
    fn case600_wall() -> Construction {
        Construction::new(vec![
            ConstructionLayer::new("Plasterboard", 0.16, 950.0, 840.0, 0.012),
            ConstructionLayer::new("Fiberglass", 0.04, 12.0, 840.0, 0.066),
            ConstructionLayer::new("Wood siding", 0.14, 500.0, 1300.0, 0.009),
        ])
    }

    #[test]
    fn test_compute_r_interior_to_mass_case600_wall() {
        // Reference (Python): R_plaster + R_fiberglass/2 = 0.075 + 1.65/2 = 0.9 m²K/W.
        let wall = case600_wall();
        let r_int = compute_r_interior_to_mass(&wall, SimSurfaceType::Wall, 10.0);
        assert!(
            approx_eq(r_int, 0.9, 1e-12),
            "expected R_interior_to_mass = 0.9, got {r_int}"
        );
    }

    #[test]
    fn test_compute_r_exterior_to_mass_case600_wall_finite_and_includes_film() {
        // The exterior path must start at the exterior film resistance
        // (1 / EXTERIOR_FILM_COEFF_DEFAULT) and stay finite + positive.
        let wall = case600_wall();
        let r_ext = compute_r_exterior_to_mass(&wall, SimSurfaceType::Wall, 10.0);
        let r_film =
            1.0 / crate::physics::constants::thermal::ashrae_140::EXTERIOR_FILM_COEFF_DEFAULT;
        assert!(r_ext.is_finite());
        assert!(r_ext > 0.0);
        assert!(r_ext >= r_film - TOL);
        // Must include at least the film + half the dominant insulation (1.65/2).
        assert!(r_ext >= r_film + (1.65 / 2.0) - TOL);
    }

    /// Regression for issue #2613: the exterior-to-mass path must sum only
    /// EXTERIOR-side layers (wood siding) plus half the insulation, never the
    /// interior plasterboard. Verified against a layer-by-layer Python reference
    /// (RULES.md: no parameter tuning; ASHRAE 140 Case 600 wall constants).
    ///
    /// Reference (Python, h_ext = EXTERIOR_FILM_COEFF_DEFAULT = 18.3 W/m²K):
    ///   R_film       = 1/18.3           = 0.054645 m²K/W
    ///   R_wood_siding = 0.009/0.14       = 0.064286 m²K/W   (exterior layer, full)
    ///   R_fiberglass/2 = (0.066/0.04)/2  = 0.825000 m²K/W   (insulation, half)
    ///   R_exterior_to_mass              = 0.943930 m²K/W
    ///
    /// The pre-fix code walked the slice interior→exterior with a `reverse_idx`
    /// comparison, which folded the interior plasterboard (R=0.075) into the
    /// exterior path instead of the wood siding, yielding 0.954645 m²K/W
    /// (delta = R_plaster − R_wood = 0.010714).
    #[test]
    fn test_compute_r_exterior_to_mass_case600_wall_physical_value() {
        let wall = case600_wall();
        let r_ext = compute_r_exterior_to_mass(&wall, SimSurfaceType::Wall, 10.0);
        let r_film =
            1.0 / crate::physics::constants::thermal::ashrae_140::EXTERIOR_FILM_COEFF_DEFAULT;
        let r_wood_siding = 0.009_f64 / 0.14;
        let r_fiberglass = 0.066_f64 / 0.04;
        let expected = r_film + r_wood_siding + r_fiberglass / 2.0; // ≈ 0.943930
        assert!(
            approx_eq(r_ext, expected, 1e-9),
            "R_exterior_to_mass: expected {expected} (≈0.9439), got {r_ext}"
        );
        // Hard guard against the pre-fix regression: must NOT equal the buggy value.
        let buggy = r_film + 0.075 + r_fiberglass / 2.0; // plasterboard in exterior path ≈ 0.954645
        assert!(
            !approx_eq(r_ext, buggy, 1e-3),
            "R_exterior_to_mass matched the pre-fix buggy value {buggy} (issue #2613 regressed)"
        );
    }

    #[test]
    fn test_compute_r_exterior_to_mass_always_includes_film_resistance() {
        // The exterior film resistance 1/h_ext is always present regardless of
        // the wall construction, so R_exterior_to_mass ≥ 1/h_ext.
        let wall = case600_wall();
        let r_ext = compute_r_exterior_to_mass(&wall, SimSurfaceType::Wall, 1.0);
        let r_film =
            1.0 / crate::physics::constants::thermal::ashrae_140::EXTERIOR_FILM_COEFF_DEFAULT;
        assert!(r_ext >= r_film - TOL);
        assert!(r_ext.is_finite() && r_ext > 0.0);
    }

    #[test]
    fn test_compute_r_exterior_to_mass_single_layer_half_value() {
        // With one layer, the dominant insulation IS that layer. The exterior
        // path is: film + half the layer's R-value. ins_idx=0, num=1, so
        // reverse_idx=0==ins_idx on the first (only) iteration → R/2, break.
        let single = Construction::new(vec![ConstructionLayer::new(
            "Foam", 0.03, 30.0, 1400.0, 0.10,
        )]);
        let r_ext = compute_r_exterior_to_mass(&single, SimSurfaceType::Wall, 5.0);
        let expected = 1.0
            / crate::physics::constants::thermal::ashrae_140::EXTERIOR_FILM_COEFF_DEFAULT
            + (0.10 / 0.03) / 2.0;
        assert!(
            approx_eq(r_ext, expected, 1e-12),
            "single-layer R_exterior_to_mass = {expected}, got {r_ext}"
        );
    }

    #[test]
    fn test_compute_r_interior_to_mass_single_layer_half_value() {
        // With one layer, the dominant insulation IS that layer, so only half
        // of its R-value contributes to the interior path.
        let single = Construction::new(vec![ConstructionLayer::new(
            "Foam", 0.03, 30.0, 1400.0, 0.10,
        )]);
        let r = compute_r_interior_to_mass(&single, SimSurfaceType::Wall, 5.0);
        let expected = (0.10 / 0.03) / 2.0;
        assert!(approx_eq(r, expected, 1e-12));
    }

    #[test]
    fn test_compute_r_interior_to_mass_floored_at_1e_minus_3() {
        // The function clamps at 0.001 m²K/W to avoid zero resistances when
        // there are no interior layers (insulation is the innermost layer).
        let wall = Construction::new(vec![
            ConstructionLayer::new("Insulation-only", 0.025, 20.0, 1450.0, 0.20),
            ConstructionLayer::new("Brick", 0.81, 1700.0, 800.0, 0.10),
        ]);
        let r_int = compute_r_interior_to_mass(&wall, SimSurfaceType::Wall, 10.0);
        assert!(r_int >= 0.001 - TOL);
        assert!(r_int.is_finite() && r_int > 0.0);
    }

    // ---------------------------------------------------------------------
    // validate_energy_conservation: HVAC + Solar + Internal = Envelope + ΔU_mass
    // ---------------------------------------------------------------------
    #[test]
    fn test_validate_energy_conservation_balanced_returns_none() {
        let mut model = ThermalModel::new(1);
        model.0.mass.mass_energy_change_cumulative = 5.0e6; // 5 MJ stored in mass
        let hvac = 10.0e6;
        let solar = 3.0e6;
        let internal = 2.0e6;
        let envelope = hvac + solar + internal - 5.0e6; // exact balance
        assert!(model
            .validate_energy_conservation(hvac, solar, internal, envelope)
            .is_none());
    }

    #[test]
    fn test_validate_energy_conservation_zero_energy_returns_none() {
        // Trivially balanced (and within the absolute 1 MJ floor of the tol).
        let model = ThermalModel::new(1);
        assert!(model
            .validate_energy_conservation(0.0, 0.0, 0.0, 0.0)
            .is_none());
    }

    #[test]
    fn test_validate_energy_conservation_small_imbalance_within_tolerance() {
        // tolerance = |In|*0.001 + 1e6. With In = 1e9, tolerance = 2e6 J.
        // A 5e5 J imbalance is below the threshold → None.
        let mut model = ThermalModel::new(1);
        model.0.mass.mass_energy_change_cumulative = 0.0;
        let hvac = 1.0e9;
        let solar = 0.0;
        let internal = 0.0;
        let envelope = hvac - 5.0e5; // 0.5 MJ short
        assert!(model
            .validate_energy_conservation(hvac, solar, internal, envelope)
            .is_none());
    }

    #[test]
    fn test_validate_energy_conservation_large_imbalance_returns_message() {
        // 5e7 J imbalance with tolerance 2e6 J → violation.
        let mut model = ThermalModel::new(1);
        model.0.mass.mass_energy_change_cumulative = 0.0;
        let hvac = 1.0e9;
        let envelope = hvac - 5.0e7;
        let result = model.validate_energy_conservation(hvac, 0.0, 0.0, envelope);
        assert!(result.is_some());
        let msg = result.unwrap();
        assert!(
            msg.contains("Energy conservation violation"),
            "message missing label: {msg}"
        );
        assert!(msg.contains("Imbalance="));
    }

    #[test]
    fn test_validate_energy_conservation_outflow_via_mass_storage() {
        // Negative mass change (mass cooling down) must count as energy OUT,
        // balancing a positive HVAC input → None.
        let mut model = ThermalModel::new(1);
        model.0.mass.mass_energy_change_cumulative = -8.0e6; // mass releases 8 MJ
        let hvac = 2.0e6;
        let envelope = 10.0e6; // 2 (HVAC) + 8 (mass release) = 10 out
        assert!(model
            .validate_energy_conservation(hvac, 0.0, 0.0, envelope)
            .is_none());
    }

    // ---------------------------------------------------------------------
    // Cumulative-energy / peak-power bucket accessors & resets
    // ---------------------------------------------------------------------
    #[test]
    fn test_get_zone_energies_kwh_sums_heating_and_cooling_per_zone() {
        let mut model = ThermalModel::new(2);
        model.0.hvac.zone_heating_energy_kwh = VectorField::new(vec![10.0, 4.0]);
        model.0.hvac.zone_cooling_energy_kwh = VectorField::new(vec![3.0, 7.5]);
        let total = model.get_zone_energies_kwh();
        assert_eq!(total.len(), 2);
        assert!(approx_eq(total[0], 13.0, TOL));
        assert!(approx_eq(total[1], 11.5, TOL));
    }

    #[test]
    fn test_reset_heating_cooling_energy_zeroes_scalars_and_per_zone() {
        let mut model = ThermalModel::new(2);
        model.0.hvac.annual_heating_energy = 42.0;
        model.0.hvac.annual_cooling_energy = 17.0;
        model.0.hvac.zone_heating_energy_kwh = VectorField::new(vec![1.0, 2.0]);
        model.0.hvac.zone_cooling_energy_kwh = VectorField::new(vec![3.0, 4.0]);

        model.reset_heating_cooling_energy();

        assert!(approx_eq(model.get_heating_energy_kwh(), 0.0, TOL));
        assert!(approx_eq(model.get_cooling_energy_kwh(), 0.0, TOL));
        for h in model.get_zone_heating_energy_kwh() {
            assert!(approx_eq(h, 0.0, TOL));
        }
        for c in model.get_zone_cooling_energy_kwh() {
            assert!(approx_eq(c, 0.0, TOL));
        }
    }

    #[test]
    fn test_reset_thermal_mass_energy_zeroes_all_three_accumulators() {
        let mut model = ThermalModel::new(1);
        model.0.mass.mass_energy_change_cumulative = 1.0e7;
        model.0.mass.envelope_mass_energy_change_cumulative = 6.0e6;
        model.0.mass.internal_mass_energy_change_cumulative = 4.0e6;

        model.reset_thermal_mass_energy();

        assert!(approx_eq(model.get_mass_energy_change_joules(), 0.0, TOL));
        assert!(approx_eq(
            model.get_envelope_mass_energy_change_joules(),
            0.0,
            TOL
        ));
        assert!(approx_eq(
            model.get_internal_mass_energy_change_joules(),
            0.0,
            TOL
        ));
    }

    #[test]
    fn test_reset_peak_power_zeroes_both_powers() {
        let mut model = ThermalModel::new(1);
        model.0.hvac.peak_power_heating = 5_000.0; // W
        model.0.hvac.peak_power_cooling = 7_500.0;

        model.reset_peak_power();

        assert!(approx_eq(model.get_peak_heating_power_kw(), 0.0, TOL));
        assert!(approx_eq(model.get_peak_cooling_power_kw(), 0.0, TOL));
    }

    #[test]
    fn test_peak_power_getters_convert_watts_to_kilowatts() {
        let mut model = ThermalModel::new(1);
        model.0.hvac.peak_power_heating = 4_300.0;
        model.0.hvac.peak_power_cooling = 9_100.0;
        assert!(approx_eq(model.get_peak_heating_power_kw(), 4.3, TOL));
        assert!(approx_eq(model.get_peak_cooling_power_kw(), 9.1, TOL));
    }

    #[test]
    fn test_reset_all_energy_tracking_clears_every_category() {
        let mut model = ThermalModel::new(1);
        model.0.hvac.peak_power_heating = 1.0;
        model.0.hvac.peak_power_cooling = 2.0;
        model.0.hvac.annual_heating_energy = 3.0;
        model.0.hvac.annual_cooling_energy = 4.0;
        model.0.mass.mass_energy_change_cumulative = 5.0;
        model.0.mass.envelope_mass_energy_change_cumulative = 6.0;
        model.0.mass.internal_mass_energy_change_cumulative = 7.0;

        model.reset_all_energy_tracking();

        assert!(approx_eq(model.get_peak_heating_power_kw(), 0.0, TOL));
        assert!(approx_eq(model.get_peak_cooling_power_kw(), 0.0, TOL));
        assert!(approx_eq(model.get_heating_energy_kwh(), 0.0, TOL));
        assert!(approx_eq(model.get_cooling_energy_kwh(), 0.0, TOL));
        assert!(approx_eq(model.get_mass_energy_change_joules(), 0.0, TOL));
        assert!(approx_eq(
            model.get_envelope_mass_energy_change_joules(),
            0.0,
            TOL
        ));
        assert!(approx_eq(
            model.get_internal_mass_energy_change_joules(),
            0.0,
            TOL
        ));
    }

    // ---------------------------------------------------------------------
    // cached_solar_position: cache identity invariants (Issue #1391).
    // ---------------------------------------------------------------------
    #[test]
    fn test_cached_solar_position_is_deterministic_for_same_key() {
        // Same (timestep, hour) key must return the identical cached value on
        // repeat calls — this is the regression guard for #1391.
        let mut model = ThermalModel::new(1);
        let sp1 = model.cached_solar_position(10, 2023, 6, 21, 12.0);
        let sp2 = model.cached_solar_position(10, 2023, 6, 21, 12.0);
        assert_eq!(sp1, sp2);
        // The cache must now contain an entry for this key.
        let hour_slot = (12.0_f64 * 2.0).round() as i32;
        assert!(model.0.solar.sun_pos_cache.contains_key(&(10, hour_slot)));
    }

    #[test]
    fn test_cached_solar_position_distinct_hour_slots_distinct_cache_entries() {
        // Two different hour slots produce two distinct cache keys; both must
        // be stored, and the values must be finite, valid solar positions.
        let mut model = ThermalModel::new(1);
        let _ = model.cached_solar_position(0, 2023, 3, 21, 9.0);
        let _ = model.cached_solar_position(0, 2023, 3, 21, 15.0);
        assert!(model.0.solar.sun_pos_cache.contains_key(&(0, 18)));
        assert!(model.0.solar.sun_pos_cache.contains_key(&(0, 30)));
        for sp in model.0.solar.sun_pos_cache.values() {
            assert!(sp.altitude_deg.is_finite());
            assert!(sp.azimuth_deg.is_finite());
            assert!(sp.zenith_deg.is_finite());
            // zenith = 90 − altitude (within float tolerance).
            assert!(approx_eq(sp.zenith_deg + sp.altitude_deg, 90.0, 1e-6));
        }
    }

    #[test]
    fn test_cached_solar_position_hour_slot_rounding() {
        // hour_slot = round(hour * 2). 12.0 → 24, 12.5 → 25, 11.7 → 23.
        let mut model = ThermalModel::new(1);
        let _ = model.cached_solar_position(5, 2023, 6, 21, 12.0);
        let _ = model.cached_solar_position(5, 2023, 6, 21, 12.5);
        let _ = model.cached_solar_position(5, 2023, 6, 21, 11.7);
        assert!(model.0.solar.sun_pos_cache.contains_key(&(5, 24)));
        assert!(model.0.solar.sun_pos_cache.contains_key(&(5, 25)));
        assert!(model.0.solar.sun_pos_cache.contains_key(&(5, 23)));
    }

    // ===== Issue #3291 (PR2.3–PR2.5): gauge-init helpers =====

    fn two_layer_construction() -> Construction {
        // ConstructionLayer::new(name, conductivity, density, specific_heat, thickness)
        Construction::new(vec![
            ConstructionLayer::new("Concrete", 1.73, 2243.0, 837.0, 0.10),
            ConstructionLayer::new("Foam", 0.03, 10.0, 1400.0, 0.05),
        ])
    }

    #[test]
    fn wall_spec_from_construction_reverses_layers() {
        let construction = two_layer_construction();
        let spec = wall_spec_from_construction(&construction, "test_wall");
        assert_eq!(spec.name, "test_wall");
        // Gauge expects exterior-to-interior; the spec stores
        // interior-to-exterior, so the first gauge layer is the LAST
        // construction layer ("Foam").
        assert_eq!(spec.layers.len(), 2);
        assert_eq!(spec.layers[0].name, "Foam");
        assert_eq!(spec.layers[1].name, "Concrete");
        // Spot-check a physical property round-trip.
        assert!((spec.layers[0].thickness - 0.05).abs() < 1e-12);
        assert!((spec.layers[1].conductivity - 1.73).abs() < 1e-12);
    }

    #[cfg(feature = "gauge-solver")]
    #[test]
    fn window_glass_wall_spec_matches_u_value() {
        use crate::validation::ashrae_140_cases::WindowSpec;
        let window_props = WindowSpec::double_clear_glass(); // U = 2.1
        let spec = window_glass_wall_spec(&window_props, 12.0);
        assert_eq!(spec.layers.len(), 1);
        // k = U × thickness; R = thickness / k = 1 / U.
        let expected_r = 1.0 / window_props.u_value;
        let actual_r = spec.layers[0].thickness / spec.layers[0].conductivity;
        assert!((actual_r - expected_r).abs() < 1e-9);
        // Degenerate U falls back to single-pane conductivity without
        // panicking and still yields a positive R.
        let zero_u = WindowSpec::new(
            0.0,
            0.0,
            0.0,
            crate::validation::ashrae_140_cases::GlassType::SingleClear,
        );
        let fallback = window_glass_wall_spec(&zero_u, 12.0);
        assert!(fallback.layers[0].conductivity > 0.0);
    }

    #[cfg(feature = "gauge-solver")]
    #[test]
    fn add_gauge_windows_is_noop_without_gauge_backend() {
        // Default (no gauge-solver feature) or FiveROneC selector leaves
        // `gauge_zone_solver = None`; add_gauge_windows must be a no-op.
        let spec = crate::validation::ashrae_140_cases::ASHRAE140Case::Case600.spec();
        let mut model = ThermalModel::<VectorField>::from_spec_with_selector(
            &spec,
            &crate::sim::thermal_selector::ThermalSelector {
                zone_solver: crate::sim::thermal_selector::ZoneSolverKind::FiveROneC,
                conduction_solver: crate::sim::thermal_selector::ConductionSolverKind::Default,
            },
        )
        .expect("FiveROneC selector must initialise");
        assert!(model.0.conduction.backend.gauge_zone_solver.is_none());
        // Must not panic and must leave the backend as None.
        model
            .add_gauge_windows(&spec)
            .expect("add_gauge_windows on empty backend is a no-op");
        assert!(model.0.conduction.backend.gauge_zone_solver.is_none());
    }

    #[cfg(feature = "gauge-solver")]
    #[test]
    fn enable_gauge_solver_multi_zone_rejects_single_zone() {
        // The multi-zone init is a fail-fast path for `num_zones < 2`.
        let mut spec = crate::validation::ashrae_140_cases::ASHRAE140Case::Case600.spec();
        spec.num_zones = 1;
        let mut model = ThermalModel::<VectorField>::from_spec_with_selector(
            &spec,
            &crate::sim::thermal_selector::ThermalSelector::default(),
        )
        .expect("default selector must initialise");
        let result = model.enable_gauge_solver_multi_zone(&spec);
        assert!(
            result.is_err(),
            "multi-zone init must reject num_zones == 1"
        );
        let msg = format!("{}", result.err().unwrap());
        assert!(msg.contains("num_zones >= 2"), "msg: {msg}");
    }

    #[cfg(feature = "gauge-solver")]
    #[test]
    fn enable_gauge_solver_fail_fast_on_missing_wall_spec() {
        // A CaseSpec whose WallSurfaces have wall_spec = None must fail
        // gauge init with a diagnostic identifying the zone/orientation.
        let spec = crate::validation::ashrae_140_cases::ASHRAE140Case::Case600.spec();
        let mut model = ThermalModel::<VectorField>::from_spec_with_selector(
            &spec,
            &crate::sim::thermal_selector::ThermalSelector {
                zone_solver: crate::sim::thermal_selector::ZoneSolverKind::FiveROneC,
                conduction_solver: crate::sim::thermal_selector::ConductionSolverKind::Default,
            },
        )
        .expect("FiveROneC selector must initialise");
        // Clear every wall_spec to simulate a caller that opted out.
        for zone_surfaces in model.0.solar.surfaces.iter_mut() {
            for surface in zone_surfaces.iter_mut() {
                surface.wall_spec = None;
            }
        }
        let result = model.enable_gauge_solver();
        assert!(result.is_err(), "missing wall_spec must fail gauge init");
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("wall_spec"),
            "diagnostic must identify the missing wall_spec; msg: {msg}"
        );
    }
}
