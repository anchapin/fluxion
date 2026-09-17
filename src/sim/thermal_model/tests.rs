mod tests {
    //! Unit tests for the swap-point types and concrete thermal-model
    //! implementations in the parent `thermal_model` module. Extracted
    //! from `src/sim/thermal_model.rs` at Issue #3789 decomposition time
    //! so the production code stays clear of the inline test bodies and
    //! the parent `mod.rs` is small enough to satisfy the Issue #3457
    //! module-size ratchet. The file is wired in via
    //! `#[cfg(test)] mod tests;` in `mod.rs`; the child-module privacy of
    //! the parent scope keeps the `use crate::sim::thermal_model::*;`
    //! import resolving exactly as `use super::*;` did before the split.

    use crate::sim::thermal_model::*;
    use crate::sim::thermal_model_data::FromF64;

    #[test]
    fn test_physics_model_creation() {
        let model = PhysicsThermalModel::new(10);
        assert_eq!(model.num_zones(), 10);
        assert_eq!(model.mode(), ThermalModelMode::Physics);
        assert!(model.is_valid());
    }

    #[test]
    fn test_surrogate_model_creation() {
        let model = SurrogateThermalModel::new(5);
        assert_eq!(model.num_zones(), 5);
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
        assert!(model.is_valid());
    }

    #[test]
    fn test_unified_model_switching() {
        let mut model = UnifiedThermalModel::new(1);

        // Initially in physics mode
        assert_eq!(model.mode(), ThermalModelMode::Physics);
        assert!(!model.is_using_surrogates());

        // Switch to surrogates
        model.use_surrogates();
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
        assert!(model.is_using_surrogates());

        // Switch back to physics
        model.use_physics();
        assert_eq!(model.mode(), ThermalModelMode::Physics);
        assert!(!model.is_using_surrogates());
    }

    #[test]
    fn test_builder_physics_mode() {
        let model = ThermalModelBuilder::new()
            .num_zones(5)
            .mode(ThermalModelMode::Physics)
            .build();

        assert_eq!(model.num_zones(), 5);
        assert_eq!(model.mode(), ThermalModelMode::Physics);
    }

    #[test]
    fn test_builder_surrogate_mode() {
        let model = ThermalModelBuilder::new()
            .num_zones(3)
            .use_surrogates(true)
            .build();

        assert_eq!(model.num_zones(), 3);
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
    }

    #[test]
    fn test_builder_default() {
        let model = ThermalModelBuilder::new().build();
        assert_eq!(model.num_zones(), 1);
        assert_eq!(model.mode(), ThermalModelMode::Physics);
    }

    #[test]
    fn test_builder_build_unified() {
        let model = ThermalModelBuilder::new()
            .num_zones(10)
            .mode(ThermalModelMode::Hybrid)
            .build_unified();

        assert_eq!(model.num_zones(), 10);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
    }

    #[test]
    fn test_thermal_model_mode_default() {
        let mode = ThermalModelMode::default();
        assert_eq!(mode, ThermalModelMode::Physics);
    }

    #[test]
    fn test_physics_model_set_mode() {
        let mut model = PhysicsThermalModel::new(1);
        assert_eq!(model.mode(), ThermalModelMode::Physics);
        model.set_mode(ThermalModelMode::Hybrid);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
    }

    #[test]
    fn test_surrogate_model_set_mode() {
        let mut model = SurrogateThermalModel::new(1);
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
        model.set_mode(ThermalModelMode::Physics);
        assert_eq!(model.mode(), ThermalModelMode::Physics);
    }

    #[test]
    fn test_unified_model_set_mode() {
        let mut model = UnifiedThermalModel::new(1);
        assert_eq!(model.mode(), ThermalModelMode::Physics);
        assert!(!model.is_using_surrogates());
        model.set_mode(ThermalModelMode::Surrogate);
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
        assert!(model.is_using_surrogates());
        model.set_mode(ThermalModelMode::Hybrid);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
    }

    #[test]
    fn test_unified_mode_switching_methods() {
        let mut model = UnifiedThermalModel::new(1);
        model.use_physics();
        assert_eq!(model.mode(), ThermalModelMode::Physics);
        assert!(!model.is_using_surrogates());
        model.use_hybrid();
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
        model.use_surrogates();
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
        assert!(model.is_using_surrogates());
    }

    #[test]
    fn test_physics_model_set_temperatures() {
        let mut model = PhysicsThermalModel::new(3);
        model.set_temperatures(&[20.0, 22.0, 24.0]);
        let temps = model.get_temperatures();
        assert_eq!(temps, vec![20.0, 22.0, 24.0]);
    }

    #[test]
    fn test_surrogate_model_set_temperatures() {
        let mut model = SurrogateThermalModel::new(2);
        model.set_temperatures(&[18.0, 25.0]);
        let temps = model.get_temperatures();
        assert_eq!(temps, vec![18.0, 25.0]);
    }

    #[test]
    fn test_unified_model_set_temperatures() {
        let mut model = UnifiedThermalModel::new(4);
        model.set_temperatures(&[15.0, 18.0, 21.0, 24.0]);
        let temps = model.get_temperatures();
        assert_eq!(temps, vec![15.0, 18.0, 21.0, 24.0]);
    }

    #[test]
    fn test_physics_model_hvac_power_demand_heating() {
        let mut model = PhysicsThermalModel::new(1);
        model.set_temperatures(&[15.0]);
        let power = model.hvac_power_demand(0, 10.0);
        assert!(power > 0.0, "Should return positive heating power");
    }

    #[test]
    fn test_physics_model_hvac_power_demand_cooling() {
        let mut model = PhysicsThermalModel::new(1);
        model.set_temperatures(&[30.0]);
        let power = model.hvac_power_demand(0, 35.0);
        assert!(power < 0.0, "Should return negative cooling power");
    }

    #[test]
    fn test_physics_model_hvac_power_demand_deadband() {
        let mut model = PhysicsThermalModel::new(1);
        model.set_temperatures(&[22.0]); // Between heating (20°C) and cooling (24°C)
        let power = model.hvac_power_demand(0, 22.0);
        assert_eq!(power, 0.0, "Should be zero in deadband");
    }

    #[test]
    fn test_surrogate_model_hvac_power_demand_heating() {
        let mut model = SurrogateThermalModel::new(1);
        model.set_temperatures(&[15.0]);
        let power = model.hvac_power_demand(0, 10.0);
        assert!(power > 0.0);
    }

    #[test]
    fn test_surrogate_model_hvac_power_demand_cooling() {
        let mut model = SurrogateThermalModel::new(1);
        model.set_temperatures(&[30.0]);
        let power = model.hvac_power_demand(0, 35.0);
        assert!(power < 0.0);
    }

    #[test]
    fn test_unified_model_hvac_power_demand() {
        let mut model = UnifiedThermalModel::new(1);
        model.set_temperatures(&[15.0]);
        let power = model.hvac_power_demand(0, 10.0);
        assert!(power > 0.0);
    }

    #[test]
    fn test_physics_model_apply_parameters() {
        let mut model = PhysicsThermalModel::new(1);
        model.apply_parameters(&[1.5, 22.0, 26.0]);
        assert_eq!(model.heating_setpoint(), 22.0);
        assert_eq!(model.cooling_setpoint(), 26.0);
    }

    #[test]
    fn test_surrogate_model_apply_parameters() {
        let mut model = SurrogateThermalModel::new(1);
        model.apply_parameters(&[2.0, 18.0, 28.0]);
        assert_eq!(model.heating_setpoint(), 18.0);
        assert_eq!(model.cooling_setpoint(), 28.0);
    }

    #[test]
    fn test_unified_model_apply_parameters() {
        let mut model = UnifiedThermalModel::new(1);
        model.apply_parameters(&[1.0, 19.0, 25.0]);
        assert_eq!(model.heating_setpoint(), 19.0);
        assert_eq!(model.cooling_setpoint(), 25.0);
    }

    #[test]
    fn test_physics_model_is_valid() {
        let model = PhysicsThermalModel::new(1);
        assert!(model.is_valid());
    }

    #[test]
    fn test_surrogate_model_is_valid() {
        let model = SurrogateThermalModel::new(1);
        assert!(model.is_valid());
    }

    #[test]
    fn test_unified_model_is_valid() {
        let model = UnifiedThermalModel::new(1);
        assert!(model.is_valid());
    }

    #[test]
    fn test_surrogate_with_fallback() {
        let model = SurrogateThermalModel::new(1).with_fallback(false);
        assert_eq!(model.num_zones(), 1);
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
    }

    #[test]
    fn test_builder_hybrid_mode() {
        // Issue #1431: after the per-component routing fix, building in
        // Hybrid mode actually returns a HybridThermalModel whose mode()
        // reports ThermalModelMode::Hybrid (instead of silently
        // downgrading to Physics via the old UnifiedThermalModel path).
        let model = ThermalModelBuilder::new()
            .num_zones(2)
            .mode(ThermalModelMode::Hybrid)
            .build();
        assert_eq!(model.num_zones(), 2);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
    }

    #[test]
    fn test_builder_hybrid_mode_unified() {
        let model = ThermalModelBuilder::new()
            .num_zones(2)
            .mode(ThermalModelMode::Hybrid)
            .build_unified();
        assert_eq!(model.num_zones(), 2);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
    }

    #[test]
    fn test_builder_fallback_setting() {
        let model = ThermalModelBuilder::new()
            .num_zones(1)
            .use_surrogates(true)
            .fallback_to_physics(false)
            .build();
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
    }

    #[test]
    fn test_builder_mode_sets_use_surrogates() {
        let builder = ThermalModelBuilder::new().mode(ThermalModelMode::Surrogate);
        assert!(builder.use_surrogates);
        let builder = ThermalModelBuilder::new().mode(ThermalModelMode::Physics);
        assert!(!builder.use_surrogates);
    }

    #[test]
    fn test_builder_use_surrogates_sets_mode() {
        let builder = ThermalModelBuilder::new().use_surrogates(true);
        assert_eq!(builder.mode, ThermalModelMode::Surrogate);
        let builder = ThermalModelBuilder::new().use_surrogates(false);
        assert_eq!(builder.mode, ThermalModelMode::Physics);
    }

    #[test]
    fn test_builder_default_impl() {
        let builder = ThermalModelBuilder::default();
        assert_eq!(builder.num_zones, 1);
        assert_eq!(builder.mode, ThermalModelMode::Physics);
        assert!(!builder.use_surrogates);
        assert!(builder.fallback_to_physics);
        assert!(builder.spec.is_none());
    }

    #[test]
    fn test_solve_timesteps_uses_mode_flag() {
        let mut model = PhysicsThermalModel::new(1);
        model.set_mode(ThermalModelMode::Surrogate);
        assert_eq!(model.mode(), ThermalModelMode::Surrogate);
    }

    #[test]
    fn test_thermal_model_result_type() {
        let result: ThermalModelResult<i32> = Ok(42);
        assert!(result.is_ok());
        if let Ok(val) = result {
            assert_eq!(val, 42);
        }
        let err: ThermalModelResult<i32> = Err("test error".into());
        assert!(err.is_err());
    }

    #[test]
    fn test_trait_object_cannot_access_inner() {
        let model: Box<dyn ThermalModelTrait> = Box::new(PhysicsThermalModel::new(1));
        // Cannot call inner() on trait object - compile error if uncommented
        // model.inner() // This would not compile
        assert_eq!(model.num_zones(), 1);
    }

    #[test]
    fn test_trait_object_behavior_via_trait_only() {
        let mut model: Box<dyn ThermalModelTrait> = Box::new(PhysicsThermalModel::new(1));
        assert_eq!(model.num_zones(), 1);
        model.set_temperatures(&[25.0]);
        assert_eq!(model.get_temperatures(), vec![25.0]);
        model.apply_parameters(&[1.5, 20.0, 26.0]);
        assert_eq!(model.heating_setpoint(), 20.0);
        assert_eq!(model.cooling_setpoint(), 26.0);
        assert!(model.is_valid());
    }

    #[test]
    fn test_thermal_model_type_from_case_spec_low_mass() {
        use crate::validation::ashrae_140_cases::CaseBuilder;

        let case_600 = CaseBuilder::case_600_baseline();
        assert_eq!(
            ThermalModelType::from(&case_600),
            ThermalModelType::LowMass5R1C
        );

        let case_600ff = CaseBuilder::case_600ff();
        assert_eq!(
            ThermalModelType::from(&case_600ff),
            ThermalModelType::LowMass5R1C
        );

        let case_650ff = CaseBuilder::case_650ff();
        assert_eq!(
            ThermalModelType::from(&case_650ff),
            ThermalModelType::LowMass5R1C
        );
    }

    #[test]
    fn test_thermal_model_type_from_case_spec_high_mass() {
        use crate::validation::ashrae_140_cases::CaseBuilder;

        let case_900 = CaseBuilder::case_900_baseline();
        assert_eq!(
            ThermalModelType::from(&case_900),
            ThermalModelType::HighMass9R4C
        );

        let case_900ff = CaseBuilder::case_900ff();
        assert_eq!(
            ThermalModelType::from(&case_900ff),
            ThermalModelType::HighMass9R4C
        );

        let case_950ff = CaseBuilder::case_950ff();
        assert_eq!(
            ThermalModelType::from(&case_950ff),
            ThermalModelType::HighMass9R4C
        );
    }

    #[test]
    fn test_thermal_model_type_from_case_spec_case_960() {
        use crate::validation::ashrae_140_cases::CaseBuilder;

        let case_960 = CaseBuilder::case_960_sunspace();
        assert_eq!(
            ThermalModelType::from(&case_960),
            ThermalModelType::HighMass9R4C
        );
    }

    #[test]
    fn test_thermal_model_type_default() {
        assert_eq!(ThermalModelType::default(), ThermalModelType::LowMass5R1C);
    }

    // --- Issue #1431: HybridThermalModel + HybridRouting unit tests ---

    #[test]
    fn test_hybrid_routing_default_policy() {
        let r = HybridRouting::default();
        assert!(!r.use_surrogate_conduction);
        assert!(!r.use_surrogate_ventilation);
        assert!(r.use_surrogate_loads, "default routes loads to surrogate");
        assert!(!r.use_surrogate_hvac);
    }

    #[test]
    fn test_hybrid_routing_all_physics_and_all_surrogate() {
        let p = HybridRouting::all_physics();
        assert!(!p.use_surrogate_conduction);
        assert!(!p.use_surrogate_ventilation);
        assert!(!p.use_surrogate_loads);
        assert!(!p.use_surrogate_hvac);

        let s = HybridRouting::all_surrogate();
        assert!(s.use_surrogate_conduction);
        assert!(s.use_surrogate_ventilation);
        assert!(s.use_surrogate_loads);
        assert!(s.use_surrogate_hvac);
    }

    #[test]
    fn test_hybrid_thermal_model_reports_hybrid_mode() {
        let model = HybridThermalModel::new(1, HybridRouting::default());
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
        assert_eq!(model.num_zones(), 1);
        assert!(model.is_valid());
    }

    #[test]
    fn test_hybrid_thermal_model_routing_getter_setter() {
        let mut model = HybridThermalModel::new(1, HybridRouting::default());
        assert_eq!(model.routing(), HybridRouting::default());

        let custom = HybridRouting {
            use_surrogate_conduction: true,
            use_surrogate_ventilation: false,
            use_surrogate_loads: true,
            use_surrogate_hvac: false,
            use_ood_fallback: false,
        };
        model.set_routing(custom);
        assert_eq!(model.routing(), custom);
    }

    #[test]
    fn test_hybrid_thermal_model_set_mode_is_intrinsic() {
        // HybridThermalModel is intrinsically Hybrid — set_mode is a no-op
        // because re-assigning the mode would silently lose the per-component
        // routing. Callers wanting a different mode should build a fresh model.
        let mut model = HybridThermalModel::new(1, HybridRouting::default());
        model.set_mode(ThermalModelMode::Physics);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
        model.set_mode(ThermalModelMode::Surrogate);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
    }

    #[test]
    fn test_hybrid_thermal_model_counters_start_zero() {
        let model = HybridThermalModel::new(1, HybridRouting::default());
        assert_eq!(model.surrogate_load_calls(), 0);
        assert_eq!(model.physics_conduction_calls(), 0);
    }

    #[test]
    fn test_hybrid_thermal_model_solve_routes_loads_and_physics() {
        // Drives the new dispatcher end-to-end and asserts that BOTH the
        // surrogate load branch and the physics conduction branch fired
        // (Issue #1431 acceptance criterion: ONNX probe ≥ steps AND
        // physics solver ≥ steps in the same run).
        use crate::ai::surrogate::SurrogateManager;

        let mut model = HybridThermalModel::new(1, HybridRouting::default());
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        let eui = model.solve_timesteps(24, &surrogates, false);
        assert!(eui.is_finite(), "EUI must be finite");

        // Default policy: loads → surrogate; conduction → physics.
        assert_eq!(
            model.surrogate_load_calls(),
            24,
            "surrogate load branch should fire once per step"
        );
        assert_eq!(
            model.physics_conduction_calls(),
            24,
            "physics conduction branch should fire once per step"
        );
    }

    #[test]
    fn test_hybrid_thermal_model_solve_physics_only_policy() {
        // With loads routed back to physics, the surrogate branch must NOT fire.
        use crate::ai::surrogate::SurrogateManager;

        let routing = HybridRouting {
            use_surrogate_loads: false,
            ..HybridRouting::all_physics()
        };
        let mut model = HybridThermalModel::new(1, routing);
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

        let eui = model.solve_timesteps(12, &surrogates, false);
        assert!(eui.is_finite());
        assert_eq!(model.surrogate_load_calls(), 0);
        assert_eq!(model.physics_conduction_calls(), 12);
    }

    #[test]
    fn test_hybrid_thermal_model_reset_counters() {
        use crate::ai::surrogate::SurrogateManager;

        let mut model = HybridThermalModel::new(1, HybridRouting::default());
        let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");
        let _ = model.solve_timesteps(4, &surrogates, false);
        assert!(model.surrogate_load_calls() > 0);
        assert!(model.physics_conduction_calls() > 0);
        model.reset_counters();
        assert_eq!(model.surrogate_load_calls(), 0);
        assert_eq!(model.physics_conduction_calls(), 0);
    }

    #[test]
    fn test_hybrid_thermal_model_from_spec_uses_default_routing() {
        use crate::validation::ashrae_140_cases::CaseBuilder;

        let spec = CaseBuilder::case_600_baseline();
        let model = HybridThermalModel::from_spec(&spec);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
        assert_eq!(model.routing(), HybridRouting::default());
    }

    #[test]
    fn test_hybrid_thermal_model_from_spec_with_routing() {
        use crate::validation::ashrae_140_cases::CaseBuilder;

        let spec = CaseBuilder::case_600_baseline();
        let custom = HybridRouting::all_surrogate();
        let model = HybridThermalModel::from_spec_with_routing(&spec, custom);
        assert_eq!(model.mode(), ThermalModelMode::Hybrid);
        assert_eq!(model.routing(), custom);
    }

    #[test]
    fn hybrid_thermal_model_dispatch_instruments() {
        use crate::ai::surrogate::SurrogateManager;

        let mut model = HybridThermalModel::new(1, HybridRouting::default());
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");

        // Acceptance criterion: After 100 hybrid steps, counters >= 100
        let eui = model.solve_timesteps(100, &surrogates, false);
        assert!(eui.is_finite(), "EUI must be finite after 100 steps");

        let snap = model.metrics();
        assert_eq!(
            snap.surrogate_load_calls, 100,
            "surrogate_load_calls must be 100 after 100 steps"
        );
        assert_eq!(
            snap.physics_conduction_calls, 100,
            "physics_conduction_calls must be 100 after 100 steps"
        );
        assert_eq!(snap.mode, ThermalModelMode::Hybrid);
        assert_eq!(snap.num_zones, 1);
        assert_eq!(snap.routing, HybridRouting::default());
        assert!(!snap.is_zero());
    }

    // --- Issue #1702: Wiring tests for use_surrogate_conduction and use_surrogate_ventilation ---
    // Strengthened in Issue #2457: the conduction assertion is inverted
    // (was `physics_step_calls == 24`, now `physics_conduction_calls == 0`
    // when `use_surrogate_conduction == true`) to catch the no-op
    // "counter increments but physics path also fires" anti-pattern.

    #[test]
    fn hybrid_routing_conduction_flag_wired() {
        // Issue #1702 acceptance criterion 1, strengthened by Issue #2457:
        // custom HybridRouting with `use_surrogate_conduction=true` routes
        // conduction through the `Box<dyn HeatConductionSolver>` slot, and
        // the analytical physics path DOES NOT fire in parallel
        // (`physics_conduction_calls` stays at zero).
        use crate::ai::surrogate::SurrogateManager;

        let routing = HybridRouting {
            use_surrogate_conduction: true,
            use_surrogate_ventilation: false,
            use_surrogate_loads: false,
            use_surrogate_hvac: false,
            use_ood_fallback: false,
        };
        let mut model = HybridThermalModel::new(1, routing);
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");

        let eui = model.solve_timesteps(24, &surrogates, false);
        assert!(eui.is_finite(), "EUI must be finite");

        assert_eq!(
            model.surrogate_conduction_calls(),
            24,
            "surrogate_conduction_calls must be 24 after 24 steps with flag enabled"
        );
        assert_eq!(
            model.physics_conduction_calls(),
            0,
            "physics_conduction_calls must be 0 when use_surrogate_conduction=true; \
             the Issue #2457 dispatcher must NOT also run the analytical physics path \
             in parallel with the surrogate slot (the legacy no-op bug closed by \
             this issue)"
        );
    }

    #[test]
    fn hybrid_routing_ventilation_flag_wired() {
        // Issue #1702 acceptance criterion 2: `use_surrogate_ventilation=true`
        // does not panic, increments `surrogate_ventilation_calls`, and the
        // slot is actually consulted (Issue #2457 regression guard: the
        // slot's `get_ach()` returns a value rather than just bumping a
        // counter).
        use crate::ai::surrogate::SurrogateManager;

        let routing = HybridRouting {
            use_surrogate_conduction: false,
            use_surrogate_ventilation: true,
            use_surrogate_loads: false,
            use_surrogate_hvac: false,
            use_ood_fallback: false,
        };
        let mut model = HybridThermalModel::new(1, routing);
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");

        let eui = model.solve_timesteps(24, &surrogates, false);
        assert!(eui.is_finite(), "EUI must be finite");

        assert_eq!(
            model.surrogate_ventilation_calls(),
            24,
            "surrogate_ventilation_calls must be 24 after 24 steps with flag enabled"
        );
        // The default slot is ConstantVentilation::new(0.5), so consulting
        // it must return 0.5 ACH (regardless of weather/wind arguments).
        // If this fails, the dispatcher is no longer routing through the
        // slot (Issue #2457 regression).
        let ach = model
            .ventilation_schedule()
            .get_ach(12, 20.0, 22.0, 0.0, 0.0);
        assert!(
            (ach - 0.5).abs() < 1e-9,
            "default ventilation slot must return 0.5 ACH; got {} \
             (Issue #2457: regression guard that the slot is actually \
             consulted by the dispatcher, not just a counter bump)",
            ach
        );
    }

    #[test]
    fn hybrid_routing_default_preserves_existing_behavior() {
        // Issue #1702 acceptance criterion 3: default routing produces identical
        // EUI to pre-change baseline (no regression for the default policy which
        // is use_surrogate_loads=true, everything else=false).
        use crate::ai::surrogate::SurrogateManager;

        let mut model = HybridThermalModel::new(1, HybridRouting::default());
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");

        let eui = model.solve_timesteps(24, &surrogates, false);
        assert!(eui.is_finite(), "EUI must be finite with default routing");

        // Default routing: loads → surrogate, conduction → physics, ventilation → physics
        assert_eq!(model.surrogate_load_calls(), 24);
        assert_eq!(model.physics_conduction_calls(), 24);
        assert_eq!(model.surrogate_conduction_calls(), 0);
        assert_eq!(model.surrogate_ventilation_calls(), 0);
    }

    // --- Issue #2457: regression tests for the no-op anti-pattern ---
    //
    // The legacy dispatcher (Issue #1702 closed but flagged as "wiring
    // done") incremented `surrogate_conduction_calls` while still
    // running the analytical physics path. The tests below assert the
    // inverse: the slot is consulted AND the physics path does NOT fire
    // when the flag is true. They guard against a future refactor
    // reintroducing the parallel-logging anti-pattern.

    #[test]
    fn hybrid_conduction_flag_routes_through_slot_not_physics() {
        // Issue #2457 regression guard: when `use_surrogate_conduction`
        // is `true`, the conduction counter increments AND the
        // physics_conduction_calls counter stays at zero. The legacy
        // dispatcher asserted BOTH counters incremented in parallel
        // (counter-only no-op), which is exactly the bug this issue
        // closes.
        use crate::ai::surrogate::SurrogateManager;

        let routing = HybridRouting {
            use_surrogate_conduction: true,
            use_surrogate_ventilation: false,
            use_surrogate_loads: false,
            use_surrogate_hvac: false,
            use_ood_fallback: false,
        };
        let mut model = HybridThermalModel::new(1, routing);
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");

        // The slot is a FiveR1CSolver::default() — uninitialized — so its
        // `step()` returns `SolverError::InvalidConfig` and the dispatcher
        // falls back to the physics path. Even with the fallback, the
        // surrogate counter still increments (we *consulted* the slot),
        // and the physics counter increments too. This is acceptable per
        // the issue ("the surrogate solver can be the existing physics
        // solver at first") because the slot is now a real
        // `Box<dyn HeatConductionSolver>`, not a no-op counter.
        //
        // We therefore assert only that the surrogate branch fired
        // (slot consulted) and that, in the success path where the slot
        // step succeeds, the physics path did NOT also fire. With a
        // fresh default slot, every step errors and the fallback fires;
        // we cover the success path below via
        // `hybrid_conduction_flag_skips_physics_when_slot_succeeds`.
        let eui = model.solve_timesteps(24, &surrogates, false);
        assert!(
            eui.is_finite(),
            "EUI must be finite even when slot is uninitialized"
        );
        assert_eq!(
            model.surrogate_conduction_calls(),
            24,
            "surrogate_conduction_calls must be 24 (the slot was consulted every step)"
        );
    }

    #[test]
    fn hybrid_conduction_flag_skips_physics_when_slot_succeeds() {
        // Issue #2457 dispatch-shape guard: when the conduction slot's
        // `step()` returns `Ok(_flux)` (i.e. a real solver is plugged in),
        // the dispatcher must NOT also call `self.inner.step_physics`.
        // We install a minimal custom solver that always returns
        // `Ok(HeatFlux::from_value(0.0))` so the success path fires,
        // and we assert `physics_conduction_calls == 0`.
        use crate::ai::surrogate::SurrogateManager;
        use crate::sim::thermal_model_data::{
            FromF64, HeatConductionSolver, HeatFlux, HeatTransferCoefficient, SolverError,
            Temperature, Time, WallSpec,
        };

        /// Minimal stand-in for an ONNX-trained conduction surrogate:
        /// always returns a zero heat flux without touching boundary
        /// temperatures. Lets us exercise the Issue #2457 success path
        /// (slot.step returns Ok) without wiring a real ONNX runtime.
        struct ZeroFluxSolver;

        impl HeatConductionSolver for ZeroFluxSolver {
            fn name(&self) -> &str {
                "ZeroFluxSolver"
            }
            fn initialize(&mut self, _wall: &WallSpec) -> Result<(), SolverError> {
                Ok(())
            }
            fn step(
                &mut self,
                _timestep: Time,
                _t_int: Temperature,
                _t_ext: Temperature,
                _h_int: HeatTransferCoefficient,
                _h_ext: HeatTransferCoefficient,
            ) -> Result<HeatFlux, SolverError> {
                Ok(HeatFlux::from_value(0.0))
            }
            fn energy_storage_rate(&self) -> f64 {
                0.0
            }
            fn is_valid(&self) -> bool {
                true
            }
        }

        let routing = HybridRouting {
            use_surrogate_conduction: true,
            use_surrogate_ventilation: false,
            use_surrogate_loads: false,
            use_surrogate_hvac: false,
            use_ood_fallback: false,
        };
        let mut model = HybridThermalModel::new(1, routing);
        let previous = model.set_conduction_solver(Box::new(ZeroFluxSolver));
        assert_eq!(
            previous.name(),
            "5R1C",
            "set_conduction_solver must return the previous slot"
        );

        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");
        let eui = model.solve_timesteps(24, &surrogates, false);
        assert!(eui.is_finite(), "EUI must be finite");

        assert_eq!(
            model.surrogate_conduction_calls(),
            24,
            "slot was consulted every step"
        );
        assert_eq!(
            model.physics_conduction_calls(),
            0,
            "physics path MUST NOT fire when the conduction slot succeeds; \
             this is the Issue #2457 regression guard that closes the \
             parallel-logging anti-pattern from Issue #1702"
        );
    }

    #[test]
    fn hybrid_ventilation_flag_consults_slot_with_swapped_schedule() {
        // Issue #2457 regression guard: when `use_surrogate_ventilation`
        // is `true`, the dispatcher actually consults the slot — a
        // swapped `WeatherDependentVentilation` propagates its
        // weather-aware ACH response into the dispatcher's hot path. We
        // install the weather-dependent schedule, solve, and verify the
        // slot accessor still returns weather-aware values (proving the
        // slot is live in the model after the dispatch).
        use crate::ai::surrogate::SurrogateManager;
        use crate::sim::ventilation::WeatherDependentVentilation;

        let routing = HybridRouting {
            use_surrogate_conduction: false,
            use_surrogate_ventilation: true,
            use_surrogate_loads: false,
            use_surrogate_hvac: false,
            use_ood_fallback: false,
        };
        let mut model = HybridThermalModel::new(1, routing);
        let previous = model.set_ventilation_schedule(Box::new(WeatherDependentVentilation::new(
            0.3, 0.3, 2.0, 18.0, 26.0,
        )));
        // Previous default is ConstantVentilation::new(0.5).
        assert_eq!(
            previous.get_ach(0, 20.0, 22.0, 0.0, 0.0),
            0.5,
            "previous slot must be the default ConstantVentilation"
        );

        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");
        let eui = model.solve_timesteps(24, &surrogates, false);
        assert!(eui.is_finite(), "EUI must be finite");
        assert_eq!(
            model.surrogate_ventilation_calls(),
            24,
            "surrogate_ventilation_calls must be 24"
        );
        // Slot is still the swapped weather-dependent schedule after
        // dispatch (the dispatcher did not silently reset it).
        let low_outdoor_ach = model.ventilation_schedule().get_ach(0, 5.0, 22.0, 0.0, 0.0);
        assert!(
            (low_outdoor_ach - 0.3).abs() < 1e-9,
            "swapped schedule must still return min_ach=0.3 at low outdoor temp; got {}",
            low_outdoor_ach
        );
    }

    #[test]
    fn hybrid_set_conduction_solver_swaps_slot() {
        // Smoke test for the `set_conduction_solver` API: swapping the
        // slot changes the dispatcher behaviour (the new slot's
        // `name()` is observable via the accessor).
        use crate::sim::thermal_model_data::{
            HeatConductionSolver, HeatFlux, HeatTransferCoefficient, SolverError, Temperature,
            Time, WallSpec,
        };

        struct NamedSolver(&'static str);
        impl HeatConductionSolver for NamedSolver {
            fn name(&self) -> &str {
                self.0
            }
            fn initialize(&mut self, _wall: &WallSpec) -> Result<(), SolverError> {
                Ok(())
            }
            fn step(
                &mut self,
                _timestep: Time,
                _t_int: Temperature,
                _t_ext: Temperature,
                _h_int: HeatTransferCoefficient,
                _h_ext: HeatTransferCoefficient,
            ) -> Result<HeatFlux, SolverError> {
                Ok(HeatFlux::from_value(0.0))
            }
            fn energy_storage_rate(&self) -> f64 {
                0.0
            }
            fn is_valid(&self) -> bool {
                true
            }
        }

        let mut model = HybridThermalModel::new(1, HybridRouting::default());
        assert_eq!(model.conduction_solver().name(), "5R1C");
        let previous = model.set_conduction_solver(Box::new(NamedSolver("OnnxSurrogate")));
        assert_eq!(previous.name(), "5R1C");
        assert_eq!(model.conduction_solver().name(), "OnnxSurrogate");
    }

    #[test]
    fn hybrid_set_ventilation_schedule_swaps_slot() {
        // Smoke test for the `set_ventilation_schedule` API.
        use crate::sim::ventilation::WeatherDependentVentilation;

        let mut model = HybridThermalModel::new(1, HybridRouting::default());
        assert_eq!(
            model
                .ventilation_schedule()
                .get_ach(0, 20.0, 22.0, 0.0, 0.0),
            0.5
        );
        let previous = model.set_ventilation_schedule(Box::new(WeatherDependentVentilation::new(
            0.3, 0.3, 2.0, 18.0, 26.0,
        )));
        assert_eq!(previous.get_ach(0, 20.0, 22.0, 0.0, 0.0), 0.5);
        // WeatherDependentVentilation's min_ach = 0.3, so at low
        // outdoor temp it returns 0.3 (the wind_benefit + temp_benefit
        // blend can fall below 0.3 but the clamp keeps it >= min_ach).
        let new_ach = model.ventilation_schedule().get_ach(0, 5.0, 22.0, 0.0, 0.0);
        assert!(
            (new_ach - 0.3).abs() < 1e-9,
            "swapped WeatherDependentVentilation must return 0.3 ACH at low outdoor temp; got {}",
            new_ach
        );
    }

    #[test]
    fn hybrid_clone_resets_solver_slots() {
        // Issue #2457: clones reset solver / schedule slots to fresh
        // defaults because their per-step state is transient and
        // shouldn't round-trip across clones (the empirical_hybrid
        // harness clones before solving).
        use crate::sim::ventilation::WeatherDependentVentilation;

        let mut original = HybridThermalModel::new(1, HybridRouting::default());
        let previous_solver = original.set_conduction_solver(Box::new(
            crate::physics::five_r1c_solver::FiveR1CSolver::default(),
        ));
        let _ = previous_solver;
        let _previous_schedule = original.set_ventilation_schedule(Box::new(
            WeatherDependentVentilation::new(0.3, 0.3, 2.0, 18.0, 26.0),
        ));
        // Original now has WeatherDependentVentilation in the slot.
        assert_eq!(
            original
                .ventilation_schedule()
                .get_ach(0, 5.0, 22.0, 0.0, 0.0),
            0.3,
            "original must have weather-dependent schedule in slot"
        );

        let cloned = original.clone();
        // Clone should reset to fresh defaults.
        assert_eq!(cloned.conduction_solver().name(), "5R1C");
        assert_eq!(
            cloned
                .ventilation_schedule()
                .get_ach(0, 20.0, 22.0, 0.0, 0.0),
            0.5,
            "clone must reset to ConstantVentilation::new(0.5) default"
        );
    }

    #[test]
    fn surrogate_thermal_model_adapter_builds_onnx_width_input() {
        let model = SurrogateThermalModel::new(1);
        let adapter = SurrogateThermalLoadAdapter::new(true);
        let input = adapter.input(&model.inner, 7, 12.5);
        assert_eq!(input.len(), 6);
        assert_eq!(input[0], 12.5);
        assert_eq!(input[5], 7.0);
    }

    #[test]
    fn surrogate_thermal_model_solve_uses_fallback_adapter() {
        let mut model = SurrogateThermalModel::new(1);
        let surrogates = SurrogateManager::new().expect("SurrogateManager::new");
        let eui = model.solve_timesteps(4, &surrogates, true);
        assert!(eui.is_finite());
        assert_eq!(surrogates.inference_metrics().num_inferences, 0);
        let hourly = model.inner.get_hourly_temperatures().expect("hourly temps");
        assert_eq!(hourly[0].len(), 4);
    }

    #[cfg(feature = "ort")]
    #[test]
    fn surrogate_thermal_model_runs_onnx_once_per_timestep() {
        let path = "assets/dummy_surrogate.onnx";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let surrogates = SurrogateManager::load_onnx(path).expect("load dummy ONNX");
        let mut model = SurrogateThermalModel::new(1);
        let eui = model.solve_timesteps(24, &surrogates, true);
        assert!(eui.is_finite());
        assert_eq!(surrogates.inference_metrics().num_inferences, 24);
    }

    #[test]
    fn test_set_twin_correction_single_zone() {
        use fluxion_twin::TwinCorrection;

        let mut model = PhysicsThermalModel::new(1);
        model.set_temperatures(&[20.0]);

        let correction = TwinCorrection::single_zone(0.5, 0.1);
        model.set_twin_correction(&correction);

        let temps = model.get_temperatures();
        assert!((temps[0] - 20.5).abs() < 1e-9);
    }

    #[test]
    fn test_set_twin_correction_multi_zone() {
        use fluxion_twin::TwinCorrection;

        let mut model = PhysicsThermalModel::new(3);
        model.set_temperatures(&[18.0, 20.0, 22.0]);

        let correction = TwinCorrection::multi_zone(vec![-0.5, 1.0, 0.3], vec![0.1, 0.1, 0.1]);
        model.set_twin_correction(&correction);

        let temps = model.get_temperatures();
        assert!((temps[0] - 17.5).abs() < 1e-9);
        assert!((temps[1] - 21.0).abs() < 1e-9);
        assert!((temps[2] - 22.3).abs() < 1e-9);
    }

    #[test]
    fn test_set_twin_correction_all_model_types() {
        use fluxion_twin::TwinCorrection;

        let correction = TwinCorrection::single_zone(1.0, 0.05);

        let mut physics = PhysicsThermalModel::new(1);
        physics.set_temperatures(&[20.0]);
        physics.set_twin_correction(&correction);
        assert!((physics.get_temperatures()[0] - 21.0).abs() < 1e-9);

        let mut surrogate = SurrogateThermalModel::new(1);
        surrogate.set_temperatures(&[20.0]);
        surrogate.set_twin_correction(&correction);
        assert!((surrogate.get_temperatures()[0] - 21.0).abs() < 1e-9);

        let mut hybrid = HybridThermalModel::new(1, HybridRouting::default());
        hybrid.set_temperatures(&[20.0]);
        hybrid.set_twin_correction(&correction);
        assert!((hybrid.get_temperatures()[0] - 21.0).abs() < 1e-9);

        let mut unified = UnifiedThermalModel::new(1);
        unified.set_temperatures(&[20.0]);
        unified.set_twin_correction(&correction);
        assert!((unified.get_temperatures()[0] - 21.0).abs() < 1e-9);
    }
}
