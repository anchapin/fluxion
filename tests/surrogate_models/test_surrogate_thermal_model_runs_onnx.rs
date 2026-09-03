use fluxion::ai::surrogate::SurrogateManager;
#[cfg(feature = "ort")]
use fluxion::sim::thermal_model::PhysicsThermalModel;
use fluxion::sim::thermal_model::{
    SurrogateThermalModel, ThermalModelBuilder, ThermalModelMode, ThermalModelTrait,
};

#[cfg(feature = "ort")]
const DUMMY_ONNX_MODEL: &str = "assets/dummy_surrogate.onnx";

#[test]
fn surrogate_thermal_model_builder_swap_is_one_liner() {
    let model = ThermalModelBuilder::new()
        .mode(ThermalModelMode::Surrogate)
        .build();
    assert_eq!(model.mode(), ThermalModelMode::Surrogate);
    assert!(model.is_valid());
}

#[test]
fn surrogate_thermal_model_fallback_path_is_finite_without_onnx() {
    let mut model = SurrogateThermalModel::new(1);
    let manager = SurrogateManager::new().expect("SurrogateManager::new");
    let eui = model.solve_timesteps(24, &manager, true);
    assert!(eui.is_finite());
    assert_eq!(manager.inference_metrics().num_inferences, 0);
}

#[cfg(feature = "ort")]
#[test]
fn surrogate_thermal_model_runs_onnx_once_per_annual_timestep() {
    if !std::path::Path::new(DUMMY_ONNX_MODEL).exists() {
        return;
    }
    let manager = SurrogateManager::load_onnx(DUMMY_ONNX_MODEL).expect("load dummy ONNX");
    let mut model = SurrogateThermalModel::new(1);
    let eui = model.solve_timesteps(8760, &manager, true);
    assert!(eui.is_finite());
    assert_eq!(manager.inference_metrics().num_inferences, 8760);
}

#[cfg(feature = "ort")]
#[test]
fn surrogate_thermal_model_onnx_result_differs_from_physics() {
    if !std::path::Path::new(DUMMY_ONNX_MODEL).exists() {
        return;
    }
    let manager = SurrogateManager::load_onnx(DUMMY_ONNX_MODEL).expect("load dummy ONNX");
    let mut physics = PhysicsThermalModel::new(1);
    let mut surrogate = SurrogateThermalModel::new(1);
    let physics_eui = physics.solve_timesteps(168, &manager, false);
    let surrogate_eui = surrogate.solve_timesteps(168, &manager, true);
    assert!(physics_eui.is_finite());
    assert!(surrogate_eui.is_finite());
    assert!((physics_eui - surrogate_eui).abs() > 1e-9);
}
