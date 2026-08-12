//! Python bindings for Fluxion Model interior structs (Issue #1812).
//!
//! This module exposes the core geometric, material, and HVAC structs that make
//! up a `FluxionModel` to Python so users can write rich Measures that read and
//! mutate the building's topology (zones, surfaces, materials, HVAC systems,
//! shading devices).
//!
//! # Lifetime / ownership story (PyO3 memory safety)
//!
//! Python objects exposed via PyO3 follow the **snapshot / owned-value** model:
//!
//! - Each [`PyZone`], [`PySurface`], [`PyMaterial`], [`PyHVACSystem`] holds
//!   **cloned** primitive data (floats, ints, strings, owned child structs).
//!   There are *no* references back into the Rust [`crate::sim::engine::ThermalModel`].
//! - Calling [`Model::zones`](crate::Model::zones) /
//!   [`Model::surfaces`](crate::Model::surfaces) **clones** the current zone /
//!   surface data from the model into fresh PyO3-owned heap objects. The model
//!   is borrowed immutably for the duration of the clone and released before
//!   the call returns.
//! - Python garbage collection of any returned zone / surface / material
//!   objects only frees those standalone objects — it cannot invalidate the
//!   parent model because no references are held back into it.
//! - Conversely, the parent model can be dropped, re-allocated, mutated, or
//!   re-simulated while Python still holds references to previously returned
//!   `Zone` / `Surface` snapshots. Those snapshots remain internally
//!   consistent and safe to read.
//!
//! The cost of this safety is that **mutations on a snapshot are not
//! automatically propagated back to the model**. To push a snapshot back, use
//! the model's `set_surfaces(...)` / `set_zones(...)` methods (which clone the
//! data back into model storage). This trade-off is intentional and matches
//! the existing pattern from PRs #1795 (9R4C bindings) and #1797 (HVAC config
//! bindings).
//!
//! # Iteration
//!
//! `model.zones()` and `model.surfaces()` return Python lists, so iteration
//! (`for z in model.zones(): ...`) works out of the box via CPython's list
//! iterator protocol. We do not implement a custom `__iter__` because it
//! would add complexity without changing the user-visible semantics.

use crate::physics::cta::VectorField;
use crate::sim::construction::{ConstructionLayer, WallSurface};
use crate::sim::engine::ThermalModel;
use crate::sim::shading::{Overhang, ShadeFin};
use crate::validation::ashrae140::HVACSystem;
use fluxion_core::ashrae_cases::{Orientation, ShadingDevice, ShadingType};
use pyo3::prelude::*;

// =============================================================================
// Orientation enum
// =============================================================================

/// Compass orientation of a surface.
///
/// Python-side mirror of [`fluxion_core::ashrae_cases::Orientation`]. Equality
/// and hashing are preserved via `eq, eq_int` so users can compare with `==`
/// and `!=` (e.g. `s.orientation == Orientation.South`).
#[pyclass(name = "Orientation", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyOrientation {
    North,
    East,
    South,
    West,
    /// Roof / upward-facing surface.
    Up,
    /// Floor / downward-facing surface.
    Down,
    /// Horizontal surface (treated separately from Up/Down).
    Horizontal,
}

impl From<Orientation> for PyOrientation {
    fn from(o: Orientation) -> Self {
        match o {
            Orientation::North => PyOrientation::North,
            Orientation::East => PyOrientation::East,
            Orientation::South => PyOrientation::South,
            Orientation::West => PyOrientation::West,
            Orientation::Up => PyOrientation::Up,
            Orientation::Down => PyOrientation::Down,
            Orientation::Horizontal => PyOrientation::Horizontal,
        }
    }
}

impl From<PyOrientation> for Orientation {
    fn from(o: PyOrientation) -> Self {
        match o {
            PyOrientation::North => Orientation::North,
            PyOrientation::East => Orientation::East,
            PyOrientation::South => Orientation::South,
            PyOrientation::West => Orientation::West,
            PyOrientation::Up => Orientation::Up,
            PyOrientation::Down => Orientation::Down,
            PyOrientation::Horizontal => Orientation::Horizontal,
        }
    }
}

#[pymethods]
impl PyOrientation {
    /// Returns the canonical short prefix: "N", "S", "E", "W", "Up", "Down", "H".
    #[getter]
    fn prefix(&self) -> &'static str {
        Orientation::from(*self).prefix()
    }

    /// Returns the azimuth in degrees (0° = South, clockwise) per ASHRAE 140.
    #[getter]
    fn azimuth_deg(&self) -> f64 {
        Orientation::from(*self).azimuth()
    }

    fn __repr__(&self) -> &'static str {
        match self {
            PyOrientation::North => "Orientation.North",
            PyOrientation::East => "Orientation.East",
            PyOrientation::South => "Orientation.South",
            PyOrientation::West => "Orientation.West",
            PyOrientation::Up => "Orientation.Up",
            PyOrientation::Down => "Orientation.Down",
            PyOrientation::Horizontal => "Orientation.Horizontal",
        }
    }
}

// =============================================================================
// ShadingDevice
// =============================================================================

/// Shading device attached to a surface (overhang, fins, or both).
#[pyclass(name = "ShadingDevice", from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct PyShadingDevice {
    /// Type of shading (overhang, fins, both, or none).
    pub shading_type: PyShadingType,
    /// Depth of the overhang in meters (m).
    pub overhang_depth: f64,
    /// Width of the shade fins in meters (m).
    pub fin_width: f64,
    /// Height at which the device is mounted in meters (m).
    pub mounting_height: f64,
}

#[pyclass(name = "ShadingType", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyShadingType {
    None,
    Overhang,
    Fins,
    OverhangAndFins,
}

impl From<ShadingType> for PyShadingType {
    fn from(t: ShadingType) -> Self {
        match t {
            ShadingType::None => PyShadingType::None,
            ShadingType::Overhang => PyShadingType::Overhang,
            ShadingType::Fins => PyShadingType::Fins,
            ShadingType::OverhangAndFins => PyShadingType::OverhangAndFins,
        }
    }
}

impl From<PyShadingType> for ShadingType {
    fn from(t: PyShadingType) -> Self {
        match t {
            PyShadingType::None => ShadingType::None,
            PyShadingType::Overhang => ShadingType::Overhang,
            PyShadingType::Fins => ShadingType::Fins,
            PyShadingType::OverhangAndFins => ShadingType::OverhangAndFins,
        }
    }
}

impl From<ShadingDevice> for PyShadingDevice {
    fn from(d: ShadingDevice) -> Self {
        Self {
            shading_type: d.shading_type.into(),
            overhang_depth: d.overhang_depth,
            fin_width: d.fin_width,
            mounting_height: d.mounting_height,
        }
    }
}

impl From<PyShadingDevice> for ShadingDevice {
    fn from(d: PyShadingDevice) -> Self {
        Self {
            shading_type: d.shading_type.into(),
            overhang_depth: d.overhang_depth,
            fin_width: d.fin_width,
            mounting_height: d.mounting_height,
        }
    }
}

#[pymethods]
impl PyShadingDevice {
    /// Create a no-shading device.
    #[staticmethod]
    fn none() -> Self {
        ShadingDevice::none().into()
    }

    /// Create an overhang-only shading device.
    ///
    /// # Arguments
    /// * `depth` - depth of overhang in meters (m)
    /// * `height` - mounting height (top of overhang above window) in meters (m)
    #[staticmethod]
    fn overhang(depth: f64, height: f64) -> Self {
        ShadingDevice::overhang(depth, height).into()
    }

    /// Create a fins-only shading device.
    #[staticmethod]
    fn fins(width: f64) -> Self {
        ShadingDevice::fins(width).into()
    }

    /// Create an overhang+fins combined shading device.
    #[staticmethod]
    fn overhang_and_fins(overhang_depth: f64, fin_width: f64, height: f64) -> Self {
        ShadingDevice::overhang_and_fins(overhang_depth, fin_width, height).into()
    }

    fn __repr__(&self) -> String {
        format!(
            "ShadingDevice(type={:?}, depth={:.3}, fin_width={:.3}, height={:.3})",
            self.shading_type, self.overhang_depth, self.fin_width, self.mounting_height
        )
    }
}

// =============================================================================
// Material (snapshot of ConstructionLayer)
// =============================================================================

/// Single material layer in a construction assembly.
///
/// This is the Python-side mirror of [`ConstructionLayer`]. Each `Material`
/// is an owned snapshot — mutations on it do not propagate back to the parent
/// model unless explicitly written via the model's setters.
#[pyclass(name = "Material", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMaterial {
    /// Material name (e.g. "Gypsum", "Concrete").
    #[pyo3(get, set)]
    pub name: String,
    /// Thermal conductivity (W/m·K).
    #[pyo3(get, set)]
    pub conductivity: f64,
    /// Density (kg/m³).
    #[pyo3(get, set)]
    pub density: f64,
    /// Specific heat capacity (J/kg·K).
    #[pyo3(get, set)]
    pub specific_heat: f64,
    /// Thickness (m).
    #[pyo3(get, set)]
    pub thickness: f64,
}

impl From<&ConstructionLayer> for PyMaterial {
    fn from(l: &ConstructionLayer) -> Self {
        Self {
            name: l.name.clone(),
            conductivity: l.conductivity,
            density: l.density,
            specific_heat: l.specific_heat,
            thickness: l.thickness,
        }
    }
}

impl From<PyMaterial> for ConstructionLayer {
    fn from(m: PyMaterial) -> Self {
        ConstructionLayer::new(
            m.name,
            m.conductivity,
            m.density,
            m.specific_heat,
            m.thickness,
        )
    }
}

#[pymethods]
impl PyMaterial {
    /// Create a new Material.
    #[new]
    #[pyo3(signature = (name, conductivity, density, specific_heat, thickness))]
    fn new(
        name: String,
        conductivity: f64,
        density: f64,
        specific_heat: f64,
        thickness: f64,
    ) -> Self {
        Self {
            name,
            conductivity,
            density,
            specific_heat,
            thickness,
        }
    }

    /// R-value of this single layer (m²·K/W).
    fn r_value(&self) -> f64 {
        if self.conductivity > 0.0 {
            self.thickness / self.conductivity
        } else {
            0.0
        }
    }

    /// Thermal capacitance per unit area (J/m²·K).
    fn thermal_capacitance_per_area(&self) -> f64 {
        self.density * self.thickness * self.specific_heat
    }

    fn __repr__(&self) -> String {
        format!(
            "Material(name='{}', k={:.4}, rho={:.1}, cp={:.1}, t={:.4})",
            self.name, self.conductivity, self.density, self.specific_heat, self.thickness
        )
    }
}

// =============================================================================
// Surface (snapshot of WallSurface)
// =============================================================================

/// Building surface (opaque wall, window, roof, or floor).
///
/// Surfaces are organized per zone in the underlying thermal model. This
/// snapshot type captures the visible surface state (`area`, `window_area`,
/// `u_value`, `orientation`) and the existing shading devices
/// (`overhang`/`fins`). Appending a [`PyShadingDevice`] mutates only the
/// snapshot — see module docs for the lifetime story.
#[pyclass(name = "Surface", from_py_object)]
#[derive(Clone, Debug)]
pub struct PySurface {
    /// Total surface area in square meters (m²).
    #[pyo3(get, set)]
    pub area: f64,
    /// Window area on this surface (m²).
    #[pyo3(get, set)]
    pub window_area: f64,
    /// Thermal transmittance (U-value) of the opaque portion in W/m²·K.
    #[pyo3(get, set)]
    pub u_value: f64,
    /// Compass orientation.
    #[pyo3(get, set)]
    pub orientation: PyOrientation,
    /// Optional horizontal overhang (depth in meters).
    #[pyo3(get, set)]
    pub overhang_depth: Option<f64>,
    /// Optional overhang height above window in meters.
    #[pyo3(get, set)]
    pub overhang_height: Option<f64>,
    /// Optional shade fin width in meters.
    #[pyo3(get, set)]
    pub fin_width: Option<f64>,
    /// Shading devices attached to this surface (e.g. overhangs, fins).
    #[pyo3(get)]
    pub shading_devices: Vec<PyShadingDevice>,
}

impl From<&WallSurface> for PySurface {
    fn from(s: &WallSurface) -> Self {
        let (overhang_depth, overhang_height) = match s.overhang {
            Some(o) => (Some(o.depth), Some(o.distance_above)),
            None => (None, None),
        };
        let fin_width = s.fins.first().map(|f| f.depth);

        // Build the canonical shading_devices list from the underlying overhang + fins.
        let mut shading_devices: Vec<PyShadingDevice> = Vec::new();
        let has_overhang = s.overhang.is_some();
        let has_fins = !s.fins.is_empty();
        match (has_overhang, has_fins) {
            (true, false) => {
                if let Some(o) = s.overhang {
                    shading_devices.push(PyShadingDevice {
                        shading_type: PyShadingType::Overhang,
                        overhang_depth: o.depth,
                        fin_width: 0.0,
                        mounting_height: o.distance_above,
                    });
                }
            }
            (false, true) => {
                let width = s.fins.first().map(|f| f.depth).unwrap_or(0.0);
                shading_devices.push(PyShadingDevice {
                    shading_type: PyShadingType::Fins,
                    overhang_depth: 0.0,
                    fin_width: width,
                    mounting_height: 0.0,
                });
            }
            (true, true) => {
                if let Some(o) = s.overhang {
                    let width = s.fins.first().map(|f| f.depth).unwrap_or(0.0);
                    shading_devices.push(PyShadingDevice {
                        shading_type: PyShadingType::OverhangAndFins,
                        overhang_depth: o.depth,
                        fin_width: width,
                        mounting_height: o.distance_above,
                    });
                }
            }
            (false, false) => {}
        }

        Self {
            area: s.area,
            window_area: s.window_area,
            u_value: s.u_value,
            orientation: s.orientation.into(),
            overhang_depth,
            overhang_height,
            fin_width,
            shading_devices,
        }
    }
}

#[pymethods]
impl PySurface {
    /// Create a new Surface with sensible defaults for window_area, overhang, and fins.
    #[new]
    #[pyo3(signature = (area, u_value, orientation, window_area=0.0))]
    fn new(area: f64, u_value: f64, orientation: PyOrientation, window_area: f64) -> Self {
        Self {
            area,
            window_area,
            u_value,
            orientation,
            overhang_depth: None,
            overhang_height: None,
            fin_width: None,
            shading_devices: Vec::new(),
        }
    }

    /// Append a shading device to this surface (snapshot mutation only).
    ///
    /// The shading is added to the snapshot's `shading_devices` list. To
    /// persist the shading back to the parent model, the caller must invoke
    /// `model.set_surfaces(...)` with the updated list.
    fn append_shading(&mut self, device: PyShadingDevice) {
        self.shading_devices.push(device);
    }

    /// Build a `ShadingDevice::overhang` and append it. Convenience wrapper.
    fn add_overhang(&mut self, depth: f64, height: f64) {
        self.overhang_depth = Some(depth);
        self.overhang_height = Some(height);
        self.shading_devices
            .push(PyShadingDevice::overhang(depth, height));
    }

    /// Build a `ShadingDevice::fins` and append it. Convenience wrapper.
    fn add_fins(&mut self, width: f64) {
        self.fin_width = Some(width);
        self.shading_devices.push(PyShadingDevice::fins(width));
    }

    /// Total area including any windows (currently = `area`, kept as a method
    /// for forward-compatibility with multi-layer assemblies).
    fn total_area(&self) -> f64 {
        self.area
    }

    fn __repr__(&self) -> String {
        format!(
            "Surface(area={:.2}, u={:.3}, orientation={:?}, window_area={:.2}, shading={})",
            self.area,
            self.u_value,
            self.orientation,
            self.window_area,
            self.shading_devices.len()
        )
    }
}

// =============================================================================
// Zone (per-zone snapshot of thermal model state)
// =============================================================================

/// Single thermal zone in the building.
///
/// A `Zone` snapshot captures the runtime state of one zone from the
/// parent model: its index, current air temperature, zone floor area, and
/// the surfaces that bound it. Like [`PySurface`], this is an owned
/// snapshot — see the module-level docs for the lifetime story.
#[pyclass(name = "Zone", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyZone {
    /// Zero-based zone index in the parent model.
    #[pyo3(get)]
    pub index: usize,
    /// Current zone air temperature in °C.
    #[pyo3(get, set)]
    pub temperature: f64,
    /// Floor area in square meters (m²).
    #[pyo3(get, set)]
    pub area: f64,
    /// Heating setpoint in °C.
    #[pyo3(get, set)]
    pub heating_setpoint: f64,
    /// Cooling setpoint in °C.
    #[pyo3(get, set)]
    pub cooling_setpoint: f64,
    /// Whether HVAC is enabled for this zone.
    #[pyo3(get, set)]
    pub hvac_enabled: bool,
    /// Surfaces bounding this zone (opaque + windows).
    #[pyo3(get)]
    pub surfaces: Vec<PySurface>,
}

#[pymethods]
impl PyZone {
    /// Create a new Zone snapshot from explicit fields (mostly for tests).
    #[new]
    #[pyo3(signature = (index, temperature, area, heating_setpoint=20.0, cooling_setpoint=24.0, hvac_enabled=true, surfaces=Vec::new()))]
    fn new(
        index: usize,
        temperature: f64,
        area: f64,
        heating_setpoint: f64,
        cooling_setpoint: f64,
        hvac_enabled: bool,
        surfaces: Vec<PySurface>,
    ) -> Self {
        Self {
            index,
            temperature,
            area,
            heating_setpoint,
            cooling_setpoint,
            hvac_enabled,
            surfaces,
        }
    }

    /// Number of surfaces in this zone.
    fn surface_count(&self) -> usize {
        self.surfaces.len()
    }

    /// Total area of all surfaces (m²).
    fn total_surface_area(&self) -> f64 {
        self.surfaces.iter().map(|s| s.area).sum()
    }

    /// Return surfaces matching a given orientation (e.g. `Orientation.South`).
    ///
    /// Convenience wrapper for the common Measure pattern:
    /// `zone.surfaces_with_orientation(Orientation.South)`.
    fn surfaces_with_orientation(&self, orientation: PyOrientation) -> Vec<PySurface> {
        self.surfaces
            .iter()
            .filter(|s| s.orientation == orientation)
            .cloned()
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Zone(idx={}, T={:.2}, area={:.2}, sp=[{:.1}/{:.1}], hvac={}, n_surfaces={})",
            self.index,
            self.temperature,
            self.area,
            self.heating_setpoint,
            self.cooling_setpoint,
            self.hvac_enabled,
            self.surfaces.len()
        )
    }
}

// =============================================================================
// HVACSystem (snapshot of validation::ashrae140::HVACSystem)
// =============================================================================

/// HVAC system configuration snapshot.
///
/// Mirrors [`crate::validation::ashrae140::HVACSystem`] (the validation-layer
/// HVAC type) with the most commonly-used ASHRAE 140 fields exposed. Used by
/// Measures that want to inspect or tweak the heating / cooling plant.
#[pyclass(name = "HVACSystem", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHVACSystem {
    /// Heating capacity (W).
    #[pyo3(get, set)]
    pub heating_capacity: f64,
    /// Cooling capacity (W).
    #[pyo3(get, set)]
    pub cooling_capacity: f64,
    /// Heating coefficient of performance (W_th/W_e).
    #[pyo3(get, set)]
    pub cop_heating: f64,
    /// Cooling coefficient of performance (W_th/W_e).
    #[pyo3(get, set)]
    pub cop_cooling: f64,
    /// Number of stages (1 = single-stage, 2 = two-stage, etc.).
    #[pyo3(get, set)]
    pub stages: u32,
    /// Minimum outdoor temperature for HVAC operation (°C).
    #[pyo3(get, set)]
    pub min_outdoor_temp: f64,
    /// Maximum outdoor temperature for HVAC operation (°C).
    #[pyo3(get, set)]
    pub max_outdoor_temp: f64,
    /// VAV (variable air volume) enabled flag.
    #[pyo3(get, set)]
    pub vav_enabled: bool,
    /// Economizer enabled flag.
    #[pyo3(get, set)]
    pub economizer_enabled: bool,
    /// Supply air temperature (°C).
    #[pyo3(get, set)]
    pub supply_air_temp: f64,
}

impl From<&HVACSystem> for PyHVACSystem {
    fn from(h: &HVACSystem) -> Self {
        Self {
            heating_capacity: h.heating_capacity,
            cooling_capacity: h.cooling_capacity,
            cop_heating: h.cop_heating,
            cop_cooling: h.cop_cooling,
            stages: h.stages,
            min_outdoor_temp: h.min_outdoor_temp,
            max_outdoor_temp: h.max_outdoor_temp,
            vav_enabled: h.vav_enabled,
            economizer_enabled: h.economizer_enabled,
            supply_air_temp: h.supply_air_temp,
        }
    }
}

#[pymethods]
impl PyHVACSystem {
    /// Create a new HVACSystem with default parameters.
    #[new]
    #[pyo3(signature = (
        heating_capacity=10000.0,
        cooling_capacity=8000.0,
        cop_heating=3.0,
        cop_cooling=3.2,
        stages=1,
        min_outdoor_temp=-10.0,
        max_outdoor_temp=40.0,
        vav_enabled=false,
        economizer_enabled=false,
        supply_air_temp=13.0,
    ))]
    // PyO3 `#[new]` constructors must accept a flat argument list (Python doesn't
    // expose keyword-only args for `__init__` ergonomically), so the 10-arg signature
    // is intentional and required by the bindings API contract.
    #[allow(clippy::too_many_arguments)]
    fn new(
        heating_capacity: f64,
        cooling_capacity: f64,
        cop_heating: f64,
        cop_cooling: f64,
        stages: u32,
        min_outdoor_temp: f64,
        max_outdoor_temp: f64,
        vav_enabled: bool,
        economizer_enabled: bool,
        supply_air_temp: f64,
    ) -> Self {
        Self {
            heating_capacity,
            cooling_capacity,
            cop_heating,
            cop_cooling,
            stages,
            min_outdoor_temp,
            max_outdoor_temp,
            vav_enabled,
            economizer_enabled,
            supply_air_temp,
        }
    }

    /// Returns the steady-state heating electrical input at full load (W_e).
    fn heating_electrical_input(&self) -> f64 {
        if self.cop_heating > 0.0 {
            self.heating_capacity / self.cop_heating
        } else {
            0.0
        }
    }

    /// Returns the steady-state cooling electrical input at full load (W_e).
    fn cooling_electrical_input(&self) -> f64 {
        if self.cop_cooling > 0.0 {
            self.cooling_capacity / self.cop_cooling
        } else {
            0.0
        }
    }

    /// Whether this HVAC system can operate at the given outdoor temperature (°C).
    fn can_operate_at(&self, outdoor_temp: f64) -> bool {
        outdoor_temp >= self.min_outdoor_temp && outdoor_temp <= self.max_outdoor_temp
    }

    fn __repr__(&self) -> String {
        format!(
            "HVACSystem(Q_h={:.0} W, Q_c={:.0} W, COP_h={:.2}, COP_c={:.2}, stages={})",
            self.heating_capacity,
            self.cooling_capacity,
            self.cop_heating,
            self.cop_cooling,
            self.stages
        )
    }
}

// =============================================================================
// Snapshot builders (called from the Model methods in lib.rs)
// =============================================================================

/// Build a [`PyZone`] snapshot from a [`ThermalModel<VectorField>`] at zone index `i`.
///
/// The model is borrowed immutably for the duration of this call; the returned
/// `PyZone` does not retain any reference into the model.
pub fn zone_from_model(model: &ThermalModel<VectorField>, idx: usize) -> PyZone {
    let temp = if idx < model.temperatures.len() {
        model.temperatures.as_slice()[idx]
    } else {
        0.0
    };
    let area = if idx < model.zone_area.len() {
        model.zone_area.as_slice()[idx]
    } else {
        0.0
    };
    let surfaces = model
        .surfaces
        .get(idx)
        .map(|v| v.iter().map(PySurface::from).collect())
        .unwrap_or_default();

    // Extract hvac_enabled from the underlying vector (if present).
    let hvac_enabled = if idx < model.hvac_enabled.len() {
        model.hvac_enabled.as_slice()[idx] != 0.0
    } else {
        true
    };

    PyZone {
        index: idx,
        temperature: temp,
        area,
        heating_setpoint: model.heating_setpoint,
        cooling_setpoint: model.cooling_setpoint,
        hvac_enabled,
        surfaces,
    }
}

/// Build a flat [`Vec<PySurface>`] of every surface in every zone of the model.
pub fn all_surfaces_from_model(model: &ThermalModel<VectorField>) -> Vec<PySurface> {
    model
        .surfaces
        .iter()
        .flat_map(|zone_surfaces| zone_surfaces.iter().map(PySurface::from))
        .collect()
}

/// Build the [`Vec<PyZone>`] snapshot list from a model.
pub fn all_zones_from_model(model: &ThermalModel<VectorField>) -> Vec<PyZone> {
    (0..model.num_zones)
        .map(|i| zone_from_model(model, i))
        .collect()
}

// =============================================================================
// Internal helpers (not exported to Python) — used by Model setters
// =============================================================================

/// Convert a Python-supplied `PySurface` back into the model's native
/// `WallSurface`. Used by `Model.set_surfaces()` to commit snapshot mutations.
pub fn surface_to_wall(s: &PySurface) -> WallSurface {
    let mut wall = WallSurface::new(s.area, s.u_value, s.orientation.into());
    wall.window_area = s.window_area;

    // The shading_devices list is the canonical source for shading on the
    // snapshot; prefer it over the optional single-field overhang/fin
    // shorthand that the convenience setters also update.
    if !s.shading_devices.is_empty() {
        for device in &s.shading_devices {
            match device.shading_type {
                PyShadingType::Overhang => {
                    wall.overhang = Some(Overhang {
                        depth: device.overhang_depth,
                        distance_above: device.mounting_height,
                        extension: 0.0,
                    });
                }
                PyShadingType::Fins => {
                    wall.fins.push(ShadeFin {
                        depth: device.fin_width,
                        distance_from_edge: 0.0,
                        side: crate::sim::shading::Side::Left,
                        height: 0.0,
                    });
                }
                PyShadingType::OverhangAndFins => {
                    wall.overhang = Some(Overhang {
                        depth: device.overhang_depth,
                        distance_above: device.mounting_height,
                        extension: 0.0,
                    });
                    wall.fins.push(ShadeFin {
                        depth: device.fin_width,
                        distance_from_edge: 0.0,
                        side: crate::sim::shading::Side::Left,
                        height: 0.0,
                    });
                }
                PyShadingType::None => {
                    // Explicit "no shading" — leave defaults.
                }
            }
        }
    } else {
        // No canonical shading_devices list — fall back to the shorthand fields
        // so older callers that only set overhang_depth/fin_width still work.
        if let (Some(depth), Some(height)) = (s.overhang_depth, s.overhang_height) {
            wall.overhang = Some(Overhang {
                depth,
                distance_above: height,
                extension: 0.0,
            });
        }
        if let Some(width) = s.fin_width {
            wall.fins.push(ShadeFin {
                depth: width,
                distance_from_edge: 0.0,
                side: crate::sim::shading::Side::Left,
                height: 0.0,
            });
        }
    }
    wall
}

/// Re-borrow helper for `Model.set_surfaces()`. Given a flat list of surfaces
/// (one per zone-or-mixed), pad/truncate to `num_zones` × `surfaces_per_zone`
/// so the model can absorb it.
pub fn reshape_surfaces_for_model(
    model: &ThermalModel<VectorField>,
    flat: Vec<PySurface>,
) -> Vec<Vec<WallSurface>> {
    let per_zone = model.surfaces.first().map(|z| z.len()).unwrap_or(4).max(1);
    let mut out: Vec<Vec<WallSurface>> = (0..model.num_zones)
        .map(|_| Vec::with_capacity(per_zone))
        .collect();

    for (i, s) in flat.into_iter().enumerate() {
        let zone_idx = i / per_zone;
        if zone_idx >= out.len() {
            break;
        }
        out[zone_idx].push(surface_to_wall(&s));
    }
    out
}

/// Build a [`PyHVACSystem`] snapshot from a model's heating / cooling capacity
/// and supply-air temperature.
pub fn hvac_system_from_model(model: &ThermalModel<VectorField>) -> PyHVACSystem {
    PyHVACSystem {
        heating_capacity: model.hvac_heating_capacity,
        cooling_capacity: model.hvac_cooling_capacity,
        cop_heating: 3.0,
        cop_cooling: 3.2,
        stages: 1,
        min_outdoor_temp: -10.0,
        max_outdoor_temp: 40.0,
        vav_enabled: false,
        economizer_enabled: false,
        supply_air_temp: 13.0,
    }
}

/// Apply a [`PyHVACSystem`] snapshot's heating/cooling capacity back to the model.
pub fn apply_hvac_system_to_model(model: &mut ThermalModel<VectorField>, hvac: &PyHVACSystem) {
    model.hvac_heating_capacity = hvac.heating_capacity;
    model.hvac_cooling_capacity = hvac.cooling_capacity;
}

#[cfg(all(test, feature = "python-bindings"))]
mod tests {
    //! Rust-side inline tests for the PyO3 wrappers in this module (Issue #2532).
    //!
    //! Coverage focuses on the pure-Rust conversion / helper layer:
    //! - `From` round-trips for the enums (Orientation, ShadingType,
    //!   ShadingDevice) and structs (PyMaterial <-> ConstructionLayer),
    //! - `PySurface::from(&WallSurface)` for the four shading branches
    //!   (none / overhang / fins / both),
    //! - `surface_to_wall` round-trip preserving shading,
    //! - `PyMaterial` derived properties (r_value, thermal capacitance),
    //! - `PySurface` / `PyZone` aggregate helpers,
    //! - `PyHVACSystem` derived properties (electrical input, can_operate_at),
    //! - the model↔snapshot helpers
    //!   (`hvac_system_from_model` / `apply_hvac_system_to_model`).

    use super::*;
    use crate::sim::shading::{Overhang, ShadeFin};

    // ========================================================================
    // Orientation round-trip
    // ========================================================================

    #[test]
    fn orientation_round_trip_preserves_all_variants() {
        let variants = [
            Orientation::North,
            Orientation::East,
            Orientation::South,
            Orientation::West,
            Orientation::Up,
            Orientation::Down,
            Orientation::Horizontal,
        ];
        for v in variants {
            let py: PyOrientation = v.into();
            let back: Orientation = py.into();
            assert_eq!(back, v, "round-trip lost variant");
        }
    }

    // ========================================================================
    // ShadingType / ShadingDevice round-trip
    // ========================================================================

    #[test]
    fn shading_type_round_trip_preserves_all_variants() {
        let variants = [
            ShadingType::None,
            ShadingType::Overhang,
            ShadingType::Fins,
            ShadingType::OverhangAndFins,
        ];
        for v in variants {
            let py: PyShadingType = v.into();
            let back: ShadingType = py.into();
            assert_eq!(back, v);
        }
    }

    #[test]
    fn shading_device_round_trip_preserves_fields() {
        let device = ShadingDevice::overhang_and_fins(0.7, 0.3, 2.1);
        let py: PyShadingDevice = device.into();
        let back: ShadingDevice = py.into();
        assert_eq!(back, device);
    }

    #[test]
    fn shading_device_none_factory_yields_none_type() {
        let py = PyShadingDevice::none();
        let back: ShadingDevice = py.into();
        assert_eq!(back.shading_type, ShadingType::None);
        assert_eq!(back.overhang_depth, 0.0);
        assert_eq!(back.fin_width, 0.0);
    }

    #[test]
    fn shading_device_overhang_factory_preserves_depth_and_height() {
        let py = PyShadingDevice::overhang(0.8, 1.9);
        let back: ShadingDevice = py.into();
        assert_eq!(back.shading_type, ShadingType::Overhang);
        assert_eq!(back.overhang_depth, 0.8);
        assert_eq!(back.mounting_height, 1.9);
        assert_eq!(back.fin_width, 0.0);
    }

    #[test]
    fn shading_device_fins_factory_preserves_width() {
        let py = PyShadingDevice::fins(0.45);
        let back: ShadingDevice = py.into();
        assert_eq!(back.shading_type, ShadingType::Fins);
        assert_eq!(back.fin_width, 0.45);
        assert_eq!(back.overhang_depth, 0.0);
    }

    #[test]
    fn shading_device_combined_factory_preserves_all_three() {
        let py = PyShadingDevice::overhang_and_fins(0.9, 0.25, 2.4);
        let back: ShadingDevice = py.into();
        assert_eq!(back.shading_type, ShadingType::OverhangAndFins);
        assert_eq!(back.overhang_depth, 0.9);
        assert_eq!(back.fin_width, 0.25);
        assert_eq!(back.mounting_height, 2.4);
    }

    // ========================================================================
    // Material (ConstructionLayer) round-trip
    // ========================================================================

    fn sample_layer() -> ConstructionLayer {
        // All ConstructionLayer::new args must be > 0 (it asserts positivity).
        ConstructionLayer::new("Concrete", 1.7, 2200.0, 900.0, 0.2)
    }

    #[test]
    fn material_round_trip_preserves_thermal_fields() {
        let layer = sample_layer();
        let py: PyMaterial = (&layer).into();
        let back: ConstructionLayer = py.into();
        assert_eq!(back.name, layer.name);
        assert_eq!(back.conductivity, layer.conductivity);
        assert_eq!(back.density, layer.density);
        assert_eq!(back.specific_heat, layer.specific_heat);
        assert_eq!(back.thickness, layer.thickness);
    }

    #[test]
    fn material_r_value_is_thickness_over_conductivity() {
        let layer = sample_layer(); // k=1.7, t=0.2  ->  R = 0.1176...
        let py: PyMaterial = (&layer).into();
        let expected = layer.thickness / layer.conductivity;
        assert!((py.r_value() - expected).abs() < 1e-12);
    }

    #[test]
    fn material_r_value_zero_conductivity_yields_zero() {
        // Defensive: zero conductivity should not divide-by-zero.
        let py = PyMaterial::new("Vacuum".to_string(), 0.0, 1.0, 1.0, 0.1);
        assert_eq!(py.r_value(), 0.0);
    }

    #[test]
    fn material_thermal_capacitance_per_area_matches_formula() {
        let layer = sample_layer(); // rho=2200, t=0.2, cp=900 -> 396_000 J/m²K
        let py: PyMaterial = (&layer).into();
        let expected = layer.density * layer.thickness * layer.specific_heat;
        assert!((py.thermal_capacitance_per_area() - expected).abs() < 1e-6);
    }

    // ========================================================================
    // Surface shading branches (From<&WallSurface>)
    // ========================================================================

    #[test]
    fn surface_from_wall_no_shading_yields_empty_devices() {
        let wall = WallSurface::new(10.0, 0.5, Orientation::South);
        let py = PySurface::from(&wall);
        assert!(py.shading_devices.is_empty());
        assert!(py.overhang_depth.is_none());
        assert!(py.overhang_height.is_none());
        assert!(py.fin_width.is_none());
    }

    #[test]
    fn surface_from_wall_overhang_only_populates_overhang_fields() {
        let wall = WallSurface::new(10.0, 0.5, Orientation::South).with_overhang(Overhang {
            depth: 0.6,
            distance_above: 1.5,
            extension: 0.0,
        });
        let py = PySurface::from(&wall);
        assert_eq!(py.shading_devices.len(), 1);
        assert_eq!(py.shading_devices[0].shading_type, PyShadingType::Overhang);
        assert_eq!(py.shading_devices[0].overhang_depth, 0.6);
        assert_eq!(py.shading_devices[0].mounting_height, 1.5);
        assert_eq!(py.overhang_depth, Some(0.6));
        assert_eq!(py.overhang_height, Some(1.5));
        assert!(py.fin_width.is_none());
    }

    #[test]
    fn surface_from_wall_fins_only_populates_fin_fields() {
        let mut wall = WallSurface::new(10.0, 0.5, Orientation::South);
        wall.fins.push(ShadeFin {
            depth: 0.4,
            distance_from_edge: 0.0,
            side: crate::sim::shading::Side::Left,
            height: 0.0,
        });
        let py = PySurface::from(&wall);
        assert_eq!(py.shading_devices.len(), 1);
        assert_eq!(py.shading_devices[0].shading_type, PyShadingType::Fins);
        assert_eq!(py.shading_devices[0].fin_width, 0.4);
        assert_eq!(py.fin_width, Some(0.4));
        assert!(py.overhang_depth.is_none());
    }

    #[test]
    fn surface_from_wall_overhang_and_fins_merges_into_one_device() {
        let mut wall = WallSurface::new(10.0, 0.5, Orientation::South).with_overhang(Overhang {
            depth: 0.7,
            distance_above: 2.0,
            extension: 0.0,
        });
        wall.fins.push(ShadeFin {
            depth: 0.3,
            distance_from_edge: 0.0,
            side: crate::sim::shading::Side::Left,
            height: 0.0,
        });
        let py = PySurface::from(&wall);
        // The (true, true) branch produces exactly one OverhangAndFins device.
        assert_eq!(py.shading_devices.len(), 1);
        assert_eq!(
            py.shading_devices[0].shading_type,
            PyShadingType::OverhangAndFins
        );
        assert_eq!(py.shading_devices[0].overhang_depth, 0.7);
        assert_eq!(py.shading_devices[0].mounting_height, 2.0);
        assert_eq!(py.shading_devices[0].fin_width, 0.3);
    }

    // ========================================================================
    // surface_to_wall round-trip
    // ========================================================================

    #[test]
    fn surface_to_wall_preserves_basic_fields() {
        let mut py = PySurface::new(12.0, 0.45, PyOrientation::East, 2.0);
        py.append_shading(PyShadingDevice::overhang(0.5, 1.8));
        let wall = surface_to_wall(&py);
        assert_eq!(wall.area, 12.0);
        assert_eq!(wall.u_value, 0.45);
        assert_eq!(wall.orientation, Orientation::East);
        assert_eq!(wall.window_area, 2.0);
        assert!(wall.overhang.is_some());
        let oh = wall.overhang.unwrap();
        assert_eq!(oh.depth, 0.5);
        assert_eq!(oh.distance_above, 1.8);
    }

    #[test]
    fn surface_to_wall_overhang_and_fins_round_trip() {
        let mut py = PySurface::new(8.0, 0.3, PyOrientation::West, 0.0);
        py.append_shading(PyShadingDevice::overhang_and_fins(0.6, 0.25, 1.7));
        let wall = surface_to_wall(&py);
        assert!(wall.overhang.is_some(), "overhang should be set");
        assert_eq!(wall.fins.len(), 1, "one fin should be set");
        assert_eq!(wall.overhang.unwrap().depth, 0.6);
        assert_eq!(wall.fins[0].depth, 0.25);
    }

    #[test]
    fn surface_to_wall_falls_back_to_shorthand_fields_when_no_devices() {
        // When shading_devices is empty, the helper should still respect the
        // legacy optional shorthand fields (overhang_depth / overhang_height /
        // fin_width). This is the backward-compat branch.
        let mut py = PySurface::new(5.0, 0.4, PyOrientation::North, 0.0);
        py.overhang_depth = Some(0.55);
        py.overhang_height = Some(1.6);
        py.fin_width = Some(0.22);
        // NB: intentionally NOT calling append_shading — shading_devices stays empty.
        let wall = surface_to_wall(&py);
        assert!(wall.overhang.is_some());
        assert_eq!(wall.overhang.unwrap().depth, 0.55);
        assert_eq!(wall.overhang.unwrap().distance_above, 1.6);
        assert_eq!(wall.fins.len(), 1);
        assert_eq!(wall.fins[0].depth, 0.22);
    }

    // ========================================================================
    // PySurface / PyZone aggregation helpers
    // ========================================================================

    #[test]
    fn surface_total_area_is_area_field() {
        let s = PySurface::new(15.5, 0.4, PyOrientation::South, 3.0);
        assert_eq!(s.total_area(), 15.5);
    }

    #[test]
    fn surface_append_shading_extends_list() {
        let mut s = PySurface::new(10.0, 0.4, PyOrientation::South, 0.0);
        assert_eq!(s.shading_devices.len(), 0);
        s.append_shading(PyShadingDevice::none());
        assert_eq!(s.shading_devices.len(), 1);
    }

    #[test]
    fn surface_add_overhang_and_fins_update_both_state_and_list() {
        let mut s = PySurface::new(10.0, 0.4, PyOrientation::South, 0.0);

        // add_overhang updates shorthand fields and appends a device.
        s.add_overhang(0.7, 1.9);
        assert_eq!(s.overhang_depth, Some(0.7));
        assert_eq!(s.overhang_height, Some(1.9));
        assert_eq!(s.shading_devices.len(), 1);

        // add_fins updates the shorthand field and appends another device.
        s.add_fins(0.3);
        assert_eq!(s.fin_width, Some(0.3));
        assert_eq!(s.shading_devices.len(), 2);
    }

    fn sample_zone_with_surfaces() -> PyZone {
        // Two south-facing surfaces + one east-facing surface.
        let s1 = PySurface::new(10.0, 0.4, PyOrientation::South, 0.0);
        let s2 = PySurface::new(8.0, 0.5, PyOrientation::South, 0.0);
        let s3 = PySurface::new(6.0, 0.5, PyOrientation::East, 0.0);
        PyZone::new(0, 22.0, 50.0, 20.0, 24.0, true, vec![s1, s2, s3])
    }

    #[test]
    fn zone_surface_count_matches_constructor() {
        let z = sample_zone_with_surfaces();
        assert_eq!(z.surface_count(), 3);
    }

    #[test]
    fn zone_total_surface_area_sums_all_surfaces() {
        let z = sample_zone_with_surfaces();
        // 10 + 8 + 6 = 24 m²
        assert!((z.total_surface_area() - 24.0).abs() < 1e-9);
    }

    #[test]
    fn zone_surfaces_with_orientation_filters_correctly() {
        let z = sample_zone_with_surfaces();
        let south = z.surfaces_with_orientation(PyOrientation::South);
        assert_eq!(south.len(), 2);
        for s in &south {
            assert_eq!(s.orientation, PyOrientation::South);
        }
        let east = z.surfaces_with_orientation(PyOrientation::East);
        assert_eq!(east.len(), 1);
    }

    #[test]
    fn zone_surfaces_with_orientation_returns_empty_for_no_match() {
        let z = sample_zone_with_surfaces();
        let up = z.surfaces_with_orientation(PyOrientation::Up);
        assert!(up.is_empty());
    }

    // ========================================================================
    // PyHVACSystem derived properties
    // ========================================================================

    #[test]
    fn hvac_heating_electrical_input_divides_capacity_by_cop() {
        let h = PyHVACSystem::new(
            10_000.0, 8_000.0, 4.0, 4.0, 1, -10.0, 40.0, false, false, 13.0,
        );
        assert!((h.heating_electrical_input() - 2_500.0).abs() < 1e-9);
    }

    #[test]
    fn hvac_cooling_electrical_input_divides_capacity_by_cop() {
        let h = PyHVACSystem::new(
            10_000.0, 8_000.0, 4.0, 4.0, 1, -10.0, 40.0, false, false, 13.0,
        );
        assert!((h.cooling_electrical_input() - 2_000.0).abs() < 1e-9);
    }

    #[test]
    fn hvac_electrical_input_zero_when_cop_is_zero() {
        let h = PyHVACSystem::new(
            10_000.0, 8_000.0, 0.0, 0.0, 1, -10.0, 40.0, false, false, 13.0,
        );
        assert_eq!(h.heating_electrical_input(), 0.0);
        assert_eq!(h.cooling_electrical_input(), 0.0);
    }

    #[test]
    fn hvac_can_operate_at_respects_min_max_bounds() {
        let h = PyHVACSystem::new(
            10_000.0, 8_000.0, 3.0, 3.2, 1, -10.0, 40.0, false, false, 13.0,
        );
        assert!(!h.can_operate_at(-15.0), "below min");
        assert!(h.can_operate_at(-10.0), "exactly min (inclusive)");
        assert!(h.can_operate_at(20.0), "mid-range");
        assert!(h.can_operate_at(40.0), "exactly max (inclusive)");
        assert!(!h.can_operate_at(45.0), "above max");
    }

    #[test]
    fn hvac_from_model_round_trip_through_apply() {
        // hvac_system_from_model snapshots capacities; apply_hvac_system_to_model
        // writes them back. Round-tripping should preserve the new values.
        let mut model = ThermalModel::<VectorField>::new(1);
        model.hvac_heating_capacity = 12_345.0;
        model.hvac_cooling_capacity = 6_789.0;
        let snap = hvac_system_from_model(&model);
        assert_eq!(snap.heating_capacity, 12_345.0);
        assert_eq!(snap.cooling_capacity, 6_789.0);

        // Apply modified values back.
        let mut updated = snap.clone();
        updated.heating_capacity = 99_999.0;
        updated.cooling_capacity = 88_888.0;
        apply_hvac_system_to_model(&mut model, &updated);
        assert_eq!(model.hvac_heating_capacity, 99_999.0);
        assert_eq!(model.hvac_cooling_capacity, 88_888.0);
    }

    #[test]
    fn hvac_from_model_uses_default_cop_and_stages() {
        // hvac_system_from_model hard-codes the default COPs / stages / temp
        // limits when constructing a snapshot — those should match the
        // PyHVACSystem::new defaults so Python sees consistent values.
        let model = ThermalModel::<VectorField>::new(1);
        let snap = hvac_system_from_model(&model);
        assert_eq!(snap.cop_heating, 3.0);
        assert_eq!(snap.cop_cooling, 3.2);
        assert_eq!(snap.stages, 1);
        assert_eq!(snap.min_outdoor_temp, -10.0);
        assert_eq!(snap.max_outdoor_temp, 40.0);
        assert!(!snap.vav_enabled);
        assert!(!snap.economizer_enabled);
        assert_eq!(snap.supply_air_temp, 13.0);
    }
}

// =============================================================================
// Single-building Model — top-level Python entrypoint (Issue #2493).
// Moved verbatim from `lib.rs`; the crate-root #[pymodule] registers this via
// `m.add_class::<python::model_bindings::Model>()`. Python name is unchanged
// (`#[pyclass]` with no `name` attribute => "Model").
// =============================================================================

use crate::ai::surrogate::SurrogateManager;
use crate::api::error::SurrogateError;
use crate::batch_oracle::BatchOracle;
use crate::python::batch_oracle_bindings::ParameterBounds;
use crate::weather::HourlyWeatherData;

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use ndarray::Array2;
use numpy::PyArrayMethods;

#[allow(unused_imports)]
use log::{debug, info};

/// Monotonic id generator for Python-facing `Model` instances (Issue #2548).
/// Surfaces as `simulation_id` on `Model.__repr__` so Python users can correlate
/// a divergence back to a specific simulation.
static MODEL_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Standard Single-Building Model for detailed building energy analysis.
///
/// Use this class when you need detailed simulation of a single building configuration,
/// including hourly temperature traces and ASHRAE 140 validation.
#[cfg(feature = "python-bindings")]
/// Single-building energy model for detailed simulation.
///
/// Use for validation, hourly temperature traces, or ASHRAE 140 testing.
/// Provides detailed diagnostics including hourly temperature traces, peak loads,
/// energy consumption breakdown, and comparison reports.
///
/// # Python API
/// ```python,ignore
/// from fluxion import Model
///
/// # Create from ASHRAE 140 case
/// model = Model.from_case("600")
///
/// # Run simulation
/// eui = model.simulate(years=1, use_surrogates=False)
///
/// # Get detailed diagnostics
/// temps = model.get_hourly_temperatures()
/// peak_heating = model.get_peak_heating()
/// report = model.generate_comparison_report()
/// ```
///
/// # Diagnostics
/// - Hourly temperature traces (zone, mass, surface)
/// - Peak load tracking (heating/cooling timing and magnitude)
/// - Energy consumption breakdown (heating, cooling, fans)
/// - Comparison reports against reference data (ASHRAE 140)
///
/// # Performance
/// - Single configuration: <100ms for 8760 timesteps
/// - Detailed diagnostics: Additional overhead for data collection
///
/// See docs/API_REFERENCE.md for complete API reference.
#[pyclass]
pub struct Model {
    inner: ThermalModel<VectorField>,
    surrogates: SurrogateManager,
    /// Stable id for this Python `Model` instance, surfaced via `__repr__`
    /// (Issue #2548) so users can correlate a tracing span / metric to one
    /// specific simulation when debugging a divergence.
    #[pyo3(get)]
    simulation_id: String,
    /// Wall-clock duration of the most recent `simulate()` call, recorded for
    /// the debug-friendly `__repr__` (Issue #2548). `None` until the first
    /// successful `simulate()` invocation.
    last_duration: Option<Duration>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl Model {
    /// Create a new Model instance with default configuration.
    ///
    /// # Arguments
    /// * `num_zones` - Number of thermal zones (default: 1)
    #[new]
    #[pyo3(signature = (num_zones=1))]
    fn new(num_zones: usize) -> PyResult<Self> {
        Ok(Model {
            inner: ThermalModel::<VectorField>::new(num_zones),
            surrogates: SurrogateManager::new().map_err(|e| {
                SurrogateError::new_err(format!("Failed to create SurrogateManager: {}", e))
            })?,
            simulation_id: format!("model-{}", MODEL_ID_COUNTER.fetch_add(1, Ordering::Relaxed)),
            last_duration: None,
        })
    }

    /// Get number of zones in the model.
    fn num_zones(&self) -> usize {
        self.inner.num_zones
    }

    /// Get current zone temperatures.
    fn get_temperatures(&self) -> Vec<f64> {
        self.inner.get_temperatures()
    }

    /// Set zone temperatures.
    fn set_temperatures(&mut self, temps: Vec<f64>) -> PyResult<()> {
        if temps.len() != self.inner.num_zones {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Temperature vector length ({}) must match number of zones ({})",
                temps.len(),
                self.inner.num_zones
            )));
        }
        self.inner.temperatures = VectorField::new(temps);
        Ok(())
    }

    /// Get building type for auto-loading internal load profiles (Plan 17-04).
    ///
    /// Returns the building type enum (Office, Retail, School, etc.) which is used
    /// to auto-load default internal load profiles when simulate_with_loads() is called.
    fn building_type(&self) -> String {
        // Convert BuildingType enum to string
        match self.inner.building_type {
            crate::sim::occupancy::BuildingType::Office => "Office".to_string(),
            crate::sim::occupancy::BuildingType::Retail => "Retail".to_string(),
            crate::sim::occupancy::BuildingType::School => "School".to_string(),
            crate::sim::occupancy::BuildingType::Hospital => "Hospital".to_string(),
            crate::sim::occupancy::BuildingType::Hotel => "Hotel".to_string(),
            crate::sim::occupancy::BuildingType::Restaurant => "Restaurant".to_string(),
            crate::sim::occupancy::BuildingType::Warehouse => "Warehouse".to_string(),
        }
    }

    /// Set building type for auto-loading internal load profiles (Plan 17-04).
    ///
    /// # Arguments
    /// * `building_type` - Building type string (Office, Retail, School, Hospital, Hotel, Restaurant, Warehouse)
    ///
    /// This building type is used to auto-load default internal load profiles (lighting, equipment, occupancy)
    /// when simulate_with_loads() is called without specifying custom loads.
    fn set_building_type(&mut self, building_type: String) -> PyResult<()> {
        self.inner.building_type = match building_type.as_str() {
            "Office" => crate::sim::occupancy::BuildingType::Office,
            "Retail" => crate::sim::occupancy::BuildingType::Retail,
            "School" => crate::sim::occupancy::BuildingType::School,
            "Hospital" => crate::sim::occupancy::BuildingType::Hospital,
            "Hotel" => crate::sim::occupancy::BuildingType::Hotel,
            "Restaurant" => crate::sim::occupancy::BuildingType::Restaurant,
            "Warehouse" => crate::sim::occupancy::BuildingType::Warehouse,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid building type '{}'. Must be one of: Office, Retail, School, Hospital, Hotel, Restaurant, Warehouse",
                    building_type
                )));
            }
        };
        Ok(())
    }

    /// Simulate building energy consumption over specified years.
    ///
    /// # Arguments
    /// * `years` - Number of years to simulate (1-5 typical)
    /// * `use_surrogates` - If true, use AI surrogates for load predictions; if false, use analytical calculations
    ///
    /// # Returns
    /// Total energy use intensity (EUI) in kWh/m²/year
    fn simulate(&mut self, years: u32, use_surrogates: bool) -> PyResult<f64> {
        // Issue #2548: enter a tracing span for the full duration of the
        // Python-visible call so spans/metrics emitted from the physics core
        // are correlated to this `simulation_id`.
        let _span = tracing::info_span!(
            "python_simulate",
            simulation_id = %self.simulation_id,
            years,
            use_surrogates,
        )
        .entered();

        let start = Instant::now();
        // The physics core currently infallibly returns EUI, but compute the
        // result as a `PyResult<f64>` block so the success/error outcome
        // label is derived from a real `Result` — the error branch will fire
        // as soon as a future change makes `simulate` fallible.
        let outcome: PyResult<f64> = {
            info!(
                "Starting simulation for {} years, use_surrogates={}",
                years, use_surrogates
            );
            let steps = years as usize * 8760;
            debug!("Simulation will process {} timesteps", steps);
            let result = self.inner.solve_timesteps(
                steps,
                &self.surrogates,
                use_surrogates,
                None,
                None,
                None,
            );
            info!("Simulation complete, EUI = {:.2} kWh/m²/year", result);
            Ok(result)
        };

        let duration = start.elapsed();
        // Always record the last call's duration, even on error, so the
        // `__repr__` reflects the most recent attempt when debugging.
        self.last_duration = Some(duration);

        metrics::histogram!("fluxion_python_simulate_duration_seconds")
            .record(duration.as_secs_f64());
        let outcome_label = if outcome.is_ok() { "success" } else { "error" };
        metrics::counter!("fluxion_python_simulate_total", "outcome" => outcome_label).increment(1);

        outcome
    }

    /// Simulate building energy consumption with internal loads (Plan 17-04).
    ///
    /// This method allows specifying internal loads (lighting, equipment, occupancy)
    /// for more detailed building energy modeling. If all load parameters are None,
    /// the building type profile will be auto-loaded based on model.building_type.
    ///
    /// # Arguments
    /// * `years` - Number of years to simulate (1-5 typical)
    /// * `use_surrogates` - If true, use AI surrogates for load predictions; if false, use analytical calculations
    ///
    /// # Returns
    /// Total energy use intensity (EUI) in kWh/m²/year
    ///
    /// # Note
    /// This method currently accepts None for all load parameters, which will trigger
    /// auto-loading of the building profile based on model.building_type.
    /// Full Python API for passing custom load objects will be added in a future phase.
    ///
    /// # Example
    /// ```python
    /// import fluxion
    ///
    /// model = fluxion.Model()
    /// model.building_type = fluxion.BuildingType.Office
    ///
    /// # Simulate with auto-loaded Office building profile
    /// eui = model.simulate_with_loads(1, False)
    /// ```
    fn simulate_with_loads(&mut self, years: u32, use_surrogates: bool) -> PyResult<f64> {
        info!(
            "Starting simulation with auto-loaded internal loads for {} years, use_surrogates={}",
            years, use_surrogates
        );
        let steps = years as usize * 8760;

        // Pass None for all loads to trigger auto-loading from building_type
        let result =
            self.inner
                .solve_timesteps(steps, &self.surrogates, use_surrogates, None, None, None);
        info!("Simulation complete, EUI = {:.2} kWh/m²/year", result);
        Ok(result)
    }

    /// Simulate building energy consumption with NumPy array inputs for weather data.
    ///
    /// This method enables direct NumPy memory sharing between Python and Rust,
    /// eliminating copy overhead for large simulations. Weather data is passed
    /// as NumPy arrays, and zone temperatures are returned as a 2D NumPy array.
    ///
    /// # Arguments
    /// * `dry_bulb_temp` - Outdoor dry bulb temperature (°C), shape (steps,)
    /// * `dni` - Direct Normal Irradiance (W/m²), shape (steps,)
    /// * `dhi` - Diffuse Horizontal Irradiance (W/m²), shape (steps,)
    /// * `ghi` - Global Horizontal Irradiance (W/m²), shape (steps,)
    /// * `wind_speed` - Wind speed (m/s), shape (steps,)
    /// * `humidity` - Relative humidity (%), shape (steps,)
    /// * `horizontal_infrared` - Horizontal infrared radiation (W/m²), shape (steps,)
    /// * `use_surrogates` - If true, use AI surrogates for load predictions
    ///
    /// # Returns
    /// 2D NumPy array of zone temperatures (steps x num_zones) in °C
    ///
    /// # Example
    /// ```python
    /// import fluxion
    /// import numpy as np
    ///
    /// model = fluxion.Model(num_zones=3)
    ///
    /// # Create weather data arrays (8760 hourly values)
    /// n_timesteps = 8760
    /// dry_bulb = np.random.uniform(10, 35, n_timesteps)
    /// dni = np.random.uniform(0, 1000, n_timesteps)
    /// dhi = np.random.uniform(0, 500, n_timesteps)
    /// ghi = np.random.uniform(0, 1000, n_timesteps)
    /// wind_speed = np.random.uniform(0, 10, n_timesteps)
    /// humidity = np.random.uniform(30, 80, n_timesteps)
    /// horizontal_ir = np.random.uniform(200, 500, n_timesteps)
    ///
    /// # Run simulation and get zone temperatures
    /// zone_temps = model.simulate_numpy(
    ///     dry_bulb, dni, dhi, ghi, wind_speed, humidity, horizontal_ir, False
    /// )
    /// # zone_temps.shape == (8760, 3)
    /// ```
    #[allow(clippy::too_many_arguments)]
    fn simulate_numpy<'py>(
        &mut self,
        py: Python<'py>,
        dry_bulb_temp: &Bound<'py, pyo3::types::PyAny>,
        dni: &Bound<'py, pyo3::types::PyAny>,
        dhi: &Bound<'py, pyo3::types::PyAny>,
        ghi: &Bound<'py, pyo3::types::PyAny>,
        wind_speed: &Bound<'py, pyo3::types::PyAny>,
        humidity: &Bound<'py, pyo3::types::PyAny>,
        horizontal_infrared: &Bound<'py, pyo3::types::PyAny>,
        use_surrogates: bool,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        // Helper to extract 1D numpy array as Vec<f64>
        fn extract_1d_f64(arr: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Vec<f64>> {
            if let Ok(pyarr) = arr.cast::<numpy::PyArray1<f64>>() {
                let slice = unsafe { pyarr.as_slice()? };
                return Ok(slice.to_vec());
            }
            Err(pyo3::exceptions::PyValueError::new_err(
                "Expected 1D numpy array",
            ))
        }

        // Extract weather data arrays
        let dry_bulb_vec = extract_1d_f64(dry_bulb_temp)?;
        let dni_vec = extract_1d_f64(dni)?;
        let dhi_vec = extract_1d_f64(dhi)?;
        let ghi_vec = extract_1d_f64(ghi)?;
        let wind_vec = extract_1d_f64(wind_speed)?;
        let humidity_vec = extract_1d_f64(humidity)?;
        let hir_vec = extract_1d_f64(horizontal_infrared)?;

        let steps = dry_bulb_vec.len();

        // Validate all arrays have the same length
        if dni_vec.len() != steps
            || dhi_vec.len() != steps
            || ghi_vec.len() != steps
            || wind_vec.len() != steps
            || humidity_vec.len() != steps
            || hir_vec.len() != steps
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All weather arrays must have the same length",
            ));
        }

        let num_zones = self.inner.num_zones;
        info!(
            "Starting NumPy simulation for {} timesteps, {} zones, use_surrogates={}",
            steps, num_zones, use_surrogates
        );

        // Initialize temperature storage: (steps x num_zones)
        let mut zone_temps = Array2::<f64>::zeros((steps, num_zones));

        // Build weather data and run simulation
        for t in 0..steps {
            if t % 1000 == 0 {
                info!("Progress: {}/{} timesteps", t, steps);
            }

            let weather = HourlyWeatherData {
                dry_bulb_temp: dry_bulb_vec[t],
                dni: dni_vec[t],
                dhi: dhi_vec[t],
                ghi: ghi_vec[t],
                wind_speed: wind_vec[t],
                humidity: humidity_vec[t],
                horizontal_infrared: hir_vec[t],
                hour_of_year: t,
                ground_temperature: None,
                horizontal_illuminance: None,
                diffuse_illuminance: None,
                snow_depth: None,
                snow_cover: None,
                present_weather: None,
                present_weather_code: None,
            };

            self.inner.set_weather(weather);
            let _energy = self.inner.step_physics(t, dry_bulb_vec[t], 3600.0);

            // Collect zone temperatures
            let temps = self.inner.get_temperatures();
            for (zone_idx, &temp) in temps.iter().enumerate() {
                zone_temps[[t, zone_idx]] = temp;
            }
        }

        info!("NumPy simulation complete");
        Ok(numpy::PyArray2::from_owned_array(py, zone_temps))
    }

    /// Simulate one timestep.
    ///
    /// # Arguments
    /// * `timestep` - Current timestep index (0-8759 for hourly annual simulation)
    /// * `outdoor_temp` - Outdoor air temperature (°C)
    /// * `use_surrogates` - If true, use neural surrogates; if false, use analytical calculations
    /// Register an ONNX surrogate model for this `Model` instance.
    ///
    /// The path is validated per Issue #2529 (existence, `.onnx` extension,
    /// allow-list directory via `FLUXION_MODEL_DIR`, and 256 MiB size limit)
    /// before reaching the ONNX runtime. Error messages are generic and never
    /// echo the raw user-supplied path.
    fn load_surrogate(&mut self, model_path: String) -> PyResult<()> {
        let validated = crate::ai::surrogate::validate_model_path(&model_path)
            .map_err(SurrogateError::new_err)?;
        match SurrogateManager::load_onnx(&validated.to_string_lossy()) {
            Ok(manager) => {
                self.surrogates = manager;
                Ok(())
            }
            Err(e) => Err(SurrogateError::new_err(format!(
                "Failed to load ONNX surrogate model: {e}"
            ))),
        }
    }

    /// Get the parameter bounds for building design variables.
    ///
    /// Returns a ParameterBounds struct with the valid ranges for all design
    /// parameters used by BatchOracle. This is useful for optimization libraries
    /// that need to generate valid parameter vectors.
    ///
    /// # Returns
    /// ParameterBounds struct containing min/max values for:
    /// - Window U-value (W/m²K)
    /// - Heating setpoint (°C)
    /// - Cooling setpoint (°C)
    ///
    /// # Example
    /// ```python
    /// import fluxion
    ///
    /// oracle = fluxion.BatchOracle()
    /// bounds = oracle.get_parameter_bounds()
    ///
    /// print(f"U-value range: [{bounds.min_u_value}, {bounds.max_u_value}]")
    /// print(f"Heating setpoint range: [{bounds.min_heating_setpoint}, {bounds.max_heating_setpoint}]")
    /// print(f"Cooling setpoint range: [{bounds.min_cooling_setpoint}, {bounds.max_cooling_setpoint}]")
    /// ```
    fn get_parameter_bounds(&self) -> ParameterBounds {
        ParameterBounds::get_bounds()
    }

    /// Validate a parameter vector against physical constraints.
    ///
    /// This method checks that all parameter values are within valid ranges and
    /// that heating/cooling setpoints are consistent. If validation fails, a
    /// ValidationError is raised with a clear, actionable message.
    ///
    /// # Arguments
    /// * `params` - Parameter vector to validate. Elements:
    ///   - `[0]`: Window U-value (W/m²K, must be finite and in [0.1, 5.0])
    ///   - `[1]`: Heating setpoint (°C, must be finite and in [15.0, 25.0])
    ///   - `[2]`: Cooling setpoint (°C, must be finite and in [22.0, 32.0])
    ///
    /// # Raises
    /// ValidationError with detailed message including:
    /// - Parameter index
    /// - Invalid value
    /// - Valid range
    /// - Type of error (NaN, infinite, or out of range)
    ///
    /// # Example
    /// ```python
    /// import fluxion
    ///
    /// oracle = fluxion.BatchOracle()
    ///
    /// # Valid parameters
    /// oracle.validate_parameters([1.5, 20.0, 27.0])  # OK
    ///
    /// # Invalid U-value (raises ValidationError)
    /// try:
    ///     oracle.validate_parameters([-1.0, 20.0, 27.0])
    /// except fluxion.ValidationError as e:
    ///     print(f"Validation failed: {e}")
    ///     # Output: Window U-value (index 0, -1.00 W/m²K) out of range [0.1, 5.0] W/m²K
    ///
    /// # NaN value (raises ValidationError)
    /// try:
    ///     oracle.validate_parameters([float('nan'), 20.0, 27.0])
    /// except fluxion.ValidationError as e:
    ///     print(f"Validation failed: {e}")
    ///     # Output: Window U-value (index 0) is NaN (value: nan W/m²K). Cannot use in simulation.
    /// ```
    fn validate_parameters_py(&self, params: Vec<f64>) -> PyResult<()> {
        BatchOracle::validate_parameters(&params)?;
        Ok(())
    }

    /// Set ground temperature model to constant value.
    ///
    /// # Arguments
    /// * `temperature` - Constant ground temperature (°C)
    fn set_ground_temp(&mut self, temperature: f64) {
        self.inner.set_ground_temp(temperature);
    }

    /// Get ground temperature at a specific timestep.
    ///
    /// # Arguments
    /// * `timestep` - Timestep index (0-8759 for hourly annual simulation)
    ///
    /// # Returns
    /// Ground temperature (°C)
    fn ground_temperature_at(&self, timestep: usize) -> f64 {
        self.inner.ground_temperature_at(timestep)
    }

    /// Return a Python list of [`crate::python::model_bindings::PyZone`] snapshots,
    /// one per zone in the model.
    ///
    /// Each returned `Zone` is an **owned snapshot** of the current zone state
    /// (temperature, area, surfaces, HVAC setpoints). The snapshot does **not**
    /// borrow from this model — Python garbage collection of any returned
    /// `Zone` cannot invalidate this model, and conversely this model may be
    /// mutated or re-simulated while Python still holds references to
    /// previously returned zones. See `docs/bindings.md` for the full
    /// lifetime story.
    ///
    /// Iteration works out of the box via the standard Python list iterator
    /// protocol:
    /// ```python,ignore
    /// model = fluxion.Model(num_zones=3)
    /// for z in model.zones():
    ///     print(z.index, z.temperature, z.area)
    /// ```
    fn zones(&self) -> Vec<crate::python::model_bindings::PyZone> {
        crate::python::model_bindings::all_zones_from_model(&self.inner)
    }

    /// Return a flat Python list of [`crate::python::model_bindings::PySurface`]
    /// snapshots, one for every surface in every zone.
    ///
    /// Like [`Self::zones`], each surface is an owned snapshot. Mutating a
    /// snapshot via `surface.append_shading(...)` only mutates the Python
    /// object — to push the change back into the model, use
    /// [`Self::set_surfaces`].
    ///
    /// # Example: find all south-facing surfaces
    /// ```python,ignore
    /// model = fluxion.Model(num_zones=2)
    /// south = [s for s in model.surfaces() if s.orientation == fluxion.Orientation.South]
    /// for s in south:
    ///     s.add_overhang(depth=1.0, height=2.5)
    /// model.set_surfaces(south + [s for s in model.surfaces() if s.orientation != fluxion.Orientation.South])
    /// ```
    fn surfaces(&self) -> Vec<crate::python::model_bindings::PySurface> {
        crate::python::model_bindings::all_surfaces_from_model(&self.inner)
    }

    /// Push a flat list of [`crate::python::model_bindings::PySurface`]
    /// snapshots back into the model. Surfaces are reshaped per-zone (4 per
    /// zone by default; this matches the ASHRAE 140 case-default wall
    /// configuration).
    ///
    /// The number of zones in the model does not change — only the surface
    /// data inside each zone is replaced. This is the round-trip companion
    /// to [`Self::surfaces`].
    ///
    /// # Arguments
    /// * `surfaces` - flat list of [`crate::python::model_bindings::PySurface`]
    ///   values; the list length must be a multiple of `surfaces_per_zone`,
    ///   otherwise the trailing surfaces are truncated.
    fn set_surfaces(&mut self, surfaces: Vec<crate::python::model_bindings::PySurface>) {
        self.inner.surfaces =
            crate::python::model_bindings::reshape_surfaces_for_model(&self.inner, surfaces);
    }

    /// Return an [`crate::python::model_bindings::PyHVACSystem`] snapshot of
    /// the model's current heating and cooling plant configuration.
    ///
    /// The snapshot is an owned value (no borrow back into the model). To
    /// push changes back, use [`Self::set_hvac_system`].
    fn hvac_system(&self) -> crate::python::model_bindings::PyHVACSystem {
        crate::python::model_bindings::hvac_system_from_model(&self.inner)
    }

    /// Apply a [`crate::python::model_bindings::PyHVACSystem`] snapshot's
    /// heating/cooling capacity to the model. Used together with
    /// [`Self::hvac_system`] for snapshot-then-commit mutation patterns.
    ///
    /// Only heating/cooling capacity is propagated back; other HVACSystem
    /// fields (COP, stages, etc.) are advisory and not stored on
    /// `ThermalModelData`.
    fn set_hvac_system(&mut self, hvac: crate::python::model_bindings::PyHVACSystem) {
        crate::python::model_bindings::apply_hvac_system_to_model(&mut self.inner, &hvac);
    }

    /// Debug-friendly `__repr__` (Issue #2548).
    ///
    /// Exposes the stable `simulation_id` (so users can grep tracing/metrics
    /// output for one specific run) and `last_duration_ms` from the most
    /// recent `simulate()` call. `last_duration_ms` is `0.0` before the first
    /// simulation has run.
    fn __repr__(&self) -> String {
        let last_duration_ms = self
            .last_duration
            .map(|d| d.as_secs_f64() * 1_000.0)
            .unwrap_or(0.0);
        format!(
            "Model(simulation_id='{}', last_duration_ms={:.2}, num_zones={}, use_surrogates_ready={})",
            self.simulation_id,
            last_duration_ms,
            self.inner.num_zones,
            self.surrogates.gpu_supported(),
        )
    }
}
