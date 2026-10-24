// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Unified Simulation Schema for Fluxion Building Energy Modeling.
//!
//! This module defines the canonical versioned schema for building energy simulations,
//! unifying geometry, constructions, schedules, weather, controls, and outputs
//! into a single contract for both CLI and Python pathways.
//!
//! # Schema Version
//!
//! The current schema version is `1.0`. All schema types are versioned to ensure
//! backward compatibility and clear migration paths when the schema evolves.
//!
//! # Schema Revision 1.1 — per-surface geometry, fenestration and HVAC
//! selection (Issue #3989)
//!
//! Revision 1.1 adds three capabilities that the zone-aggregate V1 shape
//! could not express:
//!
//! 1. **Per-surface geometry** — [`SurfaceSpec`] (and [`SurfaceVertex`])
//!    describe individual walls / roofs / floors with their own area,
//!    orientation, tilt and optional vertex loop, instead of only the
//!    lumped `Geometry::zones` aggregates.
//! 2. **Per-surface / per-orientation fenestration** — [`WindowSpec`]
//!    gained an optional window-to-wall ratio plus an optional surface id
//!    and orientation, so "north WWR 20 %, south 60 %" is expressible.
//! 3. **Named HVAC system selection** — [`ControlSet::hvac_system`]
//!    selects one of the engine's own [`HVACSystemType`] templates
//!    (`Simple`, `VAV`, `CAV`, `HeatPump`, `DOASFanCoil`, `Ideal`).
//!
//!    The named templates (issue #3997) are, with their modelling
//!    assumptions and limits:
//!
//!    | Template | Air side | Zone sensible load | Zone latent load | Ventilation |
//!    |---|---|---|---|---|
//!    | `Simple` | none (on/off, fixed capacity) | met by capacity | not modelled | not modelled |
//!    | `VAV` | packaged VAV terminal | met by airflow + reheat | not modelled | not modelled |
//!    | `CAV` | packaged CAV terminal | met by fixed airflow | not modelled | not modelled |
//!    | `HeatPump` | none (unitary / DX) | met by capacity | not modelled | not modelled |
//!    | `DOASFanCoil` | DOAS + four-pipe fan coil | met by the fan-coil water side | removed at the DOAS | met by the DOAS (100 % outdoor air) |
//!    | `Ideal` | ideal loads | met exactly | not modelled | not modelled |
//!
//!    The authoritative per-template documentation lives on the engine enum
//!    ([`crate::sim::hvac::HVACSystemType`]); the [`ControlSet::hvac_system`]
//!    field docs repeat the same table for modelers reading the schema.
//!
//! ## Versioning decision (why this is *inside* V1)
//!
//! The extension shipped **additively inside V1** (schema revision 1.1)
//! rather than as a parallel `SimulationSchemaV2` payload:
//!
//! * Every new field is `#[serde(default)]` and is *omitted again* when
//!   absent (`skip_serializing_if`), so a pre-#3989 V1 document still
//!   deserialises — and re-serialises — byte-for-byte as before. No
//!   existing `.flux` model, REST payload or fixture changes shape.
//! * A parallel V2 struct would duplicate the entire ~1000-line schema
//!   (geometry, constructions, schedules, weather, controls, output) for
//!   no behavioural gain, and would force every consumer that matches on
//!   [`SimulationSchema`] (the CLI `.flux` loader, the Python bindings)
//!   to handle a second payload family.
//! * The externally-tagged [`SimulationSchema`] container is the
//!   forward-compatibility mechanism: a future `SimulationSchema::V2(..)`
//!   variant can be added later without touching V1.
//!
//! [`SchemaVersion::V2`] is **reserved** for that next *breaking*
//! revision and is rejected by [`SimulationSchemaV1::validate`] while no
//! V2 payload exists, so a model cannot silently mislabel itself.
//!
//! ## Migration path
//!
//! * **V1 (pre-#3989) → V1.1** — automatic. Old documents load unchanged;
//!   the new fields default to "no per-surface surfaces", "no per-surface
//!   fenestration" and "no named HVAC system", which is exactly the old
//!   aggregate behaviour.
//! * **V1.1 → V2 (future)** — lossless by construction: every V1.1
//!   addition is optional, so a V2 payload can be derived from a V1.1
//!   payload by copying the aggregate fields and re-expressing the
//!   per-surface block. When V2 lands it will be added as
//!   `SimulationSchema::V2(SimulationSchemaV2)` plus a
//!   `SimulationSchemaV2::from_v1` conversion helper; the V1 reader
//!   stays available for the whole V1 series.
//!
//! # Core Types
//!
//! - [`SimulationSchema`]: Top-level container for all simulation data
//! - [`Geometry`]: Building geometry (zones, dimensions, per-surface specs)
//! - [`ConstructionSet`]: Building envelope constructions
//! - [`SurfaceSpec`]: Per-surface geometry (area / orientation / tilt / vertices)
//! - [`ScheduleSet`]: Time-based schedules for occupancy, lighting, HVAC
//! - [`WeatherData`]: Weather data reference or inline data
//! - [`ControlSet`]: HVAC control configurations + named HVAC system
//! - [`SimulationOutput`]: Simulation results
//!
//! # Example
//!
//! ```rust
//! use fluxion::api::schema::{
//!     SimulationSchema, SimulationSchemaV1, Geometry, ConstructionSet, ScheduleSet,
//!     WeatherData, ControlSet, SchemaVersion, SimulationOutput,
//! };
//!
//! // Create a minimal schema
//! let schema = SimulationSchema::V1(SimulationSchemaV1 {
//!     version: SchemaVersion::V1,
//!     metadata: Default::default(),
//!     geometry: Geometry::default(),
//!     constructions: ConstructionSet::default(),
//!     schedules: ScheduleSet::default(),
//!     weather: WeatherData::default(),
//!     controls: ControlSet::default(),
//!     output: SimulationOutput::default(),
//! });
//! ```

use serde::{Deserialize, Deserializer, Serialize};
use std::path::PathBuf;

use crate::sim::construction::ConstructionLayer;
use crate::sim::hvac::HVACSystemType;
use crate::sim::per_surface_conduction::SurfaceKind;
use crate::sim::schedule::{DailySchedule, HVACSchedule};
use crate::validation::ashrae_140_cases::Orientation;
use crate::weather::HourlyWeatherData;

/// Custom deserializer for the `path` field of `WeatherData::EpwFile`.
///
/// Gates inbound paths on `validate_epw_path` (Issue #2915) so that an
/// authenticated REST client cannot reach `EpwWeatherSource::from_file`
/// (which `std::fs::File::open`s the path with no canonicalization) by
/// pointing at an arbitrary server-readable file. CWE-22 closure.
fn deserialize_validated_epw_path<'de, D>(deserializer: D) -> Result<PathBuf, D::Error>
where
    D: Deserializer<'de>,
{
    let path = PathBuf::deserialize(deserializer)?;
    let as_str = path.to_string_lossy().into_owned();
    crate::weather::epw::validate_epw_path(&as_str).map_err(serde::de::Error::custom)?;
    Ok(path)
}

/// Schema version for forward compatibility.
///
/// Issue #3989 — the enum keeps the `V1` wire tag for every payload this
/// build accepts. The per-surface / fenestration / HVAC-system extension
/// shipped **additively inside V1** (schema revision 1.1, see the module
/// docs), so no existing document changes shape.
///
/// [`SchemaVersion::V2`] is reserved for the next *breaking* revision. No
/// V2 payload exists yet, so [`SimulationSchemaV1::validate`] rejects a
/// document that claims `"version": "V2"` with an actionable error instead
/// of letting it mislabel itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchemaVersion {
    V1,
    /// Issue #3989 — reserved for the next breaking schema revision.
    /// Not yet a loadable payload; see the module-level "Migration path"
    /// section.
    V2,
}

impl Default for SchemaVersion {
    fn default() -> Self {
        SchemaVersion::V1
    }
}

/// Metadata about the simulation schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaMetadata {
    pub name: String,
    pub description: String,
    pub author: Option<String>,
    pub created_at: Option<String>,
    pub schema_version: SchemaVersion,
}

impl Default for SchemaMetadata {
    fn default() -> Self {
        SchemaMetadata {
            name: "Untitled Simulation".to_string(),
            description: String::new(),
            author: None,
            created_at: None,
            schema_version: SchemaVersion::V1,
        }
    }
}

/// Zone geometry specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoneGeometry {
    pub name: String,
    pub floor_area: f64,
    pub volume: f64,
    pub height: f64,
}

impl Default for ZoneGeometry {
    fn default() -> Self {
        ZoneGeometry {
            name: "Zone 1".to_string(),
            floor_area: 48.0,
            volume: 129.6,
            height: 2.7,
        }
    }
}

/// Building geometry specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Geometry {
    pub zones: Vec<ZoneGeometry>,
    pub total_floor_area: f64,
    pub total_volume: f64,
    pub number_of_floors: usize,
    pub floor_height: f64,
    /// Issue #3989 — per-surface geometry.
    ///
    /// Empty (the default) keeps the historical zone-aggregate behaviour:
    /// the simulation derives one lumped wall / roof / floor per zone from
    /// `zones` and the `constructions` block. When populated, each entry
    /// describes one physical surface (wall, roof or floor) with its own
    /// area, orientation, tilt, optional vertex loop and optional
    /// fenestration — enough to describe a 600-series-style case without
    /// loss.
    ///
    /// Absent from the JSON when empty, so pre-#3989 documents keep their
    /// exact wire shape.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub surfaces: Vec<SurfaceSpec>,
}

impl Default for Geometry {
    fn default() -> Self {
        Geometry {
            zones: vec![ZoneGeometry::default()],
            total_floor_area: 48.0,
            total_volume: 129.6,
            number_of_floors: 1,
            floor_height: 2.7,
            surfaces: Vec::new(),
        }
    }
}

impl Geometry {
    /// Issue #3989 — look up a per-surface spec by its id.
    pub fn surface(&self, id: &str) -> Option<&SurfaceSpec> {
        self.surfaces.iter().find(|s| s.id == id)
    }

    /// Issue #3989 — all per-surface specs belonging to one zone.
    pub fn surfaces_in_zone(&self, zone_index: usize) -> impl Iterator<Item = &SurfaceSpec> {
        self.surfaces
            .iter()
            .filter(move |s| s.zone_index == zone_index)
    }

    /// Issue #3989 — every per-surface spec that faces the given
    /// orientation (walls only; roofs and floors have no compass
    /// orientation).
    pub fn surfaces_with_orientation(
        &self,
        orientation: Orientation,
    ) -> impl Iterator<Item = &SurfaceSpec> {
        self.surfaces
            .iter()
            .filter(move |s| s.orientation == Some(orientation))
    }
}

/// Window specification within a construction.
///
/// Issue #3989 — the aggregate V1 shape (a single area / U-value / SHGC
/// triple for the whole wall construction) cannot express per-orientation
/// glazing. Three optional fields make the placement explicit:
///
/// * [`WindowSpec::window_to_wall_ratio`] — the glazing fraction of the
///   host surface ("south WWR 60 %").
/// * [`WindowSpec::surface`] — the id of the [`SurfaceSpec`] the glazing
///   sits on.
/// * [`WindowSpec::orientation`] — the compass orientation the glazing
///   faces, for models that place fenestration by orientation rather than
///   by named surface.
///
/// All three are optional and omitted from the JSON when unset, so
/// pre-#3989 documents keep their exact wire shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindowSpec {
    /// Glazing area in m². When `0.0` **and**
    /// [`WindowSpec::window_to_wall_ratio`] is set, the area is derived
    /// from the host surface area (see
    /// [`WindowSpec::effective_area`]).
    pub window_area: f64,
    pub window_u_value: f64,
    pub window_shgc: f64,
    /// Issue #3989 — glazing fraction of the host surface (0-1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_to_wall_ratio: Option<f64>,
    /// Issue #3989 — id of the [`SurfaceSpec`] this glazing sits on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<String>,
    /// Issue #3989 — compass orientation this glazing faces, when the
    /// placement is given by orientation rather than by named surface.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orientation: Option<Orientation>,
}

impl Default for WindowSpec {
    fn default() -> Self {
        WindowSpec {
            window_area: 12.0,
            window_u_value: 1.5,
            window_shgc: 0.3,
            window_to_wall_ratio: None,
            surface: None,
            orientation: None,
        }
    }
}

impl WindowSpec {
    /// Issue #3989 — the glazing area actually used by the simulation.
    ///
    /// An explicit positive [`WindowSpec::window_area`] wins. Otherwise the
    /// area is derived from the host surface area and
    /// [`WindowSpec::window_to_wall_ratio`] (`host_area × wwr`), which is
    /// how a per-orientation WWR is turned into an area. Returns `0.0`
    /// when neither is usable.
    pub fn effective_area(&self, host_area: f64) -> f64 {
        if self.window_area > 0.0 {
            self.window_area
        } else {
            self.window_to_wall_ratio.unwrap_or(0.0) * host_area.max(0.0)
        }
    }
}

/// Issue #3989 — one vertex of a [`SurfaceSpec`] vertex loop.
///
/// Coordinates are metres in a right-handed frame (`x` east, `y` north,
/// `z` up). The loop is closed implicitly: the last vertex joins the
/// first.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SurfaceVertex {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Issue #3989 — a single building surface with its own geometry and
/// optional fenestration.
///
/// The zone-aggregate V1 shape can only describe "one wall construction
/// for the whole zone". A `SurfaceSpec` describes one physical surface
/// (wall, roof or floor) so a model can state, for example, "the south
/// wall is 16.2 m² of lightweight construction carrying 12 m² of
/// double clear glazing" without losing anything.
///
/// The vocabulary is the engine's own, not a parallel one:
///
/// * [`SurfaceSpec::kind`] is the per-surface conduction solver's
///   [`SurfaceKind`] (`Wall` / `Roof` / `Floor`).
/// * [`SurfaceSpec::orientation`] is the solar/orientation enum
///   [`Orientation`] (`North` / `East` / `South` / `West` / `Up` /
///   `Down` / `Horizontal`).
///
/// A surface is described by an explicit `area`, by a `vertices` loop
/// (from which the area is derived), or by both. The construction that
/// applies to it is the one matching its `kind` in the `constructions`
/// block (`wall` / `roof` / `floor`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceSpec {
    /// Stable identifier, referenced by [`WindowSpec::surface`]. Must be
    /// unique within the model.
    pub id: String,
    /// Index into [`Geometry::zones`] the surface bounds. Defaults to `0`
    /// so single-zone models can omit it.
    #[serde(default)]
    pub zone_index: usize,
    /// Wall / roof / floor classification.
    pub kind: SurfaceKind,
    /// Gross area of the surface in m². `None` means "derive it from
    /// `vertices`".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub area: Option<f64>,
    /// Compass orientation. `None` for surfaces described by `vertices`
    /// only, or for tilted roofs that give `tilt_deg` instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orientation: Option<Orientation>,
    /// Tilt from horizontal in degrees (0 = horizontal, 90 = vertical).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tilt_deg: Option<f64>,
    /// Explicit vertex loop. When present the surface geometry is exact
    /// rather than area + orientation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vertices: Option<Vec<SurfaceVertex>>,
    /// Fenestration on this surface.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<WindowSpec>,
}

impl SurfaceSpec {
    /// Issue #3989 — area of the `vertices` loop in m².
    ///
    /// Uses the polygon area vector (`½·Σ vᵢ × vᵢ₊₁`), which is exact
    /// for a planar loop regardless of its orientation in space. Returns
    /// `None` when there is no (or a degenerate) vertex loop.
    pub fn polygon_area(&self) -> Option<f64> {
        let verts = self.vertices.as_ref()?;
        if verts.len() < 3 {
            return None;
        }
        let (mut ax, mut ay, mut az) = (0.0_f64, 0.0_f64, 0.0_f64);
        for i in 0..verts.len() {
            let a = verts[i];
            let b = verts[(i + 1) % verts.len()];
            ax += a.y * b.z - a.z * b.y;
            ay += a.z * b.x - a.x * b.z;
            az += a.x * b.y - a.y * b.x;
        }
        let magnitude = (ax * ax + ay * ay + az * az).sqrt();
        if magnitude <= 0.0 {
            None
        } else {
            Some(0.5 * magnitude)
        }
    }

    /// Issue #3989 — the area the simulation should use: the explicit
    /// `area` when given, otherwise the area derived from `vertices`.
    pub fn effective_area(&self) -> Option<f64> {
        self.area.or_else(|| self.polygon_area())
    }

    /// Issue #3989 — glazing area on this surface in m² (0 when the
    /// surface carries no glazing).
    pub fn window_area(&self) -> f64 {
        match &self.window {
            Some(w) => w.effective_area(self.effective_area().unwrap_or(0.0)),
            None => 0.0,
        }
    }

    /// Issue #3989 — the glazing fraction of this surface: the explicit
    /// window-to-wall ratio when given, otherwise the ratio implied by
    /// the glazing area and the surface area.
    pub fn window_to_wall_ratio(&self) -> Option<f64> {
        if let Some(w) = &self.window {
            if let Some(wwr) = w.window_to_wall_ratio {
                return Some(wwr);
            }
        }
        let area = self.effective_area()?;
        if area <= 0.0 {
            return None;
        }
        Some(self.window_area() / area)
    }
}

/// Surface construction specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceConstruction {
    pub name: String,
    pub layers: Vec<ConstructionLayer>,
    pub window: Option<WindowSpec>,
}

impl Default for SurfaceConstruction {
    fn default() -> Self {
        SurfaceConstruction {
            name: "Default Wall".to_string(),
            layers: vec![
                ConstructionLayer::new("Plasterboard", 0.16, 950.0, 840.0, 0.012),
                ConstructionLayer::new("Fiberglass", 0.04, 12.0, 840.0, 0.066),
                ConstructionLayer::new("Wood siding", 0.14, 500.0, 1300.0, 0.009),
            ],
            window: Some(WindowSpec::default()),
        }
    }
}

/// Set of construction assemblies for a building.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstructionSet {
    pub wall: SurfaceConstruction,
    pub roof: SurfaceConstruction,
    pub floor: SurfaceConstruction,
    pub interzone: Option<SurfaceConstruction>,
}

impl Default for ConstructionSet {
    fn default() -> Self {
        ConstructionSet {
            wall: SurfaceConstruction::default(),
            roof: SurfaceConstruction::default(),
            floor: SurfaceConstruction::default(),
            interzone: None,
        }
    }
}

/// Schedule set for building operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScheduleSet {
    pub occupancy: DailySchedule,
    pub lighting: DailySchedule,
    pub hvac: HVACSchedule,
    pub infiltration: Option<DailySchedule>,
    /// Issue #4101 — lighting power density in W/m². When `None` (or
    /// non-positive) the run keeps the historical behaviour: the REST path
    /// applies a zero lighting schedule and no auto-loaded profile. Set to
    /// a positive value to apply `schedules.lighting` (hourly fractions
    /// 0-1) at this density and meter it as an end-use series.
    #[serde(default)]
    pub lighting_power_density_w_m2: Option<f64>,
    /// Issue #4101 — plug/process equipment applied to the run and metered
    /// as an end-use series. Empty keeps the historical behaviour (no
    /// equipment gains). Each entry becomes one
    /// [`crate::sim::equipment::Equipment`] via [`EquipmentSpec::build`].
    #[serde(default)]
    pub equipment: Vec<EquipmentSpec>,
}

impl Default for ScheduleSet {
    fn default() -> Self {
        ScheduleSet {
            occupancy: DailySchedule::weekly("Occupancy".to_string()),
            lighting: DailySchedule::weekly("Lighting".to_string()),
            hvac: HVACSchedule::constant_schedule(20.0, 24.0)
                .expect("constant_schedule on fresh daily schedules cannot fail"),
            infiltration: None,
            lighting_power_density_w_m2: None,
            equipment: Vec::new(),
        }
    }
}

/// Issue #4101 — schema-supplied plug/process equipment.
///
/// A minimal, hand-computable description of one equipment item. The REST
/// `/v1/simulate` path (and the CLI schema path) converts each spec into a
/// real [`crate::sim::equipment::Equipment`] with
/// [`EquipmentSpec::build`]; the item's thermal gains are applied through
/// `StepParameters` and its electric draw is metered per timestep via
/// `Equipment::power_at_hour`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EquipmentSpec {
    /// Equipment class: `"computer"`, `"server"`, or `"generic"`
    /// (case-insensitive).
    pub equipment_type: String,
    /// Rated electric power per unit, Watts.
    pub rated_power_w: f64,
    /// Number of units. Defaults to 1.
    #[serde(default = "default_equipment_count")]
    pub count: usize,
    /// Hourly utilization fractions (0-1), indexed by hour of day.
    /// Defaults to always-on.
    #[serde(default = "always_on_hourly_fractions")]
    pub hourly_fractions: [f64; 24],
    /// Fraction of the heat release that is radiative (0-1). Defaults to 0.3.
    #[serde(default = "default_radiative_fraction")]
    pub radiative_fraction: f64,
    /// Fraction of the heat release that is convective (0-1). Defaults to 0.7.
    #[serde(default = "default_convective_fraction")]
    pub convective_fraction: f64,
    /// Fraction of radiative heat absorbed by thermal mass (0-1).
    /// Defaults to 0.2.
    #[serde(default = "default_mass_coupling_factor")]
    pub mass_coupling_factor: f64,
}

fn default_equipment_count() -> usize {
    1
}

fn always_on_hourly_fractions() -> [f64; 24] {
    [1.0; 24]
}

fn default_radiative_fraction() -> f64 {
    0.3
}

fn default_convective_fraction() -> f64 {
    0.7
}

fn default_mass_coupling_factor() -> f64 {
    0.2
}

impl EquipmentSpec {
    /// Validate the spec, returning human-readable errors (empty = valid).
    /// Called from [`SimulationSchemaV1::validate`].
    pub fn validate(&self, index: usize) -> Vec<ValidationError> {
        let base = format!("schedules.equipment[{index}]");
        let mut errors = Vec::new();
        match self.equipment_type.to_ascii_lowercase().as_str() {
            "computer" | "server" | "generic" => {}
            other => errors.push(ValidationError::new(
                format!("{base}.equipment_type"),
                format!("unknown equipment type {other:?}"),
                "use \"computer\", \"server\", or \"generic\"".to_string(),
            )),
        }
        if self.rated_power_w.is_nan() || self.rated_power_w < 0.0 {
            errors.push(ValidationError::new(
                format!("{base}.rated_power_w"),
                format!("must be >= 0, got {}", self.rated_power_w),
                "set rated_power_w to the per-unit electric draw in Watts".to_string(),
            ));
        }
        if self.count == 0 {
            errors.push(ValidationError::new(
                format!("{base}.count"),
                "must be >= 1, got 0".to_string(),
                "set count to the number of installed units".to_string(),
            ));
        }
        for (h, &f) in self.hourly_fractions.iter().enumerate() {
            if !(0.0..=1.0).contains(&f) {
                errors.push(ValidationError::new(
                    format!("{base}.hourly_fractions[{h}]"),
                    format!("must be in [0, 1], got {f}"),
                    "set the hour's utilization fraction between 0 and 1".to_string(),
                ));
            }
        }
        let total = self.radiative_fraction + self.convective_fraction;
        if (total - 1.0).abs() > 1e-9 {
            errors.push(ValidationError::new(
                format!("{base}.radiative_fraction"),
                format!("radiative + convective fractions must sum to 1.0, got {total}"),
                "adjust radiative_fraction / convective_fraction to sum to 1.0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.mass_coupling_factor) {
            errors.push(ValidationError::new(
                format!("{base}.mass_coupling_factor"),
                format!("must be in [0, 1], got {}", self.mass_coupling_factor),
                "set the mass coupling factor between 0 and 1".to_string(),
            ));
        }
        errors
    }

    /// Build the runtime equipment object. `id` names the item (the REST
    /// path generates `schema-equipment-{index}`).
    ///
    /// # Errors
    /// Returns `Err` for an unknown `equipment_type`.
    pub fn build(&self, id: String) -> Result<Box<dyn crate::sim::equipment::Equipment>, String> {
        use crate::sim::equipment::{ComputerEquipment, Equipment, GenericEquipment, ServerRack};

        let mut schedule = DailySchedule::new();
        for (hour, &fraction) in self.hourly_fractions.iter().enumerate() {
            schedule
                .set_hour(hour, fraction)
                .map_err(|e| format!("equipment {id}: invalid schedule: {e}"))?;
        }

        // Concrete fields are public; apply the spec's heat-split fractions
        // after the builder (builders only set id / power / count / schedule).
        match self.equipment_type.to_ascii_lowercase().as_str() {
            "computer" => {
                let mut item = ComputerEquipment::new(id, self.rated_power_w, self.count)
                    .with_schedule(schedule);
                item.radiative_fraction = self.radiative_fraction;
                item.convective_fraction = self.convective_fraction;
                item.mass_coupling_factor = self.mass_coupling_factor;
                Ok(Box::new(item) as Box<dyn Equipment>)
            }
            "server" => {
                let mut item =
                    ServerRack::new(id, self.rated_power_w, self.count).with_schedule(schedule);
                item.radiative_fraction = self.radiative_fraction;
                item.convective_fraction = self.convective_fraction;
                item.mass_coupling_factor = self.mass_coupling_factor;
                Ok(Box::new(item) as Box<dyn Equipment>)
            }
            "generic" => {
                let mut item = GenericEquipment::new(id, self.rated_power_w, self.count)
                    .with_schedule(schedule);
                item.radiative_fraction = self.radiative_fraction;
                item.convective_fraction = self.convective_fraction;
                item.mass_coupling_factor = self.mass_coupling_factor;
                Ok(Box::new(item) as Box<dyn Equipment>)
            }
            other => Err(format!(
                "equipment {id}: unknown equipment_type {other:?} \
                 (expected \"computer\", \"server\", or \"generic\")"
            )),
        }
    }
}

/// Weather data specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WeatherData {
    /// Reference to an external EPW file.
    ///
    /// Deserialization is gated by `validate_epw_path` (Issue #2915) so an
    /// authenticated REST client cannot point at arbitrary server-readable
    /// files via `WeatherData::EpwFile { path }` on `/v1/simulate` or
    /// `/v1/campaign/*` requests.
    #[serde(rename = "epw")]
    EpwFile {
        #[serde(deserialize_with = "deserialize_validated_epw_path")]
        path: PathBuf,
    },

    /// Reference to an embedded TMY location.
    #[serde(rename = "tmy")]
    TmyLocation { location: String },

    /// Inline hourly weather data.
    #[serde(rename = "inline")]
    Inline { hourly_data: Vec<HourlyWeatherData> },
}

impl Default for WeatherData {
    fn default() -> Self {
        WeatherData::TmyLocation {
            location: "Denver, CO".to_string(),
        }
    }
}

/// HVAC control configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlConfig {
    pub heating_setpoint: f64,
    pub cooling_setpoint: f64,
    /// Issue #4103 — setpoint tolerance (°C) used ONLY for unmet-hours
    /// reporting. Defaults to [`UNMET_HOURS_TOLERANCE_DEFAULT_C`] (0.2 °C,
    /// matching EnergyPlus "Time Setpoint Not Met", EnergyPlus Input
    /// Output Reference, System Summary table) when the schema leaves the
    /// field unset; explicit values are honored unchanged. This field is
    /// not a physics control deadband — the HVAC controllers keep their
    /// own 0.5 °C default.
    #[serde(default = "default_unmet_hours_tolerance")]
    pub deadband_tolerance: f64,
    pub heating_capacity: f64,
    pub cooling_capacity: f64,
}

/// Issue #4103 — default setpoint tolerance for unmet-hours reporting:
/// 0.2 °C, matching EnergyPlus "Time Setpoint Not Met" (EnergyPlus Input
/// Output Reference, System Summary table, which applies a 0.2 °C
/// tolerance band when counting occupied hours the setpoint was not met).
pub const UNMET_HOURS_TOLERANCE_DEFAULT_C: f64 = 0.2;

fn default_unmet_hours_tolerance() -> f64 {
    UNMET_HOURS_TOLERANCE_DEFAULT_C
}

impl Default for ControlConfig {
    fn default() -> Self {
        ControlConfig {
            heating_setpoint: 20.0,
            cooling_setpoint: 24.0,
            // Issue #4103: unset schemas default to the EnergyPlus
            // "Time Setpoint Not Met" 0.2 °C tolerance, not the 0.5 °C
            // HVAC control deadband.
            deadband_tolerance: UNMET_HOURS_TOLERANCE_DEFAULT_C,
            heating_capacity: 100_000.0,
            cooling_capacity: 100_000.0,
        }
    }
}

/// Set of control configurations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlSet {
    pub zone_control: ControlConfig,
    pub global_control: Option<ControlConfig>,
    /// Issue #3989 — named HVAC system selection.
    ///
    /// Selects one of the engine's own HVAC system templates
    /// ([`HVACSystemType`]: `Simple`, `VAV`, `CAV`, `HeatPump`,
    /// `DOASFanCoil`, `Ideal`).
    /// `None` (the default) keeps the historical behaviour: the zone
    /// setpoints and the heating / cooling capacities in
    /// [`ControlConfig`] drive the simulation, and no system template is
    /// named.
    ///
    /// The variant names are the engine's own
    /// ([`crate::sim::hvac::HVACSystemType`]) — the schema does not
    /// introduce a parallel vocabulary. The serde representation is the
    /// plain variant name, e.g.
    /// `"hvac_system": "DOASFanCoil"`.
    ///
    /// # Template catalogue — modelling assumptions and limits
    ///
    /// The table below summarises each selectable template; the
    /// authoritative, fully detailed documentation lives on the engine enum
    /// ([`crate::sim::hvac::HVACSystemType`]) and is kept in sync with this
    /// table.
    ///
    /// | Template | Air side | Zone sensible load | Zone latent load | Ventilation |
    /// |---|---|---|---|---|
    /// | `Simple` | none (on/off, fixed capacity) | met by capacity | not modelled | not modelled |
    /// | `VAV` | packaged VAV terminal | met by airflow + reheat | not modelled | not modelled |
    /// | `CAV` | packaged CAV terminal | met by fixed airflow | not modelled | not modelled |
    /// | `HeatPump` | none (unitary / DX) | met by capacity | not modelled | not modelled |
    /// | `DOASFanCoil` | DOAS + four-pipe fan coil | met by the fan-coil water side | removed at the DOAS | met by the DOAS (100 % outdoor air) |
    /// | `Ideal` | ideal loads | met exactly | not modelled | not modelled |
    ///
    /// In short:
    ///
    /// - `Simple` — binary on/off with fixed heating / cooling capacity and
    ///   no airflow, humidity or ventilation modelling; a load above the
    ///   fixed capacity cannot be met.
    /// - `VAV` — packaged variable-air-volume terminal (fan, cooling coil,
    ///   optional reheat) with a minimum airflow of 30 % of rated flow and
    ///   an SPR-compensated fan-power curve; sensible-only, so it needs a
    ///   separate ventilation path.
    /// - `CAV` — packaged constant-air-volume terminal; the airflow and the
    ///   fan power are constant whenever the system runs, which is
    ///   deliberately poor at part load; sensible-only.
    /// - `HeatPump` — single reversible unit with rated COP / EER and
    ///   polynomial efficiency curves; capacity degrades with outdoor
    ///   temperature but COP is held constant; no defrost, no backup heat,
    ///   no latent removal.
    /// - `DOASFanCoil` — constant-volume dedicated outdoor air system
    ///   (cooling coil to a target dew point + neutral reheat + optional
    ///   adiabatic humidification) serving a four-pipe fan-coil unit that
    ///   handles the zone sensible load; the zone latent load is removed at
    ///   the DOAS and the ventilation air comes entirely from the DOAS. The
    ///   DOAS reports a missed dew-point target when its cooling capacity is
    ///   exceeded, and the fan-coil model is coarse (fixed rated flows,
    ///   clamped part-load ratio, no central plant).
    /// - `Ideal` — infinite-capacity ideal air loads, used for ASHRAE 140
    ///   validation; the load is met exactly and the reported energy is the
    ///   thermodynamic minimum, so it must not be used for design.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hvac_system: Option<HVACSystemType>,
}

impl Default for ControlSet {
    fn default() -> Self {
        ControlSet {
            zone_control: ControlConfig::default(),
            global_control: None,
            hvac_system: None,
        }
    }
}

/// Simulation output results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationOutput {
    pub eui: f64,
    pub total_energy: f64,
    /// Peak heating demand observed during the run, in Watts.
    pub peak_heating_load: f64,
    /// Peak cooling demand observed during the run, in Watts.
    pub peak_cooling_load: f64,
    pub heating_energy: f64,
    pub cooling_energy: f64,
    pub zone_temperatures: Option<Vec<f64>>,
    /// Issue #763 — full hourly zone temperature profiles.
    /// Format: [num_zones][8760] hourly temperatures in °C.
    pub hourly_zone_temperatures: Option<Vec<Vec<f64>>>,
    /// Issue #3305 — the zone solver that ACTUALLY executed, derived from
    /// the step dispatcher's real per-step outcome (gauge success vs.
    /// 5R1C/9R4C fall-through), not from the requested selector. Populated
    /// by the REST `/v1/simulate` run on success; schema-embedded output
    /// templates leave it `None`, and it is omitted from serialisation when
    /// `None` so existing wire shapes are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_solver: Option<String>,
    /// Issue #3988 — unmet heating hours (occupied-only): occupied hours
    /// where a zone's temperature fell below
    /// `heating_setpoint - deadband_tolerance`, summed across zones. The
    /// tolerance defaults to 0.2 °C when the schema leaves
    /// `deadband_tolerance` unset (EnergyPlus "Time Setpoint Not Met"
    /// parity; EnergyPlus Input Output Reference, System Summary table).
    /// See [`SimulationOutput::unmet_hours`].
    #[serde(default)]
    pub unmet_heating_hours: f64,
    /// Issue #3988 — unmet cooling hours (occupied-only): occupied hours
    /// where a zone's temperature rose above
    /// `cooling_setpoint + deadband_tolerance`, summed across zones.
    /// Tolerance default as for `unmet_heating_hours`. See
    /// [`SimulationOutput::unmet_hours`].
    #[serde(default)]
    pub unmet_cooling_hours: f64,
    /// Issue #4103 — unmet heating hours over ALL hours (per ASHRAE 90.1
    /// Appendix G §G3.1.2.2): same setpoint-deviation test as
    /// `unmet_heating_hours` but evaluated for every timestep, not just
    /// occupied hours. The 90.1 Performance Rating Method caps these at
    /// 300 hours and requires proposed-design unmet hours ≤ baseline + 50.
    /// See [`SimulationOutput::unmet_hours_all_hours`].
    #[serde(default)]
    pub unmet_heating_hours_all_hours: f64,
    /// Issue #4103 — unmet cooling hours over ALL hours (ASHRAE 90.1
    /// Appendix G §G3.1.2.2). See `unmet_heating_hours_all_hours`.
    #[serde(default)]
    pub unmet_cooling_hours_all_hours: f64,
    /// Issue #4101 — timestep-indexed heating energy, kWh per timestep.
    /// One entry per solver timestep; omitted when the run did not record
    /// end-use metering so existing wire shapes are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hourly_heating_kwh: Option<Vec<f64>>,
    /// Issue #4101 — timestep-indexed cooling energy, kWh per timestep.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hourly_cooling_kwh: Option<Vec<f64>>,
    /// Issue #4101 — timestep-indexed lighting electric energy, kWh per
    /// timestep.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hourly_lighting_kwh: Option<Vec<f64>>,
    /// Issue #4101 — timestep-indexed equipment electric energy, kWh per
    /// timestep.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hourly_equipment_kwh: Option<Vec<f64>>,
    /// Issue #4102 — monthly heating energy, kWh per month. Outer vec is
    /// the 0-based year index, inner vec is 12 calendar months (Jan = 0).
    /// Pure post-processing of [`SimulationOutput::hourly_heating_kwh`];
    /// omitted when the run did not record end-use metering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monthly_heating_kwh: Option<Vec<Vec<f64>>>,
    /// Issue #4102 — monthly heating peak demand, max timestep-average kW
    /// per month. Same `[year][month]` shape as `monthly_heating_kwh`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monthly_heating_peak_kw: Option<Vec<Vec<f64>>>,
    /// Issue #4102 — monthly cooling energy, kWh per month (`[year][month]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monthly_cooling_kwh: Option<Vec<Vec<f64>>>,
    /// Issue #4102 — monthly cooling peak demand, kW (`[year][month]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monthly_cooling_peak_kw: Option<Vec<Vec<f64>>>,
    /// Issue #4102 — monthly lighting energy, kWh per month (`[year][month]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monthly_lighting_kwh: Option<Vec<Vec<f64>>>,
    /// Issue #4102 — monthly lighting peak demand, kW (`[year][month]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monthly_lighting_peak_kw: Option<Vec<Vec<f64>>>,
    /// Issue #4102 — monthly equipment energy, kWh per month (`[year][month]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monthly_equipment_kwh: Option<Vec<Vec<f64>>>,
    /// Issue #4102 — monthly equipment peak demand, kW (`[year][month]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monthly_equipment_peak_kw: Option<Vec<Vec<f64>>>,
}

impl Default for SimulationOutput {
    fn default() -> Self {
        SimulationOutput {
            eui: 0.0,
            total_energy: 0.0,
            peak_heating_load: 0.0,
            peak_cooling_load: 0.0,
            heating_energy: 0.0,
            cooling_energy: 0.0,
            zone_temperatures: None,
            hourly_zone_temperatures: None,
            effective_solver: None,
            unmet_heating_hours: 0.0,
            unmet_cooling_hours: 0.0,
            unmet_heating_hours_all_hours: 0.0,
            unmet_cooling_hours_all_hours: 0.0,
            hourly_heating_kwh: None,
            hourly_cooling_kwh: None,
            hourly_lighting_kwh: None,
            hourly_equipment_kwh: None,
            monthly_heating_kwh: None,
            monthly_heating_peak_kw: None,
            monthly_cooling_kwh: None,
            monthly_cooling_peak_kw: None,
            monthly_lighting_kwh: None,
            monthly_lighting_peak_kw: None,
            monthly_equipment_kwh: None,
            monthly_equipment_peak_kw: None,
        }
    }
}

/// Issue #4102 — monthly end-use summary produced by
/// [`SimulationOutput::monthly_end_use_summary`].
///
/// Both arrays are `[end_use][year][month]` with end uses in
/// heating/cooling/lighting/equipment order and months Jan = 0. `kwh`
/// holds the month's energy sum; `peak_kw` holds the month's maximum
/// timestep-average demand.
#[derive(Debug, Clone, Default)]
pub struct MonthlyEndUseSummary {
    pub kwh: [Vec<Vec<f64>>; 4],
    pub peak_kw: [Vec<Vec<f64>>; 4],
}

impl SimulationOutput {
    /// Issue #4102 — monthly end-use summary: 12 bins per simulated year
    /// per end use, derived from the timestep-indexed hourly series
    /// (#4101). Pure post-processing — no sim-loop involvement.
    ///
    /// Each input series is kWh per timestep; `dt_seconds` is the run's
    /// constant timestep duration. Returns `None` when any series is
    /// missing or empty (all-or-nothing, mirroring the #4101 exposure).
    /// Otherwise the arrays are `[end_use][year][month]` in
    /// heating/cooling/lighting/equipment order, with timestep 0 starting
    /// Jan 1 00:00 of the weather year (non-leap, per ASHRAE 140
    /// convention).
    pub fn monthly_end_use_summary(
        heating_kwh: Option<Vec<f64>>,
        cooling_kwh: Option<Vec<f64>>,
        lighting_kwh: Option<Vec<f64>>,
        equipment_kwh: Option<Vec<f64>>,
        dt_seconds: f64,
    ) -> Option<MonthlyEndUseSummary> {
        use crate::validation::report::BenchmarkReport;
        let series = [heating_kwh?, cooling_kwh?, lighting_kwh?, equipment_kwh?];
        let mut kwh: [Vec<Vec<f64>>; 4] = Default::default();
        let mut peak_kw: [Vec<Vec<f64>>; 4] = Default::default();
        for (u, s) in series.iter().enumerate() {
            let bins = BenchmarkReport::calculate_monthly_end_use(s, dt_seconds, false);
            if bins.is_empty() {
                return None;
            }
            let years = bins.len().div_ceil(12);
            let mut kwh_years = vec![vec![0.0; 12]; years];
            let mut peak_years = vec![vec![0.0; 12]; years];
            for (flat, bin) in bins.iter().enumerate() {
                kwh_years[flat / 12][flat % 12] = bin.kwh;
                peak_years[flat / 12][flat % 12] = bin.peak_kw;
            }
            kwh[u] = kwh_years;
            peak_kw[u] = peak_years;
        }
        Some(MonthlyEndUseSummary { kwh, peak_kw })
    }

    /// Compute unmet heating/cooling hours from hourly zone temperatures.
    ///
    /// Issue #3988. `hourly_temps` is zone-major (`[zone][timestep]`, one
    /// step per hour) as returned by the thermal model's
    /// `get_hourly_temperatures`. An hour counts as occupied when the
    /// occupancy schedule's value for that hour-of-day exceeds 0.05 (the
    /// weekday profile is used for weekly schedules). A zone-hour is
    /// unmet-heating when its temperature falls below
    /// `heating_setpoint - tolerance`, unmet-cooling when it rises above
    /// `cooling_setpoint + tolerance`.
    ///
    /// Returns `(unmet_heating_hours, unmet_cooling_hours)` summed across
    /// zones. Shared by the REST path and the CLI direct path.
    pub fn unmet_hours(
        hourly_temps: &[Vec<f64>],
        occupancy: &DailySchedule,
        heating_setpoint: f64,
        cooling_setpoint: f64,
        tolerance: f64,
    ) -> (f64, f64) {
        Self::unmet_hours_impl(
            hourly_temps,
            Some(occupancy),
            heating_setpoint,
            cooling_setpoint,
            tolerance,
        )
    }

    /// Compute unmet heating/cooling hours over ALL timesteps.
    ///
    /// Issue #4103. Same setpoint-deviation test as [`Self::unmet_hours`]
    /// but evaluated for every timestep instead of only occupied ones, per
    /// ASHRAE 90.1 Appendix G §G3.1.2.2 (unmet load hours). The 90.1
    /// Performance Rating Method caps these at 300 hours and requires the
    /// proposed design to stay within baseline + 50 unmet hours.
    ///
    /// Returns `(unmet_heating_hours_all_hours,
    /// unmet_cooling_hours_all_hours)` summed across zones.
    pub fn unmet_hours_all_hours(
        hourly_temps: &[Vec<f64>],
        heating_setpoint: f64,
        cooling_setpoint: f64,
        tolerance: f64,
    ) -> (f64, f64) {
        Self::unmet_hours_impl(
            hourly_temps,
            None,
            heating_setpoint,
            cooling_setpoint,
            tolerance,
        )
    }

    /// Shared implementation for [`Self::unmet_hours`] (occupied-only,
    /// `occupancy = Some`) and [`Self::unmet_hours_all_hours`] (all hours,
    /// `occupancy = None`).
    fn unmet_hours_impl(
        hourly_temps: &[Vec<f64>],
        occupancy: Option<&DailySchedule>,
        heating_setpoint: f64,
        cooling_setpoint: f64,
        tolerance: f64,
    ) -> (f64, f64) {
        let mut unmet_heating = 0.0;
        let mut unmet_cooling = 0.0;
        for zone_temps in hourly_temps {
            for (t, &temp) in zone_temps.iter().enumerate() {
                if let Some(occ) = occupancy {
                    if occ.value(t % 24) <= 0.05 {
                        continue;
                    }
                }
                if temp < heating_setpoint - tolerance {
                    unmet_heating += 1.0;
                } else if temp > cooling_setpoint + tolerance {
                    unmet_cooling += 1.0;
                }
            }
        }
        (unmet_heating, unmet_cooling)
    }
}

/// Version 1 of the simulation schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationSchemaV1 {
    pub version: SchemaVersion,
    pub metadata: SchemaMetadata,
    pub geometry: Geometry,
    pub constructions: ConstructionSet,
    pub schedules: ScheduleSet,
    pub weather: WeatherData,
    pub controls: ControlSet,
    pub output: SimulationOutput,
}

/// Unified simulation schema container with version support.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SimulationSchema {
    V1(SimulationSchemaV1),
}

impl SimulationSchema {
    pub fn v1(schema: SimulationSchemaV1) -> Self {
        SimulationSchema::V1(schema)
    }

    pub fn version(&self) -> SchemaVersion {
        match self {
            SimulationSchema::V1(s) => s.version,
        }
    }
}

/// A single actionable input-validation error.
///
/// Issue #3993: the schema used to fail with panics or silently clamp bad
/// values (`.max(1.0)`). Each error names the offending field (JSON-path
/// style), states the problem, and tells the modeler how to fix it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationError {
    /// Field path, e.g. `geometry.zones[0].floor_area`.
    pub field: String,
    /// What is wrong, e.g. `must be > 0, got -5.0`.
    pub problem: String,
    /// How to fix it, e.g. `set floor_area to the zone's floor area in m²`.
    pub fix: String,
}

impl ValidationError {
    fn new(field: impl Into<String>, problem: impl Into<String>, fix: impl Into<String>) -> Self {
        ValidationError {
            field: field.into(),
            problem: problem.into(),
            fix: fix.into(),
        }
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}. Fix: {}", self.field, self.problem, self.fix)
    }
}

/// Issue #3989 — shared validation for a [`WindowSpec`].
///
/// `base` is the JSON path of the window object (e.g.
/// `geometry.surfaces[0].window` or `constructions.wall.window`).
/// `known_surface_ids` lists the surface ids the model defines; when it is
/// empty the surface cross-reference check is skipped (aggregate
/// construction windows have no host surface).
fn validate_window_spec(
    window: &WindowSpec,
    base: &str,
    known_surface_ids: &[&str],
) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    // A zero area is only acceptable when the area is derived from a
    // window-to-wall ratio against the host surface.
    if window.window_area <= 0.0 && window.window_to_wall_ratio.is_none() {
        errors.push(ValidationError::new(
            format!("{base}.window_area"),
            format!("must be > 0, got {}", window.window_area),
            "set window_area to the total window area in m², or remove the window object for no glazing",
        ));
    }
    if window.window_u_value <= 0.0 {
        errors.push(ValidationError::new(
            format!("{base}.window_u_value"),
            format!("must be > 0, got {}", window.window_u_value),
            "set window_u_value to the glazing U-value in W/m²·K (typical: 1.0–3.0)",
        ));
    }
    if !(0.0 < window.window_shgc && window.window_shgc <= 1.0) {
        errors.push(ValidationError::new(
            format!("{base}.window_shgc"),
            format!("must be in (0, 1], got {}", window.window_shgc),
            "set window_shgc to the solar heat gain coefficient as a fraction (typical: 0.2–0.8)",
        ));
    }
    if let Some(wwr) = window.window_to_wall_ratio {
        if !(0.0 < wwr && wwr <= 1.0) {
            errors.push(ValidationError::new(
                format!("{base}.window_to_wall_ratio"),
                format!("must be in (0, 1], got {wwr}"),
                "set window_to_wall_ratio to the glazing fraction of the host surface (e.g. 0.2 for 20 % WWR)",
            ));
        }
    }
    if let Some(id) = &window.surface {
        if !known_surface_ids.is_empty() && !known_surface_ids.contains(&id.as_str()) {
            errors.push(ValidationError::new(
                format!("{base}.surface"),
                format!("references unknown surface id {id:?}"),
                "point the window at one of the geometry.surfaces ids",
            ));
        }
    }
    errors
}

impl SimulationSchemaV1 {
    /// Validate the schema, returning every actionable error found.
    ///
    /// An empty vec means the schema is valid. Callers (REST
    /// `/v1/simulate`, the `fluxion` CLI direct path) must reject the input
    /// when this is non-empty rather than simulating with clamped values.
    pub fn validate(&self) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // --- Schema version (Issue #3989) ---
        //
        // The #3989 extension shipped additively inside V1, so a V1
        // payload must still declare V1. `SchemaVersion::V2` is reserved
        // for the next breaking revision and has no payload yet, so a
        // document claiming it is rejected here rather than silently
        // mislabelled.
        if self.version != SchemaVersion::V1 {
            errors.push(ValidationError::new(
                "version",
                format!(
                    "must be \"V1\" for a V1 payload, got {:?}",
                    self.version
                ),
                "set version to \"V1\" — the per-surface extension (Issue #3989) is additive inside V1",
            ));
        }

        // --- Geometry ---
        if self.geometry.zones.is_empty() {
            errors.push(ValidationError::new(
                "geometry.zones",
                "must contain at least one zone, got 0",
                "add a zone object with name, floor_area (m²), volume (m³), height (m)",
            ));
        }
        for (i, zone) in self.geometry.zones.iter().enumerate() {
            let base = format!("geometry.zones[{i}]");
            if zone.floor_area <= 0.0 {
                errors.push(ValidationError::new(
                    format!("{base}.floor_area"),
                    format!("must be > 0, got {}", zone.floor_area),
                    "set floor_area to the zone's floor area in m²",
                ));
            }
            if zone.height <= 0.0 {
                errors.push(ValidationError::new(
                    format!("{base}.height"),
                    format!("must be > 0, got {}", zone.height),
                    "set height to the zone's floor-to-ceiling height in m",
                ));
            }
            if zone.volume <= 0.0 {
                errors.push(ValidationError::new(
                    format!("{base}.volume"),
                    format!("must be > 0, got {}", zone.volume),
                    "set volume to the zone's air volume in m³ (usually floor_area × height)",
                ));
            }
        }
        if self.geometry.total_floor_area <= 0.0 {
            errors.push(ValidationError::new(
                "geometry.total_floor_area",
                format!("must be > 0, got {}", self.geometry.total_floor_area),
                "set total_floor_area to the sum of zone floor areas in m²",
            ));
        }

        // --- Per-surface geometry (Issue #3989) ---
        let mut seen_surface_ids: Vec<&str> = Vec::new();
        let all_surface_ids: Vec<&str> = self
            .geometry
            .surfaces
            .iter()
            .map(|s| s.id.as_str())
            .collect();
        for (i, surface) in self.geometry.surfaces.iter().enumerate() {
            let base = format!("geometry.surfaces[{i}]");
            if surface.id.trim().is_empty() {
                errors.push(ValidationError::new(
                    format!("{base}.id"),
                    "must not be empty".to_string(),
                    "give the surface a unique id (e.g. \"south_wall\") so fenestration can reference it",
                ));
            } else if seen_surface_ids.contains(&surface.id.as_str()) {
                errors.push(ValidationError::new(
                    format!("{base}.id"),
                    format!("duplicate surface id {:?}", surface.id),
                    "surface ids must be unique — rename one of the surfaces",
                ));
            } else {
                seen_surface_ids.push(surface.id.as_str());
            }
            if surface.zone_index >= self.geometry.zones.len() {
                errors.push(ValidationError::new(
                    format!("{base}.zone_index"),
                    format!(
                        "must be < {} (the number of zones), got {}",
                        self.geometry.zones.len(),
                        surface.zone_index
                    ),
                    "point the surface at an existing zone index",
                ));
            }
            match surface.effective_area() {
                None => errors.push(ValidationError::new(
                    format!("{base}.area"),
                    "must be > 0, or given as a vertex loop of at least 3 points".to_string(),
                    "set area to the gross surface area in m², or add a vertices loop",
                )),
                Some(area) if area <= 0.0 => errors.push(ValidationError::new(
                    format!("{base}.area"),
                    format!("must be > 0, got {area}"),
                    "set area to the gross surface area in m²",
                )),
                Some(_) => {}
            }
            if let Some(tilt) = surface.tilt_deg {
                if !(0.0..=90.0).contains(&tilt) {
                    errors.push(ValidationError::new(
                        format!("{base}.tilt_deg"),
                        format!("must be in [0, 90], got {tilt}"),
                        "set tilt_deg to the tilt from horizontal in degrees (0 = horizontal, 90 = vertical)",
                    ));
                }
            }
            if let Some(vertices) = &surface.vertices {
                if vertices.len() < 3 {
                    errors.push(ValidationError::new(
                        format!("{base}.vertices"),
                        format!("must contain at least 3 points, got {}", vertices.len()),
                        "give the surface a closed vertex loop of at least 3 points",
                    ));
                }
            }
            if let Some(window) = &surface.window {
                errors.extend(validate_window_spec(
                    window,
                    &format!("{base}.window"),
                    &all_surface_ids,
                ));
            }
        }

        // --- Constructions ---
        for (label, sc) in [
            ("wall", &self.constructions.wall),
            ("roof", &self.constructions.roof),
            ("floor", &self.constructions.floor),
        ] {
            let base = format!("constructions.{label}");
            if sc.layers.is_empty() {
                errors.push(ValidationError::new(
                    format!("{base}.layers"),
                    "must contain at least one material layer, got 0",
                    "add at least one layer with conductivity (W/m·K), density (kg/m³), specific_heat (J/kg·K), thickness (m)",
                ));
            }
            for (j, layer) in sc.layers.iter().enumerate() {
                let lbase = format!("{base}.layers[{j}]");
                for (prop, v) in [
                    ("thickness", layer.thickness),
                    ("conductivity", layer.conductivity),
                    ("density", layer.density),
                    ("specific_heat", layer.specific_heat),
                ] {
                    if v <= 0.0 {
                        errors.push(ValidationError::new(
                            format!("{lbase}.{prop}"),
                            format!("must be > 0, got {v}"),
                            format!(
                                "set {prop} to a positive physical value for '{}'",
                                layer.name
                            ),
                        ));
                    }
                }
            }
            if let Some(w) = &sc.window {
                errors.extend(validate_window_spec(w, &format!("{base}.window"), &[]));
            }
        }

        // --- Controls ---
        let cbase = "controls.zone_control";
        let heating = self.controls.zone_control.heating_setpoint;
        let cooling = self.controls.zone_control.cooling_setpoint;
        if heating >= cooling {
            errors.push(ValidationError::new(
                format!("{cbase}.heating_setpoint / {cbase}.cooling_setpoint"),
                format!("heating_setpoint ({heating}) must be < cooling_setpoint ({cooling})"),
                "set heating_setpoint below cooling_setpoint (typical: 20.0 / 24.0 °C)",
            ));
        }
        for (prop, v) in [
            (
                "heating_capacity",
                self.controls.zone_control.heating_capacity,
            ),
            (
                "cooling_capacity",
                self.controls.zone_control.cooling_capacity,
            ),
        ] {
            if v <= 0.0 {
                errors.push(ValidationError::new(
                    format!("{cbase}.{prop}"),
                    format!("must be > 0, got {v}"),
                    format!("set {prop} to the system capacity in W"),
                ));
            }
        }

        // --- Internal loads (Issue #4101) ---
        if let Some(density) = self.schedules.lighting_power_density_w_m2 {
            if density < 0.0 {
                errors.push(ValidationError::new(
                    "schedules.lighting_power_density_w_m2",
                    format!("must be >= 0, got {density}"),
                    "set lighting_power_density_w_m2 to the lighting power density in W/m² (omit for no lighting)",
                ));
            }
            if density.is_nan() {
                errors.push(ValidationError::new(
                    "schedules.lighting_power_density_w_m2",
                    "must not be NaN".to_string(),
                    "set lighting_power_density_w_m2 to the lighting power density in W/m² (omit for no lighting)",
                ));
            }
        }
        for (i, spec) in self.schedules.equipment.iter().enumerate() {
            errors.extend(spec.validate(i));
        }

        // --- Weather ---
        if let WeatherData::EpwFile { path } = &self.weather {
            if path.as_os_str().is_empty() {
                errors.push(ValidationError::new(
                    "weather.path",
                    "epw file path must not be empty",
                    "set weather to {\"type\": \"epw\", \"path\": \"<path to .epw>\"}",
                ));
            }
        }

        errors
    }
}

impl Default for SimulationSchemaV1 {
    fn default() -> Self {
        SimulationSchemaV1 {
            version: SchemaVersion::V1,
            metadata: SchemaMetadata::default(),
            geometry: Geometry::default(),
            constructions: ConstructionSet::default(),
            schedules: ScheduleSet::default(),
            weather: WeatherData::default(),
            controls: ControlSet::default(),
            output: SimulationOutput::default(),
        }
    }
}

impl Default for SimulationSchema {
    fn default() -> Self {
        SimulationSchema::V1(SimulationSchemaV1::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unmet_hours_all_comfortable() {
        let occupancy = DailySchedule::weekly("occ".to_string()).office_hours();
        // 48 hours, 1 zone, always 21°C within the 20/24 band.
        let hourly = vec![vec![21.0; 48]];
        let (h, c) = SimulationOutput::unmet_hours(&hourly, &occupancy, 20.0, 24.0, 0.5);
        assert_eq!((h, c), (0.0, 0.0));
    }

    #[test]
    fn test_unmet_hours_counts_occupied_only() {
        let occupancy = DailySchedule::weekly("occ".to_string()).office_hours();
        // Zone is cold (15°C) for all 48 hours, but only office hours count.
        let hourly = vec![vec![15.0; 48]];
        let (h, c) = SimulationOutput::unmet_hours(&hourly, &occupancy, 20.0, 24.0, 0.5);
        // office_hours sets 8..=17 (10 hours/day) on weekdays; value() falls
        // back to Monday's profile, so 10 occupied hours/day × 2 days.
        assert_eq!(h, 20.0);
        assert_eq!(c, 0.0);
    }

    #[test]
    fn test_unmet_hours_cooling_and_tolerance() {
        let occupancy = DailySchedule::weekly("occ".to_string()).office_hours();
        // 30°C during occupied hours -> unmet cooling; 24.2°C is inside the
        // cooling setpoint + 0.5 tolerance band and must not count.
        let mut temps = vec![21.0; 48];
        for t in [9, 10, 33] {
            temps[t] = 30.0;
        }
        temps[11] = 24.2;
        let hourly = vec![temps];
        let (h, c) = SimulationOutput::unmet_hours(&hourly, &occupancy, 20.0, 24.0, 0.5);
        assert_eq!(h, 0.0);
        assert_eq!(c, 3.0);
    }

    #[test]
    fn test_unmet_hours_sums_across_zones() {
        let occupancy = DailySchedule::weekly("occ".to_string()).office_hours();
        let hourly = vec![vec![15.0; 48], vec![15.0; 48]];
        let (h, _) = SimulationOutput::unmet_hours(&hourly, &occupancy, 20.0, 24.0, 0.5);
        assert_eq!(h, 40.0);
    }

    #[test]
    fn test_unmet_hours_tolerance_defaults_to_0_2_c() {
        // Issue #4103: an unset schema deadband defaults to the EnergyPlus
        // "Time Setpoint Not Met" 0.2 °C tolerance, not 0.5 °C.
        assert_eq!(
            ControlConfig::default().deadband_tolerance,
            UNMET_HOURS_TOLERANCE_DEFAULT_C
        );
        assert_eq!(UNMET_HOURS_TOLERANCE_DEFAULT_C, 0.2);
        // A schema JSON that omits deadband_tolerance must still
        // deserialize, picking up the 0.2 °C default.
        let cfg: ControlConfig = serde_json::from_str(
            r#"{"heating_setpoint":20.0,"cooling_setpoint":24.0,
                "heating_capacity":100000.0,"cooling_capacity":100000.0}"#,
        )
        .expect("unset deadband_tolerance must deserialize");
        assert_eq!(cfg.deadband_tolerance, 0.2);
        // An explicit value is honored unchanged.
        let cfg: ControlConfig = serde_json::from_str(
            r#"{"heating_setpoint":20.0,"cooling_setpoint":24.0,
                "deadband_tolerance":0.5,
                "heating_capacity":100000.0,"cooling_capacity":100000.0}"#,
        )
        .expect("explicit deadband_tolerance must deserialize");
        assert_eq!(cfg.deadband_tolerance, 0.5);
    }

    #[test]
    fn test_unmet_hours_all_hours_diverges_on_unoccupied_night() {
        // Issue #4103: a deviation during unoccupied night hours counts in
        // the all-hours variant (ASHRAE 90.1 G3.1.2.2) but not in the
        // occupied-only variant.
        let occupancy = DailySchedule::weekly("occ".to_string()).office_hours();
        // 24 h trace: comfortable 21 °C all day except midnight (t=0),
        // which is unoccupied and drops to 15 °C.
        let mut temps = vec![21.0; 24];
        temps[0] = 15.0;
        let hourly = vec![temps];
        let tol = UNMET_HOURS_TOLERANCE_DEFAULT_C;
        let (occ_h, occ_c) = SimulationOutput::unmet_hours(&hourly, &occupancy, 20.0, 24.0, tol);
        let (all_h, all_c) = SimulationOutput::unmet_hours_all_hours(&hourly, 20.0, 24.0, tol);
        assert_eq!((occ_h, occ_c), (0.0, 0.0));
        assert_eq!((all_h, all_c), (1.0, 0.0));
    }

    #[test]
    fn test_unmet_hours_all_hours_sums_across_zones() {
        // Issue #4103: the all-hours variant also sums across zones, with
        // the same heating/cooling deviation test.
        let hourly = vec![vec![15.0; 24], vec![30.0; 24]];
        let tol = UNMET_HOURS_TOLERANCE_DEFAULT_C;
        let (h, c) = SimulationOutput::unmet_hours_all_hours(&hourly, 20.0, 24.0, tol);
        assert_eq!(h, 24.0);
        assert_eq!(c, 24.0);
        // A fully comfortable trace reports zero in both conventions.
        let comfortable = vec![vec![21.0; 24]];
        assert_eq!(
            SimulationOutput::unmet_hours_all_hours(&comfortable, 20.0, 24.0, tol),
            (0.0, 0.0)
        );
    }

    #[test]
    fn test_validate_default_schema_is_valid() {
        // The default schema must pass validation — it is the baseline all
        // other tests and examples build from.
        let schema = SimulationSchemaV1::default();
        assert!(schema.validate().is_empty());
    }

    #[test]
    fn test_validate_catches_bad_zone_geometry() {
        let mut schema = SimulationSchemaV1::default();
        schema.geometry.zones[0].floor_area = -5.0;
        schema.geometry.zones[0].height = 0.0;
        let errors = schema.validate();
        let fields: Vec<&str> = errors.iter().map(|e| e.field.as_str()).collect();
        assert!(fields.contains(&"geometry.zones[0].floor_area"));
        assert!(fields.contains(&"geometry.zones[0].height"));
        // Every error must name the problem and a fix.
        for e in &errors {
            assert!(!e.problem.is_empty(), "missing problem for {}", e.field);
            assert!(!e.fix.is_empty(), "missing fix for {}", e.field);
        }
    }

    #[test]
    fn test_validate_catches_empty_zones_and_bad_setpoints() {
        let mut schema = SimulationSchemaV1::default();
        schema.geometry.zones.clear();
        schema.controls.zone_control.heating_setpoint = 26.0;
        schema.controls.zone_control.cooling_setpoint = 24.0;
        let errors = schema.validate();
        let fields: Vec<&str> = errors.iter().map(|e| e.field.as_str()).collect();
        assert!(fields.contains(&"geometry.zones"));
        assert!(fields.iter().any(|f| f.contains("heating_setpoint")));
    }

    #[test]
    fn test_validate_catches_bad_construction_layer() {
        let mut schema = SimulationSchemaV1::default();
        schema.constructions.wall.layers = vec![crate::sim::construction::ConstructionLayer {
            name: "bad".to_string(),
            conductivity: 0.04,
            density: 12.0,
            specific_heat: 840.0,
            thickness: 0.0,
            absorptance: 0.7,
            emissivity: 0.9,
        }];
        let errors = schema.validate();
        assert!(errors
            .iter()
            .any(|e| e.field == "constructions.wall.layers[0].thickness"));
    }

    #[test]
    fn test_validate_catches_bad_window_shgc() {
        let mut schema = SimulationSchemaV1::default();
        if let Some(w) = &mut schema.constructions.wall.window {
            w.window_shgc = 1.5;
        }
        let errors = schema.validate();
        assert!(errors
            .iter()
            .any(|e| e.field == "constructions.wall.window.window_shgc"));
    }

    #[test]
    fn test_schema_version_default() {
        let version = SchemaVersion::default();
        assert_eq!(version, SchemaVersion::V1);
    }

    #[test]
    fn test_geometry_default() {
        let geometry = Geometry::default();
        assert_eq!(geometry.zones.len(), 1);
        assert_eq!(geometry.total_floor_area, 48.0);
    }

    #[test]
    fn test_construction_set_default() {
        let construction = SurfaceConstruction::default();
        assert_eq!(construction.layers.len(), 3);
        assert!(construction.window.is_some());
    }

    #[test]
    fn test_weather_data_default() {
        let weather = WeatherData::default();
        match weather {
            WeatherData::TmyLocation { location } => {
                assert_eq!(location, "Denver, CO");
            }
            _ => panic!("Expected TmyLocation variant"),
        }
    }

    #[test]
    fn test_control_config_default() {
        let control = ControlConfig::default();
        assert_eq!(control.heating_setpoint, 20.0);
        assert_eq!(control.cooling_setpoint, 24.0);
    }

    #[test]
    fn test_simulation_schema_v1_default() {
        let schema = SimulationSchemaV1::default();
        assert_eq!(schema.version, SchemaVersion::V1);
        assert_eq!(schema.geometry.zones.len(), 1);
    }

    #[test]
    fn test_simulation_schema_default() {
        let schema = SimulationSchema::default();
        assert_eq!(schema.version(), SchemaVersion::V1);
    }

    #[test]
    fn test_schema_serialization() {
        let schema = SimulationSchema::V1(SimulationSchemaV1::default());
        let json = serde_json::to_string(&schema).unwrap();
        let deserialized: SimulationSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(schema, deserialized);
    }

    // ===== Issue #3989 — per-surface geometry / fenestration / HVAC =====

    #[test]
    fn test_per_surface_schema_roundtrip_3989() {
        let mut inner = SimulationSchemaV1::default();
        inner.geometry.surfaces = vec![
            SurfaceSpec {
                id: "south_wall".to_string(),
                zone_index: 0,
                kind: SurfaceKind::Wall,
                area: Some(21.6),
                orientation: Some(Orientation::South),
                tilt_deg: Some(90.0),
                vertices: None,
                window: Some(WindowSpec {
                    window_area: 12.0,
                    window_u_value: 3.0,
                    window_shgc: 0.79,
                    window_to_wall_ratio: None,
                    surface: Some("south_wall".to_string()),
                    orientation: Some(Orientation::South),
                }),
            },
            SurfaceSpec {
                id: "north_wall".to_string(),
                zone_index: 0,
                kind: SurfaceKind::Wall,
                area: Some(21.6),
                orientation: Some(Orientation::North),
                tilt_deg: Some(90.0),
                vertices: None,
                window: Some(WindowSpec {
                    window_area: 0.0,
                    window_u_value: 3.0,
                    window_shgc: 0.79,
                    window_to_wall_ratio: Some(0.2),
                    surface: Some("north_wall".to_string()),
                    orientation: Some(Orientation::North),
                }),
            },
        ];
        inner.controls.hvac_system = Some(HVACSystemType::Ideal);

        let schema = SimulationSchema::V1(inner);
        let json = serde_json::to_string(&schema).unwrap();
        let deserialized: SimulationSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(schema, deserialized);

        let SimulationSchema::V1(roundtripped) = deserialized;
        let south = roundtripped.geometry.surface("south_wall").unwrap();
        assert_eq!(south.window_area(), 12.0);
        let north = roundtripped.geometry.surface("north_wall").unwrap();
        // North glazing is expressed as a WWR only: 0.2 x 21.6 m2 = 4.32 m2.
        assert!((north.window_area() - 4.32).abs() < 1e-9);
        assert_eq!(
            roundtripped
                .geometry
                .surfaces_with_orientation(Orientation::South)
                .count(),
            1
        );
        assert_eq!(
            roundtripped.controls.hvac_system,
            Some(HVACSystemType::Ideal)
        );
        let surface_errors: Vec<_> = roundtripped
            .validate()
            .into_iter()
            .filter(|e| e.field.contains("surfaces") || e.field == "version")
            .collect();
        assert!(
            surface_errors.is_empty(),
            "unexpected per-surface validation errors: {surface_errors:?}"
        );
    }

    // ===== Issue #3997 — named HVAC system templates =====

    /// The named templates selectable through `controls.hvac_system`,
    /// together with the JSON name each one serialises to.
    const NAMED_HVAC_TEMPLATES: [(&str, HVACSystemType); 6] = [
        ("Simple", HVACSystemType::Simple),
        ("VAV", HVACSystemType::VAV),
        ("CAV", HVACSystemType::CAV),
        ("HeatPump", HVACSystemType::HeatPump),
        ("DOASFanCoil", HVACSystemType::DOASFanCoil),
        ("Ideal", HVACSystemType::Ideal),
    ];

    #[test]
    fn test_named_hvac_templates_deserialise_and_round_trip_3997() {
        // Every named template must parse from its documented JSON name
        // and re-serialise to exactly that name.
        for (name, expected) in NAMED_HVAC_TEMPLATES {
            let json = format!(
                r#"{{"zone_control":{{"heating_setpoint":20.0,"cooling_setpoint":24.0,"heating_capacity":100000.0,"cooling_capacity":100000.0}},"global_control":null,"hvac_system":"{name}"}}"#
            );
            let controls: ControlSet = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("template {name} failed to deserialise: {e}"));
            assert_eq!(
                controls.hvac_system,
                Some(expected),
                "template {name} did not deserialise to the expected variant"
            );

            let serialised = serde_json::to_string(&controls).unwrap();
            let value: serde_json::Value = serde_json::from_str(&serialised).unwrap();
            assert_eq!(
                value["hvac_system"],
                serde_json::json!(name),
                "template {name} did not re-serialise under its own name"
            );

            let reparsed: ControlSet = serde_json::from_str(&serialised).unwrap();
            assert_eq!(
                reparsed, controls,
                "template {name} lost data on round-trip"
            );
        }
    }

    #[test]
    fn test_named_hvac_templates_validate_in_full_schema_3997() {
        // A schema that names a template must still validate (no control
        // errors) and survive a full-document round-trip.
        for (name, expected) in NAMED_HVAC_TEMPLATES {
            let mut inner = SimulationSchemaV1::default();
            inner.controls.hvac_system = Some(expected);
            let schema = SimulationSchema::V1(inner);
            let json = serde_json::to_string(&schema).unwrap();
            assert!(
                json.contains(&format!("\"hvac_system\":\"{name}\"")),
                "template {name} missing from the serialised document: {json}"
            );

            let reparsed: SimulationSchema = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("template {name} broke the document: {e}"));
            assert_eq!(schema, reparsed, "template {name} changed the document");

            let SimulationSchema::V1(v1) = reparsed;
            assert_eq!(v1.controls.hvac_system, Some(expected));
            let control_errors: Vec<_> = v1
                .validate()
                .into_iter()
                .filter(|e| e.field.starts_with("controls"))
                .collect();
            assert!(
                control_errors.is_empty(),
                "template {name} produced control validation errors: {control_errors:?}"
            );
        }
    }

    #[test]
    fn test_unknown_hvac_template_name_is_rejected_3997() {
        // The template vocabulary is the engine's own enum: an unknown
        // (or differently spelled) name must fail loudly rather than
        // silently falling back to a default.
        for bad in ["DOAS_FanCoil", "DoasFanCoil", "doas+fan-coil", "DOAS"] {
            let json = format!(
                r#"{{"zone_control":{{"heating_setpoint":20.0,"cooling_setpoint":24.0,"heating_capacity":100000.0,"cooling_capacity":100000.0}},"global_control":null,"hvac_system":"{bad}"}}"#
            );
            assert!(
                serde_json::from_str::<ControlSet>(&json).is_err(),
                "unknown template name {bad:?} was accepted"
            );
        }
    }

    #[test]
    fn test_pre3989_v1_shapes_still_deserialise() {
        // Pre-#3989 aggregate shapes carry none of the new fields; the
        // serde defaults must fill them with the old behaviour.
        let geometry: Geometry = serde_json::from_str(
            r#"{"zones":[{"name":"Zone 1","floor_area":48.0,"volume":129.6,"height":2.7}],"total_floor_area":48.0,"total_volume":129.6,"number_of_floors":1,"floor_height":2.7}"#,
        )
        .unwrap();
        assert!(geometry.surfaces.is_empty());

        let controls: ControlSet = serde_json::from_str(
            r#"{"zone_control":{"heating_setpoint":20.0,"cooling_setpoint":24.0,"deadband_tolerance":0.5,"heating_capacity":100000.0,"cooling_capacity":100000.0},"global_control":null}"#,
        )
        .unwrap();
        assert_eq!(controls.hvac_system, None);

        let window: WindowSpec =
            serde_json::from_str(r#"{"window_area":12.0,"window_u_value":1.5,"window_shgc":0.3}"#)
                .unwrap();
        assert_eq!(window.window_to_wall_ratio, None);
        assert_eq!(window.surface, None);

        // The checked-in pre-#3989 example document still loads whole.
        let raw = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/minimal.flux"
        ))
        .unwrap();
        let schema: SimulationSchema = serde_json::from_str(&raw).unwrap();
        let SimulationSchema::V1(inner) = schema;
        assert!(inner.geometry.surfaces.is_empty());
        assert_eq!(inner.controls.hvac_system, None);
    }

    #[test]
    fn test_example_case600_per_surface_flux_loads_3989() {
        let raw = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/case600_per_surface.flux"
        ))
        .unwrap();
        let schema: SimulationSchema = serde_json::from_str(&raw).unwrap();
        let SimulationSchema::V1(inner) = schema;
        assert_eq!(inner.geometry.surfaces.len(), 6);
        let south = inner.geometry.surface("south_wall").unwrap();
        assert_eq!(south.window_area(), 12.0);
        // The south wall also carries a vertex loop: 8 m x 2.7 m = 21.6 m2.
        assert!((south.polygon_area().unwrap() - 21.6).abs() < 1e-9);
        let north = inner.geometry.surface("north_wall").unwrap();
        assert!((north.window_area() - 4.32).abs() < 1e-9);
        assert_eq!(inner.controls.hvac_system, Some(HVACSystemType::Ideal));
        let surface_errors: Vec<_> = inner
            .validate()
            .into_iter()
            .filter(|e| e.field.contains("surfaces") || e.field == "version")
            .collect();
        assert!(
            surface_errors.is_empty(),
            "example .flux has per-surface validation errors: {surface_errors:?}"
        );
    }

    #[test]
    fn test_zone_geometry_serialization() {
        let zone = ZoneGeometry::default();
        let json = serde_json::to_string(&zone).unwrap();
        let deserialized: ZoneGeometry = serde_json::from_str(&json).unwrap();
        assert_eq!(zone, deserialized);
    }

    #[test]
    fn test_construction_layer_serialization() {
        let layer = ConstructionLayer::new("Test", 0.04, 12.0, 840.0, 0.066);
        let json = serde_json::to_string(&layer).unwrap();
        let deserialized: ConstructionLayer = serde_json::from_str(&json).unwrap();
        assert_eq!(layer.name, deserialized.name);
        assert_eq!(layer.conductivity, deserialized.conductivity);
    }

    #[test]
    fn test_hvac_schedule_serialization() {
        let schedule = HVACSchedule::constant_schedule(20.0, 24.0).unwrap();
        let json = serde_json::to_string(&schedule).unwrap();
        let deserialized: HVACSchedule = serde_json::from_str(&json).unwrap();
        assert_eq!(
            schedule.heating_setpoint(0),
            deserialized.heating_setpoint(0)
        );
    }

    #[test]
    fn test_simulation_output_serialization() {
        let output = SimulationOutput::default();
        let json = serde_json::to_string(&output).unwrap();
        let deserialized: SimulationOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(output.eui, deserialized.eui);
    }

    #[test]
    fn test_schema_metadata_with_author() {
        let metadata = SchemaMetadata {
            name: "Test Schema".to_string(),
            description: "A test schema".to_string(),
            author: Some("Test Author".to_string()),
            created_at: Some("2026-04-17".to_string()),
            schema_version: SchemaVersion::V1,
        };
        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: SchemaMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(metadata.author, deserialized.author);
    }

    // ===== Issue #2915 — EpwFile path validation on deserialize =====
    //
    // `WeatherData::EpwFile { path }` is the inbound payload of every
    // `/v1/simulate` and `/v1/campaign/*` request; a missing validation
    // gate would let an authenticated REST client reach
    // `EpwWeatherSource::from_file` with `std::fs::File::open` — i.e.
    // arbitrary server-readable file read (CWE-22). These tests assert
    // that `serde_json::from_str` refuses `/etc/passwd` and round-trips a
    // real `.epw` file inside the `FLUXION_EPW_DIR` allow-list.
    //
    // `FLUXION_EPW_DIR` is the process-wide env var read by
    // `validate_epw_path`; a `Mutex` serialises every test that mutates
    // it so parallel `cargo test` threads cannot stomp on each other.

    /// Shared mutex serialising every test in this module that mutates
    /// `FLUXION_EPW_DIR`. Without it, parallel `cargo test` threads would
    /// race on the env var and produce flaky failures.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Helper: set `FLUXION_EPW_DIR` to the given path for the duration
    /// of the closure. Returns whatever the closure returns. Any prior
    /// value of `FLUXION_EPW_DIR` is restored on scope exit (success or
    /// panic) so tests cannot leak env state into siblings.
    fn with_epw_dir<F: FnOnce() -> R, R>(dir: &std::path::Path, f: F) -> R {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let previous = std::env::var("FLUXION_EPW_DIR").ok();
        std::env::set_var("FLUXION_EPW_DIR", dir);
        let result = f();
        match previous {
            Some(prev) => std::env::set_var("FLUXION_EPW_DIR", prev),
            None => std::env::remove_var("FLUXION_EPW_DIR"),
        }
        result
    }

    /// Inbound `/etc/passwd` (no `.epw` extension) is rejected by the
    /// deserializer before it ever reaches `EpwWeatherSource::from_file`.
    /// This is the canonical path-traversal probe from Issue #2915.
    #[test]
    fn deserialize_epw_file_rejects_etc_passwd() {
        if !std::path::Path::new("/etc/passwd").is_file() {
            eprintln!("skipping: /etc/passwd not present on this platform");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{"type":"epw","path":"/etc/passwd"}"#;
        let result = with_epw_dir(dir.path(), || serde_json::from_str::<WeatherData>(json));
        assert!(
            result.is_err(),
            "/etc/passwd must be rejected at deserialize time: {result:?}"
        );
        // Generic message — the raw user-supplied path must not be
        // reflected back through the deserializer error chain.
        let err = result.unwrap_err().to_string();
        assert!(!err.contains("passwd"), "error must not echo path: {err}");
    }

    /// A `.epw` file inside the `FLUXION_EPW_DIR` allow-list serializes
    /// and round-trips through the deserializer without error.
    #[test]
    fn deserialize_epw_file_round_trips_inside_allowlist() {
        let dir = tempfile::tempdir().unwrap();
        let epw = dir.path().join("USA_CO_Denver.epw");
        std::fs::write(&epw, b"LOCATION,Denver,CO\n").unwrap();

        let original = WeatherData::EpwFile { path: epw.clone() };
        let json = with_epw_dir(dir.path(), || serde_json::to_string(&original).unwrap());
        let deserialized: WeatherData =
            with_epw_dir(dir.path(), || serde_json::from_str(&json).unwrap());
        assert_eq!(original, deserialized);
    }

    /// A `.epw` file that lives outside the allow-list is rejected on
    /// deserialize, even when the path itself is well-formed.
    #[test]
    fn deserialize_epw_file_rejects_traversal_outside_allowlist() {
        let allowed = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let evil = outside.path().join("evil.epw");
        std::fs::write(&evil, b"pwned").unwrap();
        let json = serde_json::json!({
            "type": "epw",
            "path": evil.to_string_lossy().into_owned(),
        })
        .to_string();
        let result = with_epw_dir(allowed.path(), || {
            serde_json::from_str::<WeatherData>(&json)
        });
        assert!(
            result.is_err(),
            "out-of-allowlist epw must be rejected: {result:?}"
        );
    }

    /// The deserializer refuses an `.epw` path that contains `..`
    /// traversal reaching outside the allow-list (defence in depth on
    /// top of `validate_epw_path`'s canonicalize + `starts_with` check).
    #[test]
    fn deserialize_epw_file_rejects_dotdot_traversal() {
        let allowed = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let real = outside.path().join("secret.epw");
        std::fs::write(&real, b"x").unwrap();
        // Build a traversal path: from inside the allowed dir, climb out
        // via `..` and into the outside dir's basename.
        let traversal = allowed
            .path()
            .join("..")
            .join(outside.path().file_name().unwrap())
            .join("secret.epw")
            .to_string_lossy()
            .into_owned();
        let json = serde_json::json!({
            "type": "epw",
            "path": traversal,
        })
        .to_string();
        let result = with_epw_dir(allowed.path(), || {
            serde_json::from_str::<WeatherData>(&json)
        });
        assert!(result.is_err(), "dotdot traversal must be rejected");
    }

    /// `TmyLocation` and `Inline` variants are unaffected by the new
    /// `EpwFile` deserializer gate (regression guard for the tag-based
    /// enum dispatch).
    #[test]
    fn deserialize_non_epw_variants_unaffected() {
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{"type":"tmy","location":"Denver, CO"}"#;
        let result: WeatherData = with_epw_dir(dir.path(), || serde_json::from_str(json).unwrap());
        assert_eq!(
            result,
            WeatherData::TmyLocation {
                location: "Denver, CO".to_string(),
            }
        );
    }
}
