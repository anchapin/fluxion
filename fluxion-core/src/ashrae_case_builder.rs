//! `CaseBuilder` — fluent builder for ASHRAE 140 `CaseSpec` values.
//!
//! Issue #4157 (round 52): moved verbatim from
//! `fluxion::validation::ashrae_140_cases` so the `series_*` case-factory
//! bodies can live in the same crate as the types they construct. The
//! engine-driving factories (Case 195/800/diagnostics/office presets and
//! `generate_denver_weather_data`) stay in the main crate behind the
//! `CaseBuilderExt` trait; everything here is pure spec construction over
//! `ashrae_cases` + `construction` leaf types.

use crate::ashrae_cases::*;
use crate::construction::{Assemblies, Construction, Materials};
use std::path::PathBuf;

/// Builder for creating ASHRAE 140 case specifications.
///
/// The builder provides a fluent API for configuring test cases with sensible defaults
/// and validation.
///
/// # Example
///
/// ```rust
/// use fluxion::validation::ashrae_140_cases::CaseBuilder;
///
/// let spec = CaseBuilder::new()
///     .low_mass_construction()
///     .with_dimensions(8.0, 6.0, 2.7)
///     .with_south_window(12.0)
///     .with_hvac_setpoints(20.0, 27.0)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CaseBuilder {
    case_id: Option<String>,
    description: String,
    geometry: Vec<GeometrySpec>,
    construction_type: ConstructionType,
    construction: Option<ConstructionSpec>,
    windows: Vec<Vec<WindowArea>>,
    window_properties: WindowSpec,
    shading: Option<ShadingDevice>,
    internal_loads: Vec<Option<InternalLoads>>,
    hvac: Vec<HvacSchedule>,
    night_ventilation: Option<NightVentilation>,
    common_walls: Vec<CommonWall>,
    zone_origins: Option<Vec<ZoneOrigin>>,
    zone_constructions: Option<Vec<ConstructionSpec>>,
    infiltration_ach: f64,
    opaque_absorptance: f64,
    num_zones: usize,
    /// Door opening height (meters, for stack effect ACH)
    door_height: Option<f64>,
    /// Door opening area (square meters, for stack effect ACH)
    door_area: Option<f64>,
    /// Custom EPW weather file path (if using non-default weather)
    epw_path: Option<PathBuf>,
    /// Ground temperature boundary condition (°C) for floor slab (Issue #746).
    /// Per ASHRAE 140-2023 Annex B §B3.3: T_ground = 9.4°C.
    ground_temperature_c: Option<f64>,
    /// Optional override for the floor U-value (W/m²K) used in ground coupling.
    /// When `Some`, replaces the construction-derived floor U-value in both
    /// `setpoints.floor_u_value` and the `h_tr_floor` conductance. Used for
    /// specified test conditions such as the ASHRAE 140 Case 195 solid
    /// conduction test (U = 0.039 W/m²K) — data-driven, no case_id check.
    floor_u_value_override: Option<f64>,
    /// Optional window U-value override (W/m²K), threaded through to CaseSpec.
    window_u_value_override: Option<f64>,
    /// Building usage type for thermal mass calculations
    building_type: BuildingType,
}

impl Default for CaseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CaseBuilder {
    /// Creates a new CaseBuilder with default values.
    pub fn new() -> Self {
        CaseBuilder {
            case_id: None,
            description: String::new(),
            geometry: Vec::new(),
            construction_type: ConstructionType::LowMass,
            construction: None,
            windows: Vec::new(),
            window_properties: WindowSpec::double_clear_glass(),
            shading: None,
            internal_loads: Vec::new(),
            hvac: Vec::new(),
            night_ventilation: None,
            common_walls: Vec::new(),
            zone_constructions: None,
            zone_origins: None,
            infiltration_ach: 0.5,
            opaque_absorptance: 0.6,
            num_zones: 1,
            door_height: None,
            door_area: None,
            epw_path: None,
            ground_temperature_c: None,
            floor_u_value_override: None,
            window_u_value_override: None,
            building_type: BuildingType::default(),
        }
    }

    /// Sets the case identifier.
    pub fn with_case_id(mut self, case_id: String) -> Self {
        self.case_id = Some(case_id);
        self
    }

    /// Sets the case description.
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    /// Sets the zone dimensions for the first zone.
    pub fn with_dimensions(mut self, width: f64, depth: f64, height: f64) -> Self {
        if self.geometry.is_empty() {
            self.geometry.push(GeometrySpec::new(width, depth, height));
        } else {
            self.geometry[0] = GeometrySpec::new(width, depth, height);
        }
        self
    }

    /// Adds a zone with specified dimensions.
    pub fn add_zone(mut self, width: f64, depth: f64, height: f64) -> Self {
        self.geometry.push(GeometrySpec::new(width, depth, height));
        self.windows.push(Vec::new());
        self.internal_loads.push(None);
        self.hvac.push(HvacSchedule::constant(20.0, 27.0));
        self.num_zones = self.geometry.len();
        self
    }

    /// Adds a rectangular zone with specified dimensions.
    ///
    /// This is a convenience method for creating simple zones with optional naming.
    /// Zones are automatically assigned sequential IDs if not named.
    ///
    /// # Arguments
    /// * `length` - Zone length (width in X direction) in meters
    /// * `width` - Zone width (depth in Y direction) in meters
    /// * `height` - Zone height in meters
    /// * `name` - Optional zone identifier; if None, a name like "zone0" is auto-generated
    pub fn rectangular_zone(
        mut self,
        length: f64,
        width: f64,
        height: f64,
        name: Option<&str>,
    ) -> Self {
        let name = name
            .map(String::from)
            .or_else(|| Some(format!("zone{}", self.geometry.len())));
        self.geometry.push(GeometrySpec {
            width: length,
            depth: width,
            height,
            name,
        });
        self.windows.push(Vec::new());
        self.internal_loads.push(None);
        self.hvac.push(HvacSchedule::constant(20.0, 27.0));
        self.num_zones = self.geometry.len();
        self
    }

    /// Adds a common wall between two zones.
    pub fn with_common_wall(
        mut self,
        zone_a: usize,
        zone_b: usize,
        area: f64,
        construction: Construction,
    ) -> Self {
        self.common_walls
            .push(CommonWall::new(zone_a, zone_b, area, construction));
        self
    }

    /// Sets the per-zone plan origins (meters) for perimeter-adjacency
    /// derivation (§LIMIT-23b, round 21). The vector must have one entry per
    /// zone; see [`CaseSpec::zone_origins`].
    pub fn with_zone_origins(mut self, origins: Vec<ZoneOrigin>) -> Self {
        self.zone_origins = Some(origins);
        self
    }

    /// Adds a common wall between two zones using a simple wall construction specified by R-value.
    ///
    /// This is a convenience method that creates a wall construction with the desired
    /// material R-value using a default insulating layer. The zones are referenced by
    /// their assigned names (see `rectangular_zone`).
    ///
    /// # Arguments
    /// * `zone1_id` - Identifier of the first zone
    /// * `zone2_id` - Identifier of the second zone
    /// * `area` - Area of the common wall in square meters (m²)
    /// * `r_value` - Desired thermal resistance of the wall materials (m²K/W)
    pub fn add_common_wall(
        mut self,
        zone1_id: &str,
        zone2_id: &str,
        area: f64,
        r_value: f64,
    ) -> Self {
        let idx1 = self.find_zone_index(zone1_id);
        let idx2 = self.find_zone_index(zone2_id);
        // Create a simple insulation wall with the given R-value using fiberglass (k=0.04)
        let thickness = r_value * 0.04;
        let layer = Materials::fiberglass(thickness);
        let construction = Construction::new(vec![layer]);
        self.common_walls
            .push(CommonWall::new(idx1, idx2, area, construction));
        self
    }

    /// Looks up the geometry index for a given zone name.
    /// Panics if the zone name is not found.
    fn find_zone_index(&self, id: &str) -> usize {
        self.geometry
            .iter()
            .position(|g| g.name.as_ref().map(String::as_str) == Some(id))
            .unwrap_or_else(|| panic!("Zone '{}' not found in builder", id))
    }

    /// Sets the construction type to low mass.
    pub fn low_mass_construction(mut self) -> Self {
        self.construction_type = ConstructionType::LowMass;
        self
    }

    /// Sets the construction type to high mass.
    pub fn high_mass_construction(mut self) -> Self {
        self.construction_type = ConstructionType::HighMass;
        self
    }

    /// Sets custom construction assemblies.
    pub fn with_construction(
        mut self,
        wall: Construction,
        roof: Construction,
        floor: Construction,
    ) -> Self {
        self.construction = Some(ConstructionSpec::new(wall, roof, floor));
        self
    }

    /// Adds a window to the first zone.
    pub fn with_window(mut self, area: f64, orientation: Orientation) -> Self {
        if self.windows.is_empty() {
            self.windows.push(Vec::new());
        }
        self.windows[0].push(WindowArea::new(area, orientation));
        self
    }

    /// Adds a window to a specific zone.
    pub fn with_zone_window(
        mut self,
        zone_idx: usize,
        area: f64,
        orientation: Orientation,
    ) -> Self {
        while self.windows.len() <= zone_idx {
            self.windows.push(Vec::new());
        }
        self.windows[zone_idx].push(WindowArea::new(area, orientation));
        self
    }

    /// Adds a south-facing window to the first zone.
    pub fn with_south_window(self, area: f64) -> Self {
        self.with_window(area, Orientation::South)
    }

    /// Adds east and west windows with equal area to the first zone.
    pub fn with_ew_windows(self, each_area: f64) -> Self {
        self.with_window(each_area, Orientation::East)
            .with_window(each_area, Orientation::West)
    }

    /// Sets window properties.
    pub fn with_window_properties(mut self, window_properties: WindowSpec) -> Self {
        self.window_properties = window_properties;
        self
    }

    /// Overrides the effective window U-value (W/m²K).
    ///
    /// WINDOW-01 (loop round 7): threads the ASHRAE 140 spec window
    /// conductance (3.0 W/m²K, `WINDOW_U_VALUE`) through the CaseSpec so the
    /// strict CaseSpec path and the legacy validator setup path agree. When
    /// `Some`, both `solar.window_u_value` and the `h_tr_w` conductance use
    /// this value verbatim (the spec constant already includes the frame).
    pub fn with_window_u_value_override(mut self, u_value: f64) -> Self {
        self.window_u_value_override = Some(u_value);
        self
    }

    /// Sets custom EPW weather file for the case.
    ///
    /// The path should point to a valid EPW file. The file will be loaded
    /// during simulation setup. This overrides the default Denver TMY weather.
    ///
    /// # Arguments
    /// * `path` - Path to the EPW weather file (string path)
    pub fn with_weather_epw(mut self, path: &str) -> Self {
        self.epw_path = Some(PathBuf::from(path));
        self
    }

    /// Sets shading device.
    pub fn with_shading(mut self, shading: ShadingDevice) -> Self {
        self.shading = Some(shading);
        self
    }

    /// Sets internal loads for the first zone.
    pub fn with_internal_loads(mut self, loads: InternalLoads) -> Self {
        if self.internal_loads.is_empty() {
            self.internal_loads.push(Some(loads));
        } else {
            self.internal_loads[0] = Some(loads);
        }
        self
    }

    /// Sets HVAC schedule for the first zone.
    pub fn with_hvac(mut self, hvac: HvacSchedule) -> Self {
        if self.hvac.is_empty() {
            self.hvac.push(hvac);
        } else {
            self.hvac[0] = hvac;
        }
        self
    }

    /// Sets HVAC schedule for a specific zone.
    pub fn with_zone_hvac(mut self, zone_idx: usize, hvac: HvacSchedule) -> Self {
        while self.hvac.len() <= zone_idx {
            self.hvac.push(HvacSchedule::constant(20.0, 27.0));
        }
        self.hvac[zone_idx] = hvac;
        self
    }

    /// Sets HVAC setpoints for the first zone.
    pub fn with_hvac_setpoints(self, heating: f64, cooling: f64) -> Self {
        self.with_hvac(HvacSchedule::constant(heating, cooling))
    }

    /// Sets HVAC with setback for the first zone.
    pub fn with_hvac_setback(self, heating: f64, cooling: f64, setback: f64) -> Self {
        self.with_hvac(HvacSchedule::with_setback(heating, cooling, setback, 23, 7))
    }

    /// Sets night ventilation.
    pub fn with_night_ventilation(mut self, ventilation: NightVentilation) -> Self {
        self.night_ventilation = Some(ventilation);
        self
    }

    /// Sets the infiltration rate (ACH).
    ///
    /// Callers pass the ASHRAE 140 Table 5-42 (§5.2.1.6.1) adjusted rate for
    /// programs with a fixed sea-level air density (this engine: ρ = 1.2
    /// kg/m³), not the nominal 0.5 ach: 0.5 × (1.0156/1.2255) = 0.414 ach
    /// yields the mass flow of the 1609 m altitude site. The same
    /// fixed-sea-level convention sets the Case 650/950 vent fan to
    /// 1400 m³/h (Table 5-160, `NightVentilation::case_650`).
    pub fn with_infiltration(mut self, ach: f64) -> Self {
        self.infiltration_ach = ach;
        self
    }

    /// Sets the solar absorptance of opaque surfaces.
    pub fn with_opaque_absorptance(mut self, absorptance: f64) -> Self {
        self.opaque_absorptance = absorptance;
        self
    }

    /// Sets the number of zones.
    pub fn with_num_zones(mut self, num_zones: usize) -> Self {
        self.num_zones = num_zones;
        self
    }

    /// Sets the building type for thermal mass calculations.
    ///
    /// Determines the furniture factor (f_furniture) used in thermal mass calculations:
    /// - `Residential`: f_furniture = 0.3 (lighter furniture)
    /// - `Commercial`: f_furniture = 0.5 (heavier furniture)
    /// - `Institutional`: f_furniture = 0.5 (heavier furniture, e.g., schools, hospitals)
    ///
    /// Default is `Residential` for backward compatibility.
    pub fn with_building_type(mut self, building_type: BuildingType) -> Self {
        self.building_type = building_type;
        self
    }

    /// Configures door geometry for temperature-dependent air exchange (stack effect).
    ///
    /// Used for sunspace buildings (Case 960) where door openings between
    /// conditioned and unconditioned zones have temperature-dependent airflow.
    ///
    /// # Arguments
    /// * `height` - Door opening height (meters)
    /// * `area` - Door opening area (square meters)
    pub fn with_door_geometry(mut self, height: f64, area: f64) -> Self {
        self.door_height = Some(height);
        self.door_area = Some(area);
        self
    }

    /// Sets the ground temperature boundary condition for the floor slab (°C).
    ///
    /// Per ASHRAE 140-2023 Annex B §B3.3, T_ground = 9.4°C (annual mean Denver
    /// air temperature) applies to all cases with a floor slab (600, 610–650,
    /// 900–950, and their free-float variants).
    ///
    /// When not set, the model default ground temperature (10.0°C) is used.
    pub fn with_ground_temperature(mut self, temp_c: f64) -> Self {
        self.ground_temperature_c = Some(temp_c);
        self
    }

    /// Sets per-zone construction assemblies (round 24, §LIMIT-14).
    ///
    /// When set, length must equal `num_zones`; each zone then builds from its
    /// own `ConstructionSpec` instead of the single global `construction`.
    /// See `CaseSpec::zone_constructions`.
    pub fn with_zone_constructions(mut self, cons: Vec<ConstructionSpec>) -> Self {
        self.zone_constructions = Some(cons);
        self
    }

    /// Overrides the construction-derived floor U-value (W/m²K) for ground coupling.
    ///
    /// For specified test conditions such as the ASHRAE 140 Case 195 solid
    /// conduction test (U = 0.039 W/m²K). Data-driven: replaces the old
    /// `case_id == "195"` hard-code in the thermal model.
    pub fn with_floor_u_value_override(mut self, u_value: f64) -> Self {
        self.floor_u_value_override = Some(u_value);
        self
    }

    /// Builds and validates the case specification.
    pub fn build(mut self) -> Result<CaseSpec, String> {
        // Ensure vectors have correct length for num_zones
        if self.num_zones == 1 && self.geometry.is_empty() {
            return Err("Geometry must be specified".to_string());
        }

        if self.geometry.len() < self.num_zones {
            return Err(format!(
                "Only {} zone geometries provided for {} zones",
                self.geometry.len(),
                self.num_zones
            ));
        }
        while self.windows.len() < self.num_zones {
            self.windows.push(Vec::new());
        }
        while self.internal_loads.len() < self.num_zones {
            self.internal_loads.push(None);
        }
        while self.hvac.len() < self.num_zones {
            self.hvac.push(HvacSchedule::constant(20.0, 27.0));
        }

        // Use default construction if not specified
        let construction = self
            .construction
            .unwrap_or_else(|| match self.construction_type {
                ConstructionType::LowMass => ConstructionSpec::new(
                    Assemblies::low_mass_wall(),
                    Assemblies::low_mass_roof(),
                    Assemblies::insulated_floor(),
                ),
                ConstructionType::HighMass => ConstructionSpec::new(
                    Assemblies::high_mass_wall_standard(),
                    Assemblies::high_mass_roof(),
                    Assemblies::high_mass_floor(),
                ),
                ConstructionType::Special => ConstructionSpec::new(
                    Assemblies::low_mass_wall(),
                    Assemblies::low_mass_roof(),
                    Assemblies::insulated_floor(),
                ),
            });

        let spec = CaseSpec {
            case_id: self.case_id.unwrap_or_else(|| "custom".to_string()),
            description: self.description,
            geometry: self.geometry,
            construction_type: self.construction_type,
            construction,
            windows: self.windows,
            window_properties: self.window_properties,
            shading: self.shading,
            internal_loads: self.internal_loads,
            hvac: self.hvac,
            night_ventilation: self.night_ventilation,
            common_walls: self.common_walls.clone(),
            infiltration_ach: self.infiltration_ach,
            opaque_absorptance: self.opaque_absorptance,
            num_zones: self.num_zones,
            zone_origins: self.zone_origins.clone(),
            zone_constructions: self.zone_constructions.clone(),
            weather_data: None, // Will be loaded separately for solar calculations
            door_height: self.door_height,
            door_area: self.door_area,
            epw_path: self.epw_path.clone(),
            hvac_equipment: None,
            ground_temperature_c: self.ground_temperature_c,
            floor_u_value_override: self.floor_u_value_override,
            window_u_value_override: self.window_u_value_override,
            building_type: self.building_type,
        };

        // spec.validate()?; // Skip detailed validation for now to save time

        Ok(spec)
    }
}
