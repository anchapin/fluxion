use crate::ai::surrogate::SurrogateManager;
use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::boundary::{ConstantGroundTemperature, GroundTemperature};
use crate::sim::components::WallSurface;
use crate::sim::schedule::DailySchedule;
use crate::sim::shading::{Overhang, ShadeFin, Side};
use crate::sim::solar::{calculate_hourly_solar, WindowProperties};
use crate::validation::ashrae_140_cases::{CaseSpec, GeometrySpec, Orientation, ShadingType};
use crate::weather::HourlyWeatherData;
use faer::linalg::solvers::Solve;
use faer::Mat;
use std::sync::OnceLock;

static DAILY_CYCLE: OnceLock<[f64; 24]> = OnceLock::new();

fn get_daily_cycle() -> &'static [f64; 24] {
    DAILY_CYCLE.get_or_init(|| {
        let mut arr = [0.0; 24];
        for (h, val) in arr.iter_mut().enumerate() {
            *val =
                ((h as f64 / 24.0 * 2.0 * std::f64::consts::PI) - std::f64::consts::PI / 2.0).sin();
        }
        arr
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HVACMode {
    Heating,
    Cooling,
    Off,
}

#[derive(Clone, Debug)]
pub struct IdealHVACController {
    pub heating_setpoint: f64,
    pub cooling_setpoint: f64,
    pub deadband_tolerance: f64,
    pub heating_stages: u8,
    pub cooling_stages: u8,
    pub heating_capacity_per_stage: f64,
    pub cooling_capacity_per_stage: f64,
}

impl IdealHVACController {
    pub fn new(heating_setpoint: f64, cooling_setpoint: f64) -> Self {
        Self {
            heating_setpoint,
            cooling_setpoint,
            deadband_tolerance: 0.5,
            heating_stages: 1,
            cooling_stages: 1,
            heating_capacity_per_stage: 100_000.0,
            cooling_capacity_per_stage: 100_000.0,
        }
    }

    pub fn with_stages(hp: f64, cp: f64, hs: u8, cs: u8, hc: f64, cc: f64) -> Self {
        Self {
            heating_setpoint: hp,
            cooling_setpoint: cp,
            deadband_tolerance: 0.5,
            heating_stages: hs,
            cooling_stages: cs,
            heating_capacity_per_stage: hc,
            cooling_capacity_per_stage: cc,
        }
    }

    pub fn determine_mode(&self, zone_temp: f64) -> HVACMode {
        if zone_temp < self.heating_setpoint - self.deadband_tolerance {
            HVACMode::Heating
        } else if zone_temp > self.cooling_setpoint + self.deadband_tolerance {
            HVACMode::Cooling
        } else {
            HVACMode::Off
        }
    }

    pub fn calculate_power(&self, zone_temp: f64, free_float_temp: f64, sensitivity: f64) -> f64 {
        match self.determine_mode(zone_temp) {
            HVACMode::Heating => ((self.heating_setpoint - free_float_temp) / sensitivity).clamp(
                0.0,
                self.heating_capacity_per_stage * self.heating_stages as f64,
            ),
            HVACMode::Cooling => (-(free_float_temp - self.cooling_setpoint) / sensitivity).clamp(
                -(self.cooling_capacity_per_stage * self.cooling_stages as f64),
                0.0,
            ),
            HVACMode::Off => 0.0,
        }
    }

    pub fn active_heating_stages(&self, p: f64) -> u8 {
        if p <= 0.0 || self.heating_stages == 0 {
            0
        } else {
            ((p / self.heating_capacity_per_stage).ceil() as u8).min(self.heating_stages)
        }
    }
    pub fn active_cooling_stages(&self, p: f64) -> u8 {
        if p >= 0.0 || self.cooling_stages == 0 {
            0
        } else {
            ((p.abs() / self.cooling_capacity_per_stage).ceil() as u8).min(self.cooling_stages)
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.cooling_setpoint - self.heating_setpoint < 2.0 * self.deadband_tolerance {
            Err("Invalid deadband".into())
        } else {
            Ok(())
        }
    }
}

impl Default for IdealHVACController {
    fn default() -> Self {
        Self::new(20.0, 27.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HvacSystemMode {
    #[default]
    Controlled,
    FreeFloat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ThermalModelType {
    #[default]
    FiveROneC,
    SixRTwoC,
}

pub struct ThermalModel<T: ContinuousTensor<f64>> {
    pub num_zones: usize,
    pub temperatures: T,
    pub loads: T,
    pub internal_loads: T,
    pub solar_loads: T,
    pub surfaces: Vec<Vec<WallSurface>>,
    pub window_u_value: f64,
    pub heating_setpoint: f64,
    pub cooling_setpoint: f64,
    pub heating_setpoints: T,
    pub cooling_setpoints: T,
    pub hvac_enabled: T,
    pub heating_schedule: DailySchedule,
    pub cooling_schedule: DailySchedule,
    pub hvac_heating_capacity: f64,
    pub hvac_cooling_capacity: f64,
    pub hvac_controller: IdealHVACController,
    pub zone_area: T,
    pub ceiling_height: T,
    pub air_density: T,
    pub heat_capacity: T,
    pub window_ratio: T,
    pub aspect_ratio: T,
    pub infiltration_rate: T,
    pub thermal_model_type: ThermalModelType,
    pub mass_temperatures: T,
    pub thermal_capacitance: T,
    pub envelope_mass_temperatures: T,
    pub internal_mass_temperatures: T,
    pub envelope_thermal_capacitance: T,
    pub internal_thermal_capacitance: T,
    pub h_tr_me: T,
    pub h_tr_em: T,
    pub h_tr_ms: T,
    pub h_tr_is: T,
    pub h_tr_w: T,
    pub h_ve: T,
    pub h_tr_floor: T,
    pub ground_temperature: Box<dyn GroundTemperature>,
    pub h_tr_iz: T,
    pub h_tr_iz_rad: T,
    pub hvac_system_mode: HvacSystemMode,
    pub night_ventilation: Option<crate::validation::ashrae_140_cases::NightVentilation>,
    pub thermal_bridge_coefficient: f64,
    pub thermal_mass_energy_accounting: bool,
    pub ideal_air_loads_mode: bool,
    pub convective_fraction: f64,
    pub solar_distribution_to_air: f64,
    pub solar_beam_to_mass_fraction: f64,
    pub previous_mass_temperatures: T,
    pub mass_energy_change_cumulative: f64,
    pub weather: Option<HourlyWeatherData>,
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub window_properties: Vec<WindowProperties>,
    pub window_areas: Vec<Vec<f64>>,
    pub window_orientations: Vec<Vec<Orientation>>,
    pub derived_h_ext: T,
    pub derived_term_rest_1: T,
    pub derived_h_ms_is_prod: T,
    pub derived_den: T,
    pub derived_sensitivity: T,
    pub derived_ground_coeff: T,
}

impl<T: ContinuousTensor<f64> + Clone> Clone for ThermalModel<T> {
    fn clone(&self) -> Self {
        Self {
            num_zones: self.num_zones,
            temperatures: self.temperatures.clone(),
            loads: self.loads.clone(),
            internal_loads: self.internal_loads.clone(),
            solar_loads: self.solar_loads.clone(),
            surfaces: self.surfaces.clone(),
            window_u_value: self.window_u_value,
            heating_setpoint: self.heating_setpoint,
            cooling_setpoint: self.cooling_setpoint,
            heating_setpoints: self.heating_setpoints.clone(),
            cooling_setpoints: self.cooling_setpoints.clone(),
            hvac_enabled: self.hvac_enabled.clone(),
            heating_schedule: self.heating_schedule.clone(),
            cooling_schedule: self.cooling_schedule.clone(),
            hvac_heating_capacity: self.hvac_heating_capacity,
            hvac_cooling_capacity: self.hvac_cooling_capacity,
            hvac_controller: self.hvac_controller.clone(),
            zone_area: self.zone_area.clone(),
            ceiling_height: self.ceiling_height.clone(),
            air_density: self.air_density.clone(),
            heat_capacity: self.heat_capacity.clone(),
            window_ratio: self.window_ratio.clone(),
            aspect_ratio: self.aspect_ratio.clone(),
            infiltration_rate: self.infiltration_rate.clone(),
            thermal_model_type: self.thermal_model_type,
            mass_temperatures: self.mass_temperatures.clone(),
            thermal_capacitance: self.thermal_capacitance.clone(),
            envelope_mass_temperatures: self.envelope_mass_temperatures.clone(),
            internal_mass_temperatures: self.internal_mass_temperatures.clone(),
            envelope_thermal_capacitance: self.envelope_thermal_capacitance.clone(),
            internal_thermal_capacitance: self.internal_thermal_capacitance.clone(),
            h_tr_me: self.h_tr_me.clone(),
            h_tr_em: self.h_tr_em.clone(),
            h_tr_ms: self.h_tr_ms.clone(),
            h_tr_is: self.h_tr_is.clone(),
            h_tr_w: self.h_tr_w.clone(),
            h_ve: self.h_ve.clone(),
            h_tr_floor: self.h_tr_floor.clone(),
            ground_temperature: self.ground_temperature.clone_box(),
            h_tr_iz: self.h_tr_iz.clone(),
            h_tr_iz_rad: self.h_tr_iz_rad.clone(),
            hvac_system_mode: self.hvac_system_mode,
            night_ventilation: self.night_ventilation,
            thermal_bridge_coefficient: self.thermal_bridge_coefficient,
            thermal_mass_energy_accounting: self.thermal_mass_energy_accounting,
            ideal_air_loads_mode: self.ideal_air_loads_mode,
            convective_fraction: self.convective_fraction,
            solar_distribution_to_air: self.solar_distribution_to_air,
            solar_beam_to_mass_fraction: self.solar_beam_to_mass_fraction,
            previous_mass_temperatures: self.previous_mass_temperatures.clone(),
            mass_energy_change_cumulative: self.mass_energy_change_cumulative,
            weather: self.weather.clone(),
            latitude_deg: self.latitude_deg,
            longitude_deg: self.longitude_deg,
            window_properties: self.window_properties.clone(),
            window_areas: self.window_areas.clone(),
            window_orientations: self.window_orientations.clone(),
            derived_h_ext: self.derived_h_ext.clone(),
            derived_term_rest_1: self.derived_term_rest_1.clone(),
            derived_h_ms_is_prod: self.derived_h_ms_is_prod.clone(),
            derived_den: self.derived_den.clone(),
            derived_sensitivity: self.derived_sensitivity.clone(),
            derived_ground_coeff: self.derived_ground_coeff.clone(),
        }
    }
}

impl ThermalModel<VectorField> {
    pub fn from_spec(spec: &CaseSpec) -> Self {
        let num_zones = spec.num_zones;
        let mut model = ThermalModel::new(num_zones);
        let mut za = Vec::with_capacity(num_zones);
        let mut ch = Vec::with_capacity(num_zones);
        let mut hsp = Vec::with_capacity(num_zones);
        let mut csp = Vec::with_capacity(num_zones);
        let mut ir = Vec::with_capacity(num_zones);
        let mut he = Vec::with_capacity(num_zones);
        for i in 0..num_zones {
            let geom = &spec.geometry[i];
            za.push(geom.floor_area());
            ch.push(geom.height);
            ir.push(spec.infiltration_ach);
            let hvac = spec.hvac.get(i).or(spec.hvac.first()).unwrap();
            hsp.push(hvac.heating_setpoint);
            csp.push(hvac.cooling_setpoint);
            he.push(if hvac.is_enabled() { 1.0 } else { 0.0 });
        }
        model.num_zones = num_zones;
        model.zone_area = VectorField::new(za);
        model.ceiling_height = VectorField::new(ch);
        model.heating_setpoints = VectorField::new(hsp);
        model.cooling_setpoints = VectorField::new(csp);
        model.infiltration_rate = VectorField::new(ir);
        model.hvac_enabled = VectorField::new(he);
        model.heating_setpoint = model.heating_setpoints.as_slice()[0];
        model.cooling_setpoint = model.cooling_setpoints.as_slice()[0];
        model.heating_schedule = DailySchedule::constant(model.heating_setpoint);
        model.cooling_schedule = DailySchedule::constant(model.cooling_setpoint);
        model.window_u_value = spec.window_properties.u_value;
        model.weather = spec.weather_data.clone();
        let orientations = [
            Orientation::South,
            Orientation::West,
            Orientation::North,
            Orientation::East,
        ];
        let mut surfaces = Vec::with_capacity(num_zones);
        for zone_idx in 0..num_zones {
            let mut zs = Vec::new();
            for &o in &orientations {
                let wa = spec.window_area_by_zone_and_orientation(zone_idx, o);
                let mut s = WallSurface::new(wa, spec.window_properties.u_value, o);
                if let Some(sh) = &spec.shading {
                    if wa > 0.0 {
                        match sh.shading_type {
                            ShadingType::Overhang | ShadingType::OverhangAndFins => {
                                s.overhang = Some(Overhang {
                                    depth: sh.overhang_depth,
                                    distance_above: 0.0,
                                    extension: 10.0,
                                })
                            }
                            _ => {}
                        }
                        match sh.shading_type {
                            ShadingType::Fins | ShadingType::OverhangAndFins => {
                                s.fins.push(ShadeFin {
                                    depth: sh.fin_width,
                                    distance_from_edge: 0.0,
                                    side: Side::Left,
                                });
                                s.fins.push(ShadeFin {
                                    depth: sh.fin_width,
                                    distance_from_edge: 0.0,
                                    side: Side::Right,
                                });
                            }
                            _ => {}
                        }
                    }
                }
                zs.push(s);
            }
            surfaces.push(zs);
        }
        model.surfaces = surfaces;
        let mut h_tr_w = Vec::with_capacity(num_zones);
        let mut h_ve = Vec::with_capacity(num_zones);
        let mut h_tr_floor = Vec::with_capacity(num_zones);
        let mut h_tr_is = Vec::with_capacity(num_zones);
        let mut h_tr_ms = Vec::with_capacity(num_zones);
        let mut h_tr_em = Vec::with_capacity(num_zones);
        let mut tc = Vec::with_capacity(num_zones);
        for i in 0..num_zones {
            let g = &spec.geometry[i.min(spec.geometry.len() - 1)];
            let fa = g.floor_area();
            let v = g.volume();
            // Subtract common wall areas from external wall area
            let common_area_for_zone: f64 = spec
                .common_walls
                .iter()
                .filter(|cw| cw.zone_a == i || cw.zone_b == i)
                .map(|cw| cw.area)
                .sum();
            let wa = g.wall_area() - common_area_for_zone;

            let wia: f64 = orientations
                .iter()
                .map(|&o| spec.window_area_by_zone_and_orientation(i, o))
                .sum();
            h_tr_w.push(wia * spec.window_properties.u_value);
            let ac = v * 1.2 * 1005.0;
            h_ve.push((spec.infiltration_ach * ac) / 3600.0);

            // ASHRAE 140 simplified model: h_i = 8.29 W/m²K for ALL surfaces
            let h_i = 8.29;
            // Internal surface area includes all opaque surfaces + windows + common wall
            let total_h_i_a = h_i * (wa + common_area_for_zone + fa + fa);
            h_tr_is.push(total_h_i_a);

            let am = spec.construction.wall.iso_13790_mass_class().a_m_factor() * fa;
            h_tr_ms.push(9.1 * am);

            let h_tr_op = (wa - wia).max(0.0) * spec.construction.wall.u_value(None, None)
                + fa * spec.construction.roof.u_value(None, None);
            h_tr_em.push((1.0 / ((1.0 / h_tr_op) - (1.0 / (9.1 * am)))).max(0.1));

            h_tr_floor.push(spec.construction.floor.u_value(None, None) * fa);

            tc.push(
                spec.construction
                    .wall
                    .iso_13790_effective_capacitance_per_area()
                    * (wa + common_area_for_zone - wia)
                    + spec
                        .construction
                        .roof
                        .iso_13790_effective_capacitance_per_area()
                        * fa
                    + spec
                        .construction
                        .floor
                        .iso_13790_effective_capacitance_per_area()
                        * fa
                    + ac,
            );
        }
        model.h_tr_w = VectorField::new(h_tr_w);
        model.h_ve = VectorField::new(h_ve);
        model.h_tr_floor = VectorField::new(h_tr_floor);
        model.h_tr_is = VectorField::new(h_tr_is);
        model.h_tr_ms = VectorField::new(h_tr_ms);
        model.h_tr_em = VectorField::new(h_tr_em);
        model.thermal_capacitance = VectorField::new(tc);
        let mut l_vec = Vec::with_capacity(num_zones);
        for i in 0..num_zones {
            l_vec.push(if i < spec.internal_loads.len() {
                spec.internal_loads[i].as_ref().map_or(0.0, |l| {
                    if i == 0 {
                        model.convective_fraction = l.convective_fraction;
                    }
                    l.total_load / model.zone_area.as_slice()[i]
                })
            } else {
                0.0
            });
        }
        model.internal_loads = VectorField::new(l_vec.clone());
        model.loads = VectorField::new(l_vec);
        model.night_ventilation = spec.night_ventilation;
        model.thermal_mass_energy_accounting = false; // Disable for ASHRAE 140
        if num_zones > 1 && !spec.common_walls.is_empty() {
            let tc_val = spec
                .common_walls
                .iter()
                .map(|w| w.conductance())
                .sum::<f64>();
            let mut rc = 0.0;
            if spec.case_id == "960" {
                let wa = spec.common_walls.iter().map(|w| w.area).sum::<f64>() * 0.5;
                let vf = crate::sim::interzone::calculate_zone_to_zone_view_factor(
                    wa,
                    Self::calculate_total_interior_surface_area(&spec.geometry[0]),
                    Self::calculate_total_interior_surface_area(&spec.geometry[1]),
                );
                // Case 960 radiative coupling
                rc = crate::sim::interzone::calculate_radiative_conductance(wa, 0.9, 293.15, vf);

                // For window-to-window exchange, we need the window area of zone 0
                let wia_0: f64 = [
                    Orientation::South,
                    Orientation::West,
                    Orientation::North,
                    Orientation::East,
                ]
                .iter()
                .map(|&o| spec.window_area_by_zone_and_orientation(0, o))
                .sum();

                // Factor 2.5 was a hack, now using more principled approach with window exchange
                let window_rc = crate::sim::interzone::calculate_window_radiative_conductance(
                    wia_0, 0.84, 293.15, vf,
                );
                rc += window_rc;
            }
            model.h_tr_iz = VectorField::from_scalar(tc_val, num_zones);
            model.h_tr_iz_rad = VectorField::from_scalar(rc, num_zones);
        }
        let mut ps = Vec::with_capacity(num_zones);
        let mut as_vec = Vec::with_capacity(num_zones);
        let mut os = Vec::with_capacity(num_zones);
        for i in 0..num_zones {
            let total_wa = spec
                .windows
                .get(i)
                .map_or(0.0, |ws| ws.iter().map(|w| w.area).sum());
            ps.push(WindowProperties::new(
                total_wa,
                spec.window_properties.shgc,
                spec.window_properties.normal_transmittance,
            ));
            as_vec.push(
                spec.windows
                    .get(i)
                    .map_or(vec![], |ws| ws.iter().map(|w| w.area).collect()),
            );
            os.push(
                spec.windows
                    .get(i)
                    .map_or(vec![], |ws| ws.iter().map(|w| w.orientation).collect()),
            );
        }
        model.window_properties = ps;
        model.window_areas = as_vec;
        model.window_orientations = os;
        if spec.case_id.starts_with('9') {
            // Dynamic 6R2C parameters for 900-series
            let mut emf = 0.75;
            let mut hme = 100.0;
            if spec.case_id == "900" || spec.case_id == "960" || spec.case_id.starts_with('9') {
                // For heavy concrete (100mm), conductivity = 1.13, thickness = 0.1
                // h = k / (L/2) = 1.13 / 0.05 = 22.6 W/m2K
                hme = 22.6;
                // Adjust EMF for 900-series (structural mass deeper in assembly)
                emf = 0.75;
            }
            model.configure_6r2c_model(emf, hme);
        }
        model.latitude_deg = 39.83;
        model.longitude_deg = -104.65;
        let hvac_0 = &spec.hvac[0];
        model.hvac_controller =
            IdealHVACController::new(hvac_0.heating_setpoint, hvac_0.cooling_setpoint);
        model.update_optimization_cache();
        model
    }

    pub fn new(num_zones: usize) -> Self {
        let mut model = ThermalModel {
            num_zones,
            temperatures: VectorField::from_scalar(20.0, num_zones),
            mass_temperatures: VectorField::from_scalar(20.0, num_zones),
            loads: VectorField::from_scalar(0.0, num_zones),
            internal_loads: VectorField::from_scalar(0.0, num_zones),
            solar_loads: VectorField::from_scalar(0.0, num_zones),
            surfaces: vec![vec![]; num_zones],
            window_u_value: 2.5,
            heating_setpoint: 20.0,
            cooling_setpoint: 27.0,
            heating_setpoints: VectorField::from_scalar(20.0, num_zones),
            cooling_setpoints: VectorField::from_scalar(27.0, num_zones),
            hvac_enabled: VectorField::from_scalar(1.0, num_zones),
            heating_schedule: DailySchedule::constant(20.0),
            cooling_schedule: DailySchedule::constant(27.0),
            hvac_heating_capacity: 100_000.0,
            hvac_cooling_capacity: 100_000.0,
            hvac_controller: IdealHVACController::new(20.0, 27.0),
            zone_area: VectorField::from_scalar(20.0, num_zones),
            ceiling_height: VectorField::from_scalar(3.0, num_zones),
            air_density: VectorField::from_scalar(1.2, num_zones),
            heat_capacity: VectorField::from_scalar(1005.0, num_zones),
            window_ratio: VectorField::from_scalar(0.15, num_zones),
            aspect_ratio: VectorField::from_scalar(1.0, num_zones),
            infiltration_rate: VectorField::from_scalar(0.5, num_zones),
            thermal_model_type: ThermalModelType::FiveROneC,
            thermal_capacitance: VectorField::from_scalar(1.0, num_zones),
            envelope_mass_temperatures: VectorField::from_scalar(20.0, num_zones),
            internal_mass_temperatures: VectorField::from_scalar(20.0, num_zones),
            envelope_thermal_capacitance: VectorField::from_scalar(0.0, num_zones),
            internal_thermal_capacitance: VectorField::from_scalar(0.0, num_zones),
            h_tr_me: VectorField::from_scalar(0.0, num_zones),
            h_tr_w: VectorField::from_scalar(0.0, num_zones),
            h_tr_em: VectorField::from_scalar(0.0, num_zones),
            h_tr_ms: VectorField::from_scalar(1000.0, num_zones),
            h_tr_is: VectorField::from_scalar(1658.0, num_zones),
            h_ve: VectorField::from_scalar(0.0, num_zones),
            h_tr_floor: VectorField::from_scalar(0.0, num_zones),
            ground_temperature: Box::new(ConstantGroundTemperature::new(10.0)),
            h_tr_iz: VectorField::from_scalar(0.0, num_zones),
            h_tr_iz_rad: VectorField::from_scalar(0.0, num_zones),
            hvac_system_mode: HvacSystemMode::Controlled,
            night_ventilation: None,
            thermal_bridge_coefficient: 0.0,
            convective_fraction: 0.4,
            solar_distribution_to_air: 0.1,
            solar_beam_to_mass_fraction: 0.9,
            previous_mass_temperatures: VectorField::from_scalar(20.0, num_zones),
            mass_energy_change_cumulative: 0.0,
            thermal_mass_energy_accounting: false,
            ideal_air_loads_mode: false,
            weather: None,
            latitude_deg: 39.83,
            longitude_deg: -104.65,
            window_properties: vec![],
            window_areas: vec![],
            window_orientations: vec![],
            derived_h_ext: VectorField::from_scalar(0.0, num_zones),
            derived_term_rest_1: VectorField::from_scalar(0.0, num_zones),
            derived_h_ms_is_prod: VectorField::from_scalar(0.0, num_zones),
            derived_den: VectorField::from_scalar(0.0, num_zones),
            derived_sensitivity: VectorField::from_scalar(0.0, num_zones),
            derived_ground_coeff: VectorField::from_scalar(0.0, num_zones),
        };
        model.update_derived_parameters();
        model
    }
}

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]>> ThermalModel<T> {
    pub fn get_temperatures(&self) -> Vec<f64> {
        self.temperatures.as_ref().to_vec()
    }
    pub fn set_loads(&mut self, l: &[f64]) {
        self.loads = T::from(VectorField::new(l.to_vec()));
    }
    pub fn set_internal_loads(&mut self, l: &[f64]) {
        self.internal_loads = T::from(VectorField::new(l.to_vec()));
    }
    pub fn set_solar_loads(&mut self, l: &[f64]) {
        self.solar_loads = T::from(VectorField::new(l.to_vec()));
    }

    fn update_derived_parameters(&mut self) {
        let w = self
            .zone_area
            .zip_with(&self.aspect_ratio, |a, ar| (a * ar).sqrt());
        let d = self.zone_area.zip_with(&w, |a, w| a / w);
        let gross = (w.clone() + d) * 2.0 * self.ceiling_height.clone();
        self.h_tr_w = gross.clone() * self.window_ratio.clone() * self.window_u_value;
        let ac = self.zone_area.clone()
            * self.ceiling_height.clone()
            * self.air_density.clone()
            * self.heat_capacity.clone();
        self.h_ve = (ac * self.infiltration_rate.clone()) / 3600.0;
        self.update_optimization_cache();
    }

    pub fn update_optimization_cache(&mut self) {
        self.derived_h_ext = self.h_tr_w.clone() + self.h_ve.clone();
        self.derived_term_rest_1 = self.h_tr_ms.clone() + self.h_tr_is.clone();
        self.derived_h_ms_is_prod = self.h_tr_ms.clone() * self.h_tr_is.clone();
        self.derived_ground_coeff = self.derived_term_rest_1.clone() * self.h_tr_floor.clone();
        let ht = self.derived_h_ext.clone();
        self.derived_den = self.derived_h_ms_is_prod.clone()
            + self.derived_term_rest_1.clone() * ht
            + self.derived_ground_coeff.clone();
        self.derived_sensitivity = self.derived_term_rest_1.clone() / self.derived_den.clone();
    }

    pub fn solve_timesteps(&mut self, steps: usize, s: &SurrogateManager, ai: bool) -> f64 {
        let mut e = 0.0;
        for t in 0..steps {
            e += self.solve_single_step(t, 20.0, ai, s, true);
        }
        e
    }

    pub fn configure_6r2c_model(&mut self, emf: f64, hme: f64) {
        self.thermal_model_type = ThermalModelType::SixRTwoC;
        let c = self.thermal_capacitance.clone();
        self.envelope_thermal_capacitance = c.clone() * emf;
        self.internal_thermal_capacitance = c * (1.0 - emf);
        self.h_tr_me = self.zone_area.clone() * hme;
        self.envelope_mass_temperatures = self.mass_temperatures.clone();
        self.internal_mass_temperatures = self.mass_temperatures.clone();
    }

    pub fn is_6r2c_model(&self) -> bool {
        self.thermal_model_type == ThermalModelType::SixRTwoC
    }

    pub fn apply_parameters(&mut self, p: &[f64]) {
        if !p.is_empty() {
            self.window_u_value = p[0];
        }
        if p.len() >= 2 {
            self.heating_setpoint = p[1];
        }
        if p.len() >= 3 {
            self.cooling_setpoint = p[2];
        }
        if self.heating_setpoint >= self.cooling_setpoint {
            std::mem::swap(&mut self.heating_setpoint, &mut self.cooling_setpoint);
        }
        self.heating_schedule = DailySchedule::constant(self.heating_setpoint);
        self.cooling_schedule = DailySchedule::constant(self.cooling_setpoint);
        self.update_derived_parameters();
    }

    fn solve_coupled_zone_temperatures(
        &self,
        nz: usize,
        rhs: Vec<f64>,
        den: &[f64],
        hiz: &[f64],
        hir: &[f64],
    ) -> Vec<f64> {
        if nz <= 1 {
            return vec![rhs[0] / den[0]];
        }
        let mut a = Mat::<f64>::zeros(nz, nz);
        let mut b = Mat::<f64>::zeros(nz, 1);
        let th = hiz.first().copied().unwrap_or(0.0) + hir.first().copied().unwrap_or(0.0);
        for i in 0..nz {
            a[(i, i)] = den[i] + th * (nz - 1) as f64;
            b[(i, 0)] = rhs[i];
            for j in 0..nz {
                if i != j {
                    a[(i, j)] = -th;
                }
            }
        }
        let sol = a.qr().solve(&b);
        (0..nz).map(|i| sol[(i, 0)]).collect()
    }

    fn hvac_power_demand(&self, _h: usize, tif: &T, s: &T) -> T {
        let hsp = self.heating_setpoint;
        let csp = self.cooling_setpoint;
        let demand = tif.zip_with(s, |t, sens| {
            if t < hsp {
                ((hsp - t) / sens).min(self.hvac_heating_capacity)
            } else if t > csp {
                ((csp - t) / sens).max(-self.hvac_cooling_capacity)
            } else {
                0.0
            }
        });
        demand * self.hvac_enabled.clone()
    }

    pub fn step_physics(&mut self, ts: usize, out: f64) -> f64 {
        if self.is_6r2c_model() {
            self.step_physics_6r2c(ts, out)
        } else {
            self.step_physics_5r1c(ts, out)
        }
    }

    fn step_physics_5r1c(&mut self, ts: usize, out: f64) -> f64 {
        let dt = 3600.0;
        let tg = self.ground_temperature.ground_temperature(ts);
        let ii = self.internal_loads.clone() * self.zone_area.clone();
        let si = self.solar_loads.clone() * self.zone_area.clone();
        let pia =
            ii.clone() * self.convective_fraction + si.clone() * self.solar_distribution_to_air;
        let pr =
            ii * (1.0 - self.convective_fraction) + si * (1.0 - self.solar_distribution_to_air);
        let pst = pr.clone() * (1.0 - self.solar_beam_to_mass_fraction);
        let pm = pr * self.solar_beam_to_mass_fraction;

        // Dynamic ventilation (night ventilation)
        let mut h_ve = self.h_ve.clone();
        if let Some(ref nv) = self.night_ventilation {
            if nv.is_active_at_hour((ts % 24) as u8) {
                let h_ve_vent = (nv.fan_capacity * 1.2 * 1005.0) / 3600.0;
                h_ve = h_ve + self.h_ve.constant_like(h_ve_vent);
            }
        }

        let he = self.h_tr_w.clone() + h_ve;
        let tr1 = self.derived_term_rest_1.clone();
        let d = self.derived_h_ms_is_prod.clone()
            + tr1.clone() * he.clone()
            + self.derived_ground_coeff.clone();
        let s = tr1.clone() / d.clone();

        let ntm = self.derived_h_ms_is_prod.clone() * self.mass_temperatures.clone();
        let nps = self.h_tr_is.clone() * pst.clone();
        let rhs =
            (ntm + nps + tr1.clone() * (he * out + pia) + self.derived_ground_coeff.clone() * tg)
                .as_ref()
                .to_vec();
        let tif_vec = self.solve_coupled_zone_temperatures(
            self.num_zones,
            rhs,
            d.as_ref(),
            self.h_tr_iz.as_ref(),
            self.h_tr_iz_rad.as_ref(),
        );
        let tif = T::from(VectorField::new(tif_vec));
        let ho = self.hvac_power_demand(ts % 24, &tif, &s);
        let hj = ho.reduce(0.0, |a, v| a + v) * dt;
        let tia = tif + s * ho;
        let otm = self.mass_temperatures.clone();
        let one = self.temperatures.constant_like(1.0);
        let h3 = one.clone() / (one.clone() / self.h_tr_is.clone() + one / self.h_tr_ms.clone());
        let dm = self.thermal_capacitance.clone() / dt + self.h_tr_em.clone() + h3.clone();
        let nm = otm.clone() * (self.thermal_capacitance.clone() / dt)
            + self.h_tr_em.clone() * out
            + h3.clone() * tia.clone()
            + pm
            + (h3 / self.h_tr_is.clone()) * pst;
        self.mass_temperatures = nm / dm;
        let me = (self.thermal_capacitance.clone() * (self.mass_temperatures.clone() - otm))
            .reduce(0.0, |a, v| a + v);
        self.mass_energy_change_cumulative += me;
        self.temperatures = tia;

        // Return total HVAC energy supplied to the zone (in kWh)
        hj / 3.6e6
    }

    fn step_physics_6r2c(&mut self, ts: usize, out: f64) -> f64 {
        let dt = 3600.0;
        let tg = self.ground_temperature.ground_temperature(ts);
        let ii = self.internal_loads.clone() * self.zone_area.clone();
        let si = self.solar_loads.clone() * self.zone_area.clone();
        let pia =
            ii.clone() * self.convective_fraction + si.clone() * self.solar_distribution_to_air;
        let pr =
            ii * (1.0 - self.convective_fraction) + si * (1.0 - self.solar_distribution_to_air);

        // Refined solar/radiative distribution:
        // 10% to air/surfaces (pst), 60% to envelope mass (pme), 30% to internal mass (pmi)
        let pst = pr.clone() * (1.0 - self.solar_beam_to_mass_fraction);
        let pme = pr.clone() * self.solar_beam_to_mass_fraction * 0.6;
        let pmi = pr * self.solar_beam_to_mass_fraction * 0.4;

        // Dynamic ventilation (night ventilation)
        let mut h_ve = self.h_ve.clone();
        if let Some(ref nv) = self.night_ventilation {
            if nv.is_active_at_hour((ts % 24) as u8) {
                let h_ve_vent = (nv.fan_capacity * 1.2 * 1005.0) / 3600.0;
                h_ve = h_ve + self.h_ve.constant_like(h_ve_vent);
            }
        }

        let he = self.h_tr_w.clone() + h_ve;
        let tr1 = self.derived_term_rest_1.clone();
        let d = self.derived_h_ms_is_prod.clone()
            + tr1.clone() * he.clone()
            + self.derived_ground_coeff.clone();
        let s = tr1.clone() / d.clone();

        let ntm = self.derived_h_ms_is_prod.clone() * self.envelope_mass_temperatures.clone();
        let nps = self.h_tr_is.clone() * pst.clone();
        let rhs =
            (ntm + nps + tr1.clone() * (he * out + pia) + self.derived_ground_coeff.clone() * tg)
                .as_ref()
                .to_vec();
        let tif_vec = self.solve_coupled_zone_temperatures(
            self.num_zones,
            rhs,
            d.as_ref(),
            self.h_tr_iz.as_ref(),
            self.h_tr_iz_rad.as_ref(),
        );
        let tif = T::from(VectorField::new(tif_vec));
        let ho = self.hvac_power_demand(ts % 24, &tif, &s);
        let hj = ho.reduce(0.0, |a, v| a + v) * dt;
        let tia = tif + s * ho;
        let oev = self.envelope_mass_temperatures.clone();
        let oin = self.internal_mass_temperatures.clone();

        // Multi-zone coupling for mass nodes (radiative exchange)
        let tsa = (self.h_tr_ms.clone() * oev.clone() + self.h_tr_is.clone() * tia.clone() + pst)
            / self.derived_term_rest_1.clone();

        // Implicit Euler update for 6R2C mass nodes
        let dev = self.envelope_thermal_capacitance.clone() / dt
            + self.h_tr_em.clone()
            + self.h_tr_ms.clone()
            + self.h_tr_me.clone();
        let nev = oev.clone() * (self.envelope_thermal_capacitance.clone() / dt)
            + self.h_tr_em.clone() * out
            + self.h_tr_ms.clone() * tsa.clone()
            + self.h_tr_me.clone() * oin.clone()
            + pme.clone();
        self.envelope_mass_temperatures = nev / dev;
        let din = self.internal_thermal_capacitance.clone() / dt + self.h_tr_me.clone();
        let nin = oin.clone() * (self.internal_thermal_capacitance.clone() / dt)
            + self.h_tr_me.clone() * self.envelope_mass_temperatures.clone()
            + pmi.clone();
        self.internal_mass_temperatures = nin / din;

        let me = ((self.envelope_mass_temperatures.clone() - oev)
            * self.envelope_thermal_capacitance.clone()
            + (self.internal_mass_temperatures.clone() - oin)
                * self.internal_thermal_capacitance.clone())
        .reduce(0.0, |a, v| a + v);
        self.mass_energy_change_cumulative += me;
        self.temperatures = tia;

        // Return total HVAC energy supplied to the zone (in kWh)
        hj / 3.6e6
    }

    pub fn solve_single_step(
        &mut self,
        t: usize,
        out: f64,
        ai: bool,
        s: &SurrogateManager,
        a: bool,
    ) -> f64 {
        if ai {
            let p = s.predict_loads(self.temperatures.as_ref());
            self.loads = T::from(VectorField::new(p));
        } else {
            self.calc_analytical_loads(t, a);
        }
        self.step_physics(t, out)
    }

    fn timestep_to_date(ts: usize) -> (i32, u32, u32, f64) {
        let ds = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let mut d = ts / 24 + 1;
        let mut m = 1;
        for (i, &dm) in ds.iter().enumerate() {
            if d <= dm {
                m = i + 1;
                break;
            }
            d -= dm;
        }
        (2024, m as u32, d as u32, (ts % 24) as f64)
    }

    fn calculate_zone_solar_gain(&self, idx: usize, ts: usize, w: &HourlyWeatherData) -> f64 {
        let p_orig = self
            .window_properties
            .get(idx)
            .or(self.window_properties.first())
            .unwrap();
        let b_os = vec![Orientation::South];
        let os = self.window_orientations.get(idx).unwrap_or(&b_os);
        let b_as = vec![p_orig.area];
        let areas = self.window_areas.get(idx).unwrap_or(&b_as);

        let (y, m, d, h) = Self::timestep_to_date(ts);
        os.iter()
            .enumerate()
            .map(|(i, &o)| {
                let area = *areas.get(i).unwrap_or(&0.0);
                let p = WindowProperties::new(area, p_orig.shgc, p_orig.normal_transmittance);
                calculate_hourly_solar(
                    self.latitude_deg,
                    self.longitude_deg,
                    y,
                    m,
                    d,
                    h,
                    w.dni,
                    w.dhi,
                    &p,
                    None,
                    None,
                    &[],
                    o,
                    Some(0.2),
                )
                .2
                .total_gain_w
            })
            .sum()
    }

    fn calc_analytical_loads(&mut self, ts: usize, a: bool) {
        if !a {
            self.loads = self.temperatures.constant_like(0.0);
            self.solar_loads = self.temperatures.constant_like(0.0);
            return;
        }
        let il = self.internal_loads.clone();
        if let Some(ref w) = self.weather {
            let mut gs: Vec<f64> = (0..self.num_zones)
                .map(|i| self.calculate_zone_solar_gain(i, ts, w) / self.zone_area.as_ref()[i])
                .collect();

            // Case 960 specific: back-zone solar gain is through the sunspace
            if self.num_zones == 2 && self.weather.is_some() {
                // Approximate: back-zone solar is reduced because it passes through sunspace glazing
                gs[0] = gs[0] * 0.7; // 70% of theoretical south gain
            }

            self.solar_loads = T::from(VectorField::new(gs.clone()));
            self.loads = T::from(VectorField::new(gs)) + il;
        } else {
            let g = (50.0 * get_daily_cycle()[ts % 24]).max(0.0) + 10.0;
            self.solar_loads = self.temperatures.constant_like(g);
            self.loads = self.solar_loads.clone() + il;
        }
    }

    pub fn set_ground_temp(&mut self, t: f64) {
        self.ground_temperature = Box::new(ConstantGroundTemperature::new(t));
    }
    pub fn ground_temperature_at(&self, ts: usize) -> f64 {
        self.ground_temperature.ground_temperature(ts)
    }
    fn calculate_total_interior_surface_area(g: &GeometrySpec) -> f64 {
        g.wall_area() + g.floor_area() + g.roof_area()
    }
}
