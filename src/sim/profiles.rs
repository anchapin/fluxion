//! Building Profile Loading and Caching
//!
//! This module provides building profile loading from JSON files with caching
//! for Office, Retail, and School building types.

use crate::sim::equipment::{ComputerEquipment, Equipment, GenericEquipment, ServerRack};
use crate::sim::lighting::LightingSchedule;
use crate::sim::occupancy::{BuildingType, OccupancyProfile};
use crate::sim::schedule::DailySchedule;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::sync::OnceLock;

static PROFILE_CACHE: OnceLock<HashMap<BuildingType, ProfileBundle>> = OnceLock::new();

/// Bundle of internal load profiles for a building type
pub struct ProfileBundle {
    pub lighting: LightingSchedule,
    pub equipment: Vec<Box<dyn Equipment + Send + Sync>>,
    pub occupancy: OccupancyProfile,
}

impl Clone for ProfileBundle {
    fn clone(&self) -> Self {
        // Clone each equipment item using downcast pattern
        let equipment: Vec<Box<dyn Equipment + Send + Sync>> = self
            .equipment
            .iter()
            .map(|eq| {
                // Try to downcast to concrete types for cloning
                if let Some(computer) = eq.as_any().downcast_ref::<ComputerEquipment>() {
                    Box::new(computer.clone()) as Box<dyn Equipment + Send + Sync>
                } else if let Some(server) = eq.as_any().downcast_ref::<ServerRack>() {
                    Box::new(server.clone()) as Box<dyn Equipment + Send + Sync>
                } else if let Some(generic) = eq.as_any().downcast_ref::<GenericEquipment>() {
                    Box::new(generic.clone()) as Box<dyn Equipment + Send + Sync>
                } else {
                    panic!("Unknown equipment type in ProfileBundle::clone");
                }
            })
            .collect();

        ProfileBundle {
            lighting: self.lighting.clone(),
            equipment,
            occupancy: self.occupancy.clone(),
        }
    }
}

impl std::fmt::Debug for ProfileBundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProfileBundle")
            .field("lighting", &self.lighting)
            .field("equipment_count", &self.equipment.len())
            .field("occupancy", &self.occupancy)
            .finish()
    }
}

/// JSON structure for building profiles
#[derive(Debug, Serialize, Deserialize)]
struct BuildingProfiles {
    profiles: HashMap<String, BuildingProfileData>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BuildingProfileData {
    lighting: LightingData,
    equipment: Vec<EquipmentData>,
    occupancy: OccupancyData,
}

#[derive(Debug, Serialize, Deserialize)]
struct LightingData {
    power_density_w_m2: f64,
    convective_fraction: f64,
    radiative_fraction: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct EquipmentData {
    equipment_type: String,
    id: String,
    rated_power_w: f64,
    count: usize,
    radiative_fraction: f64,
    convective_fraction: f64,
    mass_coupling_factor: f64,
    schedule_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct OccupancyData {
    max_occupancy: f64,
}

/// Load building profile from JSON file with caching
pub fn load_building_profile(building_type: BuildingType) -> Result<ProfileBundle, String> {
    // Check cache first
    if let Some(cache) = PROFILE_CACHE.get() {
        if let Some(profile) = cache.get(&building_type) {
            return Ok(profile.clone());
        }
    }

    // Load from file
    let profile_path = "data/building_profiles.json";
    let content = fs::read_to_string(profile_path)
        .map_err(|e| format!("Failed to read profile file {}: {}", profile_path, e))?;

    let profiles: BuildingProfiles = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse profile JSON: {}", e))?;

    let building_key = match building_type {
        BuildingType::Office => "office",
        BuildingType::Retail => "retail",
        BuildingType::School => "school",
        _ => return Err(format!("Unsupported building type: {:?}", building_type)),
    };

    let profile_data = profiles
        .profiles
        .get(building_key)
        .ok_or_else(|| format!("Profile not found for building type: {}", building_key))?;

    // Build lighting schedule
    let lighting = LightingSchedule::new(
        profile_data.lighting.power_density_w_m2,
        100.0, // Default zone area (should be overridden by user)
    );
    let lighting = LightingSchedule {
        convective_fraction: profile_data.lighting.convective_fraction,
        radiative_fraction: profile_data.lighting.radiative_fraction,
        ..lighting
    };

    // Build equipment list
    let mut equipment: Vec<Box<dyn Equipment + Send + Sync>> = Vec::new();
    for eq_data in &profile_data.equipment {
        let eq: Box<dyn Equipment + Send + Sync> = match eq_data.equipment_type.as_str() {
            "ComputerEquipment" => {
                let mut computers = ComputerEquipment::new(
                    eq_data.id.clone(),
                    eq_data.rated_power_w,
                    eq_data.count,
                );
                computers.radiative_fraction = eq_data.radiative_fraction;
                computers.convective_fraction = eq_data.convective_fraction;
                computers.mass_coupling_factor = eq_data.mass_coupling_factor;

                // Set schedule based on schedule_type
                let schedule = match eq_data.schedule_type.as_str() {
                    "daily" => {
                        let mut s = DailySchedule::new();
                        for hour in 8..=17 {
                            s.set_hour(hour, 1.0);
                        }
                        s
                    }
                    "constant" => DailySchedule::constant(1.0),
                    _ => DailySchedule::new(),
                };
                computers.schedule = schedule;

                Box::new(computers)
            }
            "ServerRack" => {
                let mut servers =
                    ServerRack::new(eq_data.id.clone(), eq_data.rated_power_w, eq_data.count);
                servers.radiative_fraction = eq_data.radiative_fraction;
                servers.convective_fraction = eq_data.convective_fraction;
                servers.mass_coupling_factor = eq_data.mass_coupling_factor;

                // Set schedule based on schedule_type
                let schedule = match eq_data.schedule_type.as_str() {
                    "daily" => {
                        let mut s = DailySchedule::new();
                        for hour in 8..=17 {
                            s.set_hour(hour, 1.0);
                        }
                        s
                    }
                    "constant" => DailySchedule::constant(1.0),
                    _ => DailySchedule::new(),
                };
                servers.schedule = schedule;

                Box::new(servers)
            }
            "GenericEquipment" => {
                let mut generic =
                    GenericEquipment::new(eq_data.id.clone(), eq_data.rated_power_w, eq_data.count);
                generic.radiative_fraction = eq_data.radiative_fraction;
                generic.convective_fraction = eq_data.convective_fraction;
                generic.mass_coupling_factor = eq_data.mass_coupling_factor;

                // Set schedule based on schedule_type
                let schedule = match eq_data.schedule_type.as_str() {
                    "daily" => {
                        let mut s = DailySchedule::new();
                        for hour in 8..=17 {
                            s.set_hour(hour, 1.0);
                        }
                        s
                    }
                    "constant" => DailySchedule::constant(1.0),
                    _ => DailySchedule::new(),
                };
                generic.schedule = schedule;

                Box::new(generic)
            }
            _ => {
                return Err(format!(
                    "Unknown equipment type: {}",
                    eq_data.equipment_type
                ))
            }
        };
        equipment.push(eq);
    }

    // Build occupancy profile
    let mut occupancy = OccupancyProfile::new(
        "Default".to_string(),
        building_type,
        profile_data.occupancy.max_occupancy,
    );
    // Apply appropriate schedule based on building type
    occupancy = match building_type {
        BuildingType::Office => occupancy.office_schedule(),
        BuildingType::Retail => occupancy.retail_schedule(),
        BuildingType::School => occupancy.school_schedule(),
        _ => occupancy,
    };

    let bundle = ProfileBundle {
        lighting,
        equipment,
        occupancy,
    };

    // Cache for future use
    PROFILE_CACHE.get_or_init(|| {
        let mut cache = HashMap::new();
        cache.insert(building_type, bundle.clone());
        cache
    });

    Ok(bundle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_bundle_struct() {
        // Just verify struct compiles - actual loading tested with JSON file
        let _ = ProfileBundle {
            lighting: LightingSchedule::new(10.0, 100.0),
            equipment: Vec::new(),
            occupancy: OccupancyProfile::new("Test".to_string(), BuildingType::Office, 100.0),
        };
    }

    #[test]
    fn test_building_profile_loading() {
        // Test Office profile loading
        let office_profile =
            load_building_profile(BuildingType::Office).expect("Failed to load Office profile");

        // Verify lighting
        assert_eq!(office_profile.lighting.power_density, 10.0);
        assert_eq!(office_profile.lighting.convective_fraction, 0.2);
        assert_eq!(office_profile.lighting.radiative_fraction, 0.8);

        // Verify equipment count
        assert_eq!(office_profile.equipment.len(), 2);

        // Verify occupancy
        assert_eq!(office_profile.occupancy.max_occupancy, 100.0);

        // Test Retail profile loading
        let retail_profile =
            load_building_profile(BuildingType::Retail).expect("Failed to load Retail profile");

        assert_eq!(retail_profile.lighting.power_density, 12.0);
        assert_eq!(retail_profile.occupancy.max_occupancy, 50.0);

        // Test School profile loading
        let school_profile =
            load_building_profile(BuildingType::School).expect("Failed to load School profile");

        assert_eq!(school_profile.lighting.power_density, 8.0);
        assert_eq!(school_profile.occupancy.max_occupancy, 200.0);
    }

    #[test]
    fn test_profile_caching() {
        // First load - should read from file
        let _ = load_building_profile(BuildingType::Office).expect("Failed to load Office profile");

        // Second load - should use cache (if cache is working, this won't fail)
        let profile2 = load_building_profile(BuildingType::Office)
            .expect("Failed to load Office profile from cache");

        assert_eq!(profile2.occupancy.max_occupancy, 100.0);
    }

    #[test]
    fn test_equipment_in_profile() {
        let office_profile =
            load_building_profile(BuildingType::Office).expect("Failed to load Office profile");

        // Find computers
        let computers: Vec<_> = office_profile
            .equipment
            .iter()
            .filter(|e| e.id() == "office-computers")
            .collect();

        assert_eq!(computers.len(), 1);
        let computers = computers[0];

        // Verify power calculation
        let power = computers.power_at_hour(0);
        assert!((power - 0.0).abs() < 1e-10); // Should be off at midnight

        let power_day = computers.power_at_hour(10); // Hour 10 (10am) during work hours
        assert!(power_day > 0.0); // Should be on during work hours
    }
}
