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

#[cfg(test)]
static PROFILE_FILE_READS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

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

/// Build the internal-load bundle for one building type from parsed JSON data.
fn build_profile_bundle(
    building_type: BuildingType,
    profile_data: &BuildingProfileData,
) -> Result<ProfileBundle, String> {
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
                            s.set_hour(hour, 1.0)
                                .expect("set_hour on a fresh daily schedule cannot fail");
                        }
                        s
                    }
                    "constant" => DailySchedule::constant(1.0)
                        .expect("constant() on a fresh daily schedule cannot fail"),
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
                            s.set_hour(hour, 1.0)
                                .expect("set_hour on a fresh daily schedule cannot fail");
                        }
                        s
                    }
                    "constant" => DailySchedule::constant(1.0)
                        .expect("constant() on a fresh daily schedule cannot fail"),
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
                            s.set_hour(hour, 1.0)
                                .expect("set_hour on a fresh daily schedule cannot fail");
                        }
                        s
                    }
                    "constant" => DailySchedule::constant(1.0)
                        .expect("constant() on a fresh daily schedule cannot fail"),
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

    Ok(ProfileBundle {
        lighting,
        equipment,
        occupancy,
    })
}

/// Build bundles for every supported building type from one parsed profile file.
fn build_all_bundles(
    profiles: &BuildingProfiles,
) -> Result<HashMap<BuildingType, ProfileBundle>, String> {
    let mut bundles = HashMap::new();
    for (building_type, building_key) in [
        (BuildingType::Office, "office"),
        (BuildingType::Retail, "retail"),
        (BuildingType::School, "school"),
    ] {
        let profile_data = profiles
            .profiles
            .get(building_key)
            .ok_or_else(|| format!("Profile not found for building type: {}", building_key))?;
        bundles.insert(
            building_type,
            build_profile_bundle(building_type, profile_data)?,
        );
    }
    Ok(bundles)
}

/// Load building profile from JSON file with caching
///
/// The cache is keyed by [`BuildingType`] and populated eagerly on first use:
/// a single file read + parse produces bundles for every supported building
/// type (issue #3649), so a request for one type can never be served another
/// type's profile and the profile file is read at most once per process on
/// the success path.
pub fn load_building_profile(building_type: BuildingType) -> Result<ProfileBundle, String> {
    // Fail fast on unsupported types without touching the cache or filesystem.
    let building_key = match building_type {
        BuildingType::Office => "office",
        BuildingType::Retail => "retail",
        BuildingType::School => "school",
        _ => return Err(format!("Unsupported building type: {:?}", building_type)),
    };

    let cache = match PROFILE_CACHE.get() {
        Some(cache) => cache,
        None => {
            let profile_path = "data/building_profiles.json";
            #[cfg(test)]
            PROFILE_FILE_READS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let content = fs::read_to_string(profile_path)
                .map_err(|e| format!("Failed to read profile file {}: {}", profile_path, e))?;

            let profiles: BuildingProfiles = serde_json::from_str(&content)
                .map_err(|e| format!("Failed to parse profile JSON: {}", e))?;

            let bundles = build_all_bundles(&profiles)?;
            PROFILE_CACHE.get_or_init(|| bundles)
        }
    };

    cache
        .get(&building_type)
        .cloned()
        .ok_or_else(|| format!("Profile not found for building type: {}", building_key))
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

    #[test]
    fn test_profile_bundle_debug() {
        let bundle = ProfileBundle {
            lighting: LightingSchedule::new(10.0, 100.0),
            equipment: Vec::new(),
            occupancy: OccupancyProfile::new("Test".to_string(), BuildingType::Office, 100.0),
        };

        let debug_str = format!("{:?}", bundle);
        assert!(debug_str.contains("ProfileBundle"));
        assert!(debug_str.contains("equipment_count"));
    }

    #[test]
    fn test_profile_bundle_clone_empty_equipment() {
        let bundle = ProfileBundle {
            lighting: LightingSchedule::new(10.0, 100.0),
            equipment: Vec::new(),
            occupancy: OccupancyProfile::new("Test".to_string(), BuildingType::Office, 100.0),
        };

        let cloned = bundle.clone();
        assert_eq!(cloned.equipment.len(), 0);
        assert_eq!(cloned.lighting.power_density, bundle.lighting.power_density);
    }

    #[test]
    fn test_profile_bundle_clone_with_equipment() {
        let mut equipment: Vec<Box<dyn Equipment + Send + Sync>> = Vec::new();
        let mut computers = ComputerEquipment::new("test-computers".to_string(), 100.0, 5);
        computers.radiative_fraction = 0.5;
        computers.convective_fraction = 0.5;
        equipment.push(Box::new(computers));

        let bundle = ProfileBundle {
            lighting: LightingSchedule::new(10.0, 100.0),
            equipment,
            occupancy: OccupancyProfile::new("Test".to_string(), BuildingType::Office, 100.0),
        };

        let cloned = bundle.clone();
        assert_eq!(cloned.equipment.len(), 1);
        assert_eq!(cloned.equipment[0].id(), "test-computers");
    }

    #[test]
    fn test_profile_bundle_clone_with_server_rack() {
        let mut equipment: Vec<Box<dyn Equipment + Send + Sync>> = Vec::new();
        let mut servers = ServerRack::new("test-servers".to_string(), 500.0, 2);
        servers.radiative_fraction = 0.8;
        servers.convective_fraction = 0.2;
        equipment.push(Box::new(servers));

        let bundle = ProfileBundle {
            lighting: LightingSchedule::new(10.0, 100.0),
            equipment,
            occupancy: OccupancyProfile::new("Test".to_string(), BuildingType::Office, 100.0),
        };

        let cloned = bundle.clone();
        assert_eq!(cloned.equipment.len(), 1);
        assert_eq!(cloned.equipment[0].id(), "test-servers");
    }

    #[test]
    fn test_profile_bundle_clone_with_generic_equipment() {
        let mut equipment: Vec<Box<dyn Equipment + Send + Sync>> = Vec::new();
        let mut generic = GenericEquipment::new("test-generic".to_string(), 200.0, 1);
        generic.radiative_fraction = 0.6;
        generic.convective_fraction = 0.4;
        equipment.push(Box::new(generic));

        let bundle = ProfileBundle {
            lighting: LightingSchedule::new(10.0, 100.0),
            equipment,
            occupancy: OccupancyProfile::new("Test".to_string(), BuildingType::Office, 100.0),
        };

        let cloned = bundle.clone();
        assert_eq!(cloned.equipment.len(), 1);
        assert_eq!(cloned.equipment[0].id(), "test-generic");
    }

    #[test]
    fn test_building_profiles_struct() {
        let mut profiles = HashMap::new();
        profiles.insert(
            "office".to_string(),
            BuildingProfileData {
                lighting: LightingData {
                    power_density_w_m2: 10.0,
                    convective_fraction: 0.2,
                    radiative_fraction: 0.8,
                },
                equipment: vec![EquipmentData {
                    equipment_type: "ComputerEquipment".to_string(),
                    id: "test".to_string(),
                    rated_power_w: 100.0,
                    count: 5,
                    radiative_fraction: 0.5,
                    convective_fraction: 0.5,
                    mass_coupling_factor: 0.3,
                    schedule_type: "daily".to_string(),
                }],
                occupancy: OccupancyData {
                    max_occupancy: 100.0,
                },
            },
        );

        let bp = BuildingProfiles { profiles };
        assert!(bp.profiles.contains_key("office"));
    }

    #[test]
    fn test_lighting_data_struct() {
        let ld = LightingData {
            power_density_w_m2: 15.0,
            convective_fraction: 0.3,
            radiative_fraction: 0.7,
        };

        assert!((ld.power_density_w_m2 - 15.0).abs() < 1e-6);
        assert!((ld.convective_fraction + ld.radiative_fraction - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_equipment_data_struct() {
        let ed = EquipmentData {
            equipment_type: "GenericEquipment".to_string(),
            id: "test-eq".to_string(),
            rated_power_w: 250.0,
            count: 3,
            radiative_fraction: 0.4,
            convective_fraction: 0.6,
            mass_coupling_factor: 0.2,
            schedule_type: "constant".to_string(),
        };

        assert_eq!(ed.equipment_type, "GenericEquipment");
        assert_eq!(ed.count, 3);
    }

    #[test]
    fn test_occupancy_data_struct() {
        let od = OccupancyData {
            max_occupancy: 150.0,
        };
        assert!((od.max_occupancy - 150.0).abs() < 1e-6);
    }

    #[test]
    fn test_cache_serves_all_building_types_from_single_read() {
        // Regression test for issue #3649: the cache used to store only the
        // FIRST loaded building type, so requests for other types re-read the
        // profile file on every call. The fix populates all known types from
        // one read, so this sequence costs at most a single fs::read_to_string.
        let reads_before = PROFILE_FILE_READS.load(std::sync::atomic::Ordering::Relaxed);

        let office =
            load_building_profile(BuildingType::Office).expect("Failed to load Office profile");
        let retail =
            load_building_profile(BuildingType::Retail).expect("Failed to load Retail profile");
        let school =
            load_building_profile(BuildingType::School).expect("Failed to load School profile");

        // Each type must return its OWN data (source values differ per type).
        assert_eq!(office.lighting.power_density, 10.0);
        assert_eq!(retail.lighting.power_density, 12.0);
        assert_eq!(school.lighting.power_density, 8.0);
        assert_eq!(office.occupancy.max_occupancy, 100.0);
        assert_eq!(retail.occupancy.max_occupancy, 50.0);
        assert_eq!(school.occupancy.max_occupancy, 200.0);
        assert_eq!(office.equipment.len(), 2);
        assert_eq!(retail.equipment.len(), 1);
        assert_eq!(school.equipment.len(), 1);

        // Repeat loads must hit the cache — no additional file reads.
        let _ = load_building_profile(BuildingType::Office)
            .expect("Failed to reload Office profile from cache");
        let _ = load_building_profile(BuildingType::Retail)
            .expect("Failed to reload Retail profile from cache");
        let _ = load_building_profile(BuildingType::School)
            .expect("Failed to reload School profile from cache");

        let reads_after = PROFILE_FILE_READS.load(std::sync::atomic::Ordering::Relaxed);
        // The fix reads the file at most once per process; parallel tests may
        // have already paid that single read. Pre-fix behavior re-read the
        // file for every non-first type (>= 4 reads for this sequence).
        let reads = reads_after.saturating_sub(reads_before);
        assert!(
            reads <= 1,
            "expected at most 1 profile-file read for Office->Retail->School->all, got {}",
            reads
        );
    }
}
