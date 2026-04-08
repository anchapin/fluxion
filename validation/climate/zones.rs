// validation/climate/zones.rs
/// Climate zone definitions and parameters
///
/// This module defines ASHRAE climate zones and their characteristics
use std::collections::HashMap;

/// Climate zone definition
#[derive(Debug, Clone)]
pub struct ClimateZone {
    pub zone_id: String,
    pub full_name: String,
    pub description: String,
    pub temperature_range_c: (f64, f64),
    pub humidity_range: (f64, f64),
    pub heating_degree_days: f64,
    pub cooling_degree_days: f64,
    pub solar_radiation_kwh_m2: f64,
    pub wind_speed_m_s: f64,
    pub precipitation_mm: f64,
    pub typical_building_type: String,
}

impl ClimateZone {
    /// Create a new climate zone
    pub fn new(
        zone_id: &str,
        full_name: &str,
        description: &str,
        temp_min: f64,
        temp_max: f64,
        humidity_min: f64,
        humidity_max: f64,
        hdd: f64,
        cdd: f64,
        solar: f64,
        wind_speed: f64,
        precipitation: f64,
        building_type: &str,
    ) -> Self {
        Self {
            zone_id: zone_id.to_string(),
            full_name: full_name.to_string(),
            description: description.to_string(),
            temperature_range_c: (temp_min, temp_max),
            humidity_range: (humidity_min, humidity_max),
            heating_degree_days: hdd,
            cooling_degree_days: cdd,
            solar_radiation_kwh_m2: solar,
            wind_speed_m_s: wind_speed,
            precipitation_mm: precipitation,
            typical_building_type: building_type.to_string(),
        }
    }
}

/// Get all major ASHRAE climate zones
pub fn get_all_climate_zones() -> HashMap<String, ClimateZone> {
    let mut zones = HashMap::new();

    // Zone 1A: Very Hot-Humid
    zones.insert(
        "1A".to_string(),
        ClimateZone::new(
            "1A",
            "Very Hot-Humid",
            "Tropical humid climates with all months >18°C (64.4°F) and warmest month >22°C (71.6°F)",
            10.0, 35.0,
            60.0, 90.0,
            0,    // No heating degree days
            3000, // High cooling degree days
            1800, // High solar radiation
            2.5,  // Low wind speed
            1500, // High precipitation
            "residential",
        ),
    );

    // Zone 2A: Hot-Humid
    zones.insert(
        "2A".to_string(),
        ClimateZone::new(
            "2A",
            "Hot-Humid",
            "Humid climates with warm summers and mild winters",
            5.0,
            33.0,
            50.0,
            85.0,
            500,
            2500,
            1700,
            3.0,
            1200,
            "residential",
        ),
    );

    // Zone 2B: Hot-Dry
    zones.insert(
        "2B".to_string(),
        ClimateZone::new(
            "2B",
            "Hot-Dry",
            "Hot arid climates with large diurnal temperature ranges",
            0.0,
            40.0,
            10.0,
            40.0,
            800,
            2800,
            2000,
            4.5, // Higher wind speed in arid areas
            200, // Low precipitation
            "commercial",
        ),
    );

    // Zone 3A: Warm-Humid
    zones.insert(
        "3A".to_string(),
        ClimateZone::new(
            "3A",
            "Warm-Humid",
            "Warm humid climates with moderate summers and mild winters",
            0.0,
            32.0,
            40.0,
            80.0,
            1200,
            2000,
            1600,
            3.2,
            1100,
            "residential",
        ),
    );

    // Zone 3B: Warm-Dry
    zones.insert(
        "3B".to_string(),
        ClimateZone::new(
            "3B",
            "Warm-Dry",
            "Warm dry climates with moderate temperature ranges",
            -5.0,
            35.0,
            15.0,
            50.0,
            1500,
            1800,
            1900,
            3.8,
            400,
            "commercial",
        ),
    );

    // Zone 3C: Warm-Marine
    zones.insert(
        "3C".to_string(),
        ClimateZone::new(
            "3C",
            "Warm-Marine",
            "Marine climates with mild temperatures and high humidity",
            2.0,
            28.0,
            60.0,
            85.0,
            1000,
            800,
            1400,
            5.0, // Higher wind speed in marine areas
            900,
            "residential",
        ),
    );

    // Zone 4A: Mixed-Humid
    zones.insert(
        "4A".to_string(),
        ClimateZone::new(
            "4A",
            "Mixed-Humid",
            "Mixed humid climates with hot summers and cold winters",
            -10.0,
            34.0,
            30.0,
            80.0,
            2500,
            1500,
            1500,
            4.0,
            1000,
            "commercial",
        ),
    );

    // Zone 4B: Mixed-Dry
    zones.insert(
        "4B".to_string(),
        ClimateZone::new(
            "4B",
            "Mixed-Dry",
            "Mixed dry climates with hot summers and cold winters",
            -15.0,
            36.0,
            10.0,
            50.0,
            3000,
            1200,
            1800,
            4.2,
            500,
            "commercial",
        ),
    );

    // Zone 4C: Mixed-Marine
    zones.insert(
        "4C".to_string(),
        ClimateZone::new(
            "4C",
            "Mixed-Marine",
            "Marine climates with mild summers and cool winters",
            -5.0,
            26.0,
            50.0,
            80.0,
            2000,
            500,
            1300,
            5.5,
            800,
            "residential",
        ),
    );

    // Zone 5A: Cool-Humid
    zones.insert(
        "5A".to_string(),
        ClimateZone::new(
            "5A",
            "Cool-Humid",
            "Cool humid climates with warm summers and cold winters",
            -20.0,
            32.0,
            20.0,
            70.0,
            4000,
            1000,
            1400,
            3.5,
            900,
            "commercial",
        ),
    );

    // Zone 5B: Cool-Dry
    zones.insert(
        "5B".to_string(),
        ClimateZone::new(
            "5B",
            "Cool-Dry",
            "Cool dry climates with warm summers and cold winters",
            -25.0,
            34.0,
            5.0,
            40.0,
            4500,
            800,
            1700,
            4.0,
            400,
            "commercial",
        ),
    );

    // Zone 6A: Cold-Humid
    zones.insert(
        "6A".to_string(),
        ClimateZone::new(
            "6A",
            "Cold-Humid",
            "Cold humid climates with moderate summers and very cold winters",
            -30.0,
            28.0,
            15.0,
            60.0,
            5500,
            500,
            1200,
            3.0,
            700,
            "residential",
        ),
    );

    // Zone 6B: Cold-Dry
    zones.insert(
        "6B".to_string(),
        ClimateZone::new(
            "6B",
            "Cold-Dry",
            "Cold dry climates with moderate summers and very cold winters",
            -35.0,
            30.0,
            5.0,
            30.0,
            6000,
            300,
            1600,
            3.5,
            300,
            "commercial",
        ),
    );

    // Zone 7: Very Cold
    zones.insert(
        "7".to_string(),
        ClimateZone::new(
            "7",
            "Very Cold",
            "Very cold climates with short, mild summers and very cold winters",
            -40.0,
            25.0,
            10.0,
            40.0,
            7000,
            200,
            1100,
            2.5,
            400,
            "residential",
        ),
    );

    // Zone 8: Subarctic/Arctic
    zones.insert(
        "8".to_string(),
        ClimateZone::new(
            "8",
            "Subarctic/Arctic",
            "Subarctic and arctic climates with very short summers and extremely cold winters",
            -50.0,
            18.0,
            5.0,
            30.0,
            8500,
            50,
            800,
            4.0, // Higher wind speed in arctic areas
            200, // Very low precipitation
            "residential",
        ),
    );

    zones
}

/// Get climate zone by ID
pub fn get_climate_zone(zone_id: &str) -> Option<ClimateZone> {
    get_all_climate_zones().get(zone_id).cloned()
}

/// Get major climate zones (simplified list)
pub fn get_major_climate_zones() -> Vec<String> {
    vec![
        "1A".to_string(), // Very Hot-Humid
        "2B".to_string(), // Hot-Dry
        "3C".to_string(), // Warm-Marine
        "4A".to_string(), // Mixed-Humid
        "5A".to_string(), // Cool-Humid
        "6A".to_string(), // Cold-Humid
        "7".to_string(),  // Very Cold
        "8".to_string(),  // Subarctic/Arctic
    ]
}
