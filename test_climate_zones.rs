use std::collections::HashMap;

// Climate zone definition
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

pub fn get_all_climate_zones() -> HashMap<String, ClimateZone> {
    let mut zones = HashMap::new();

    // Zone 1A: Very Hot-Humid
    zones.insert(
        "1A".to_string(),
        ClimateZone::new(
            "1A",
            "Very Hot-Humid",
            "Tropical humid climates",
            10.0,
            35.0,
            60.0,
            90.0,
            0.0,    // No heating degree days
            3000.0, // High cooling degree days
            1800.0, // High solar radiation
            2.5,    // Low wind speed
            1500.0, // High precipitation
            "residential",
        ),
    );

    zones
}

fn main() {
    println!("Testing climate zone implementation...");

    let zones = get_all_climate_zones();
    println!("Loaded {} climate zones", zones.len());

    if let Some(zone_1a) = zones.get("1A") {
        println!("Zone 1A: {} - {}", zone_1a.full_name, zone_1a.description);
        println!(
            "  Temperature: {:.1}°C to {:.1}°C",
            zone_1a.temperature_range_c.0, zone_1a.temperature_range_c.1
        );
        println!(
            "  Humidity: {:.1}% to {:.1}%",
            zone_1a.humidity_range.0, zone_1a.humidity_range.1
        );
        println!(
            "  HDD: {}, CDD: {}",
            zone_1a.heating_degree_days, zone_1a.cooling_degree_days
        );
        println!(
            "  Solar: {} kWh/m², Wind: {} m/s, Precipitation: {} mm",
            zone_1a.solar_radiation_kwh_m2, zone_1a.wind_speed_m_s, zone_1a.precipitation_mm
        );
        println!("  Typical building type: {}", zone_1a.typical_building_type);
        println!("✅ Climate zone implementation working correctly!");
    } else {
        println!("❌ Failed to load zone 1A");
    }
}
