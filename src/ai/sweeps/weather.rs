//! Weather-file registry for multi-climate Monte Carlo sweeps (Issue #1776).
//!
//! The acceptance criteria require sampling distributions and ranges for
//! "weather files (multi-climate)".  [`WeatherFileRegistry`] maps ASHRAE
//! climate-zone labels (e.g. `"4A"`, `"5B"`) to representative EnergyPlus
//! TMY3 EPW reference files, so that every [`super::manifest::SweepSample`]
//! can be resolved to a concrete weather-file path that the physics solver
//! can load.
//!
//! # Climate-zone coverage
//!
//! The registry ships with the canonical ASHRAE 169 climate-zone set
//! (zones 1–8, moisture designations A/B/C) using well-known EnergyPlus
//! reference cities.  This guarantees full multi-climate coverage from
//! very-hot/moist (1A: Miami) through subarctic (8: Fairbanks).
//!
//! ```text
//!  SurrogateDomain.climate_zones ──► WeatherFileRegistry.lookup(zone)
//!                                           │
//!                                           ▼
//!                                   WeatherFileEntry
//!                                   { location, lat, lon, epw_filename }
//! ```
//!
//! The registry can also be extended or overridden at runtime via
//! [`WeatherFileRegistry::with_entries`] so that bespoke weather libraries
//! (e.g. a project-specific EPW set) can be plugged in.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A representative weather file for a climate zone.
///
/// Contains enough metadata to resolve and load the EPW/TMY3 file and to
/// record provenance in the parameter manifest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WeatherFileEntry {
    /// ASHRAE climate-zone label (e.g. `"4A"`).
    pub climate_zone: String,
    /// Human-readable location name (e.g. `"Baltimore, MD"`).
    pub location_name: String,
    /// Latitude [degrees, −90..90].
    pub latitude: f64,
    /// Longitude [degrees, −180..180].
    pub longitude: f64,
    /// EPW filename (no directory) for the EnergyPlus TMY3 reference file.
    pub epw_filename: String,
}

impl WeatherFileEntry {
    /// Construct a registry entry.
    pub fn new(
        climate_zone: impl Into<String>,
        location_name: impl Into<String>,
        latitude: f64,
        longitude: f64,
        epw_filename: impl Into<String>,
    ) -> Self {
        WeatherFileEntry {
            climate_zone: climate_zone.into(),
            location_name: location_name.into(),
            latitude,
            longitude,
            epw_filename: epw_filename.into(),
        }
    }
}

/// Registry mapping climate-zone labels to representative weather files.
///
/// Construct with [`WeatherFileRegistry::standard`] for the built-in ASHRAE
/// 169 reference set, or [`WeatherFileRegistry::with_entries`] for a custom
/// weather library.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WeatherFileRegistry {
    entries: HashMap<String, WeatherFileEntry>,
}

impl WeatherFileRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        WeatherFileRegistry {
            entries: HashMap::new(),
        }
    }

    /// The standard ASHRAE 169 climate-zone reference registry.
    ///
    /// Covers zones 1–8 with moisture designations A/B/C, using canonical
    /// EnergyPlus TMY3 reference cities.  This is the default registry used
    /// by [`super::config::SweepConfig::from_domain`].
    pub fn standard() -> Self {
        let entries: Vec<WeatherFileEntry> = standard_weather_entries();
        let mut map = HashMap::with_capacity(entries.len());
        for e in entries {
            map.insert(e.climate_zone.clone(), e);
        }
        WeatherFileRegistry { entries: map }
    }

    /// Build a registry from an explicit list of entries.
    ///
    /// Later entries for the same climate zone win.  Useful for overriding
    /// the standard set or adding project-specific weather files.
    pub fn with_entries(entries: impl IntoIterator<Item = WeatherFileEntry>) -> Self {
        let mut map = HashMap::new();
        for e in entries {
            map.insert(e.climate_zone.clone(), e);
        }
        WeatherFileRegistry { entries: map }
    }

    /// Look up the representative weather file for a climate zone.
    pub fn lookup(&self, climate_zone: &str) -> Option<&WeatherFileEntry> {
        self.entries.get(climate_zone)
    }

    /// Insert or replace an entry.
    pub fn insert(&mut self, entry: WeatherFileEntry) {
        self.entries.insert(entry.climate_zone.clone(), entry);
    }

    /// All climate-zone labels known to this registry, sorted.
    pub fn zones(&self) -> Vec<String> {
        let mut z: Vec<String> = self.entries.keys().cloned().collect();
        z.sort();
        z
    }

    /// Number of registered climate zones.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Resolve a weather file, falling back to a default entry for unknown
    /// zones so that sweep generation never stalls on an unmapped zone.
    ///
    /// Returns the matched entry, or `None` if the registry is empty.
    pub fn lookup_or_default(&self, climate_zone: &str) -> Option<&WeatherFileEntry> {
        self.lookup(climate_zone).or_else(|| {
            // Fallback: pick the alphabetically-first zone so the choice is
            // deterministic across runs.
            self.entries.values().min_by_key(|e| &e.climate_zone)
        })
    }
}

/// Canonical ASHRAE 169 climate-zone → EnergyPlus reference-city mapping.
///
/// These are the standard TMY3 reference cities used across BEM tools.
/// Latitudes/longitudes are the airport coordinates of the EPW station.
fn standard_weather_entries() -> Vec<WeatherFileEntry> {
    use WeatherFileEntry as W;
    vec![
        W::new(
            "1A",
            "Miami, FL",
            25.79,
            -80.32,
            "USA_FL_Miami.Intl.AP.722020_TMY3.epw",
        ),
        W::new(
            "2A",
            "Houston, TX",
            29.98,
            -95.36,
            "USA_TX_Houston.Intercontinental.AP.722430_TMY3.epw",
        ),
        W::new(
            "3A",
            "Atlanta, GA",
            33.64,
            -84.44,
            "USA_GA_Atlanta-Hartsfield.Jackson.Intl.AP.722190_TMY3.epw",
        ),
        W::new(
            "3B",
            "Las Vegas, NV",
            36.08,
            -115.16,
            "USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3.epw",
        ),
        W::new(
            "3C",
            "San Francisco, CA",
            37.62,
            -122.38,
            "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
        ),
        W::new(
            "4A",
            "Baltimore, MD",
            39.18,
            -76.67,
            "USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3.epw",
        ),
        W::new(
            "4B",
            "Albuquerque, NM",
            35.04,
            -106.61,
            "USA_NM_Albuquerque.Intl.AP.723650_TMY3.epw",
        ),
        W::new(
            "4C",
            "Seattle, WA",
            47.45,
            -122.30,
            "USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3.epw",
        ),
        W::new(
            "5A",
            "Boston, MA",
            42.36,
            -71.06,
            "USA_MA_Boston-Logan.Intl.AP.725045_TMY3.epw",
        ),
        W::new(
            "5B",
            "Denver, CO",
            39.83,
            -104.65,
            "USA_CO_Denver.Intl.AP.725650_TMY3.epw",
        ),
        W::new(
            "5C",
            "Portland, OR",
            45.59,
            -122.60,
            "USA_OR_Portland.Intl.AP.726980_TMY3.epw",
        ),
        W::new(
            "6A",
            "Minneapolis, MN",
            44.88,
            -93.23,
            "USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3.epw",
        ),
        W::new(
            "6B",
            "Helena, MT",
            46.61,
            -112.01,
            "USA_MT_Helena.Rgnl.AP.727725_TMY3.epw",
        ),
        W::new(
            "7",
            "Duluth, MN",
            46.84,
            -92.19,
            "USA_MN_Duluth.Intl.AP.727450_TMY3.epw",
        ),
        W::new(
            "8",
            "Fairbanks, AK",
            64.80,
            -147.88,
            "USA_AK_Fairbanks.Intl.AP.702610_TMY3.epw",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_registry_covers_default_domain_zones() {
        let reg = WeatherFileRegistry::standard();
        // SurrogateDomain::default_residential uses 4A, 5A, 6A.
        for z in ["4A", "5A", "6A"] {
            assert!(reg.lookup(z).is_some(), "standard registry missing {z}");
        }
    }

    #[test]
    fn test_standard_registry_multi_climate() {
        let reg = WeatherFileRegistry::standard();
        // Multi-climate coverage: hot (1A/2A), temperate (3A/4A), cold (5A/6A),
        // subarctic (7/8).  At least one zone from each band must be present.
        let zones = reg.zones();
        assert!(zones
            .iter()
            .any(|z| z.starts_with('1') || z.starts_with('2')));
        assert!(zones
            .iter()
            .any(|z| z.starts_with('3') || z.starts_with('4')));
        assert!(zones
            .iter()
            .any(|z| z.starts_with('5') || z.starts_with('6')));
        assert!(zones
            .iter()
            .any(|z| z.starts_with('7') || z.starts_with('8')));
        assert!(reg.len() >= 10);
    }

    #[test]
    fn test_lookup_returns_entry() {
        let reg = WeatherFileRegistry::standard();
        let e = reg.lookup("5A").unwrap();
        assert_eq!(e.climate_zone, "5A");
        assert!(e.location_name.contains("Boston"));
        assert!(e.epw_filename.ends_with(".epw"));
        assert!(e.epw_filename.contains("Boston"));
    }

    #[test]
    fn test_lookup_unknown_zone() {
        let reg = WeatherFileRegistry::standard();
        assert!(reg.lookup("99Z").is_none());
    }

    #[test]
    fn test_lookup_or_default_never_panics() {
        let reg = WeatherFileRegistry::standard();
        let e = reg.lookup_or_default("99Z").unwrap();
        assert!(!e.climate_zone.is_empty());
    }

    #[test]
    fn test_empty_registry() {
        let reg = WeatherFileRegistry::new();
        assert!(reg.is_empty());
        assert!(reg.lookup("4A").is_none());
        assert!(reg.lookup_or_default("4A").is_none());
    }

    #[test]
    fn test_with_entries_custom() {
        let reg = WeatherFileRegistry::with_entries([WeatherFileEntry::new(
            "9Z",
            "Test City",
            0.0,
            0.0,
            "test.epw",
        )]);
        assert_eq!(reg.len(), 1);
        let e = reg.lookup("9Z").unwrap();
        assert_eq!(e.location_name, "Test City");
    }

    #[test]
    fn test_insert_overrides() {
        let mut reg = WeatherFileRegistry::standard();
        let original = reg.lookup("5A").unwrap().clone();
        reg.insert(WeatherFileEntry::new(
            "5A",
            "Override",
            0.0,
            0.0,
            "override.epw",
        ));
        let e = reg.lookup("5A").unwrap();
        assert_eq!(e.location_name, "Override");
        assert_ne!(e.location_name, original.location_name);
    }

    #[test]
    fn test_zones_sorted() {
        let reg = WeatherFileRegistry::standard();
        let zones = reg.zones();
        let mut sorted = zones.clone();
        sorted.sort();
        assert_eq!(zones, sorted);
    }

    #[test]
    fn test_all_epw_filenames_valid() {
        let reg = WeatherFileRegistry::standard();
        for z in reg.zones() {
            let e = reg.lookup(&z).unwrap();
            assert!(e.epw_filename.ends_with(".epw"), "{} bad filename", z);
            assert!(!e.epw_filename.is_empty());
            assert!(e.latitude >= -90.0 && e.latitude <= 90.0, "{} bad lat", z);
            assert!(
                e.longitude >= -180.0 && e.longitude <= 180.0,
                "{} bad lon",
                z
            );
        }
    }

    #[test]
    fn test_entry_equality() {
        let a = WeatherFileEntry::new("4A", "Baltimore, MD", 39.18, -76.67, "balt.epw");
        let b = WeatherFileEntry::new("4A", "Baltimore, MD", 39.18, -76.67, "balt.epw");
        assert_eq!(a, b);
    }
}
