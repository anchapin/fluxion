//! Cross-validation reference data catalog (Issue #1933).
//!
//! A unified catalog that enumerates every reference dataset under
//! `tests/reference_data/`, classifies it by category, parses the
//! provenance headers (the leading `#` comment block shared by every CSV
//! in the tree), and computes a coverage report. The catalog is the
//! single source of truth for "what reference data exists and where",
//! replacing the ad-hoc per-test hand-rolled loaders.
//!
//! # Design
//!
//! - **Discovery** is a recursive directory walk over `tests/reference_data/`
//!   (no extra crate dependency — `std::fs::read_dir` only).
//! - **Categorisation** maps each top-level subdirectory to a
//!   [`ReferenceCategory`].
//! - **Provenance** is extracted from the leading `#`-prefixed lines of
//!   each file (the convention documented in
//!   `tests/reference_data/README.md`). Non-`#` files keep an empty
//!   provenance block.
//! - **Integrity** is provided by a SHA-256 content hash, computed
//!   lazily via the existing `sha2` dependency.
//!
//! The catalog is intentionally read-only and side-effect free: it never
//! writes or modifies reference data. Tests that need the actual numeric
//! content still load it themselves with the existing CSV helpers in
//! [`crate::validation::reference_data`] — the catalog is the *index*,
//! not the loader.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Root directory of the reference data tree (relative to the cargo
/// manifest / test working directory).
pub const REFERENCE_DATA_ROOT: &str = "tests/reference_data";

/// Category of reference data. Determined from the top-level subdirectory
/// of `tests/reference_data/`.
///
/// New categories added by Issue #1933:
/// - [`ReferenceCategory::Equipment`] — HVAC equipment performance curves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ReferenceCategory {
    /// `weather/` — TMY3 hourly weather CSVs.
    Weather,
    /// `solar/` — solar position and surface irradiance CSVs.
    Solar,
    /// `conduction/` — EnergyPlus step-response CSVs per construction.
    Conduction,
    /// `ventilation/` — infiltration / ventilation conductance CSVs.
    Ventilation,
    /// `zone_balance/` — annual + peak energy reference for ASHRAE 140 cases.
    ZoneBalance,
    /// `ashrae140/` — ASHRAE 140 monthly and ancillary reference.
    Ashrae140,
    /// `energyplus_models/` — EnergyPlus IDF / epJSON source models.
    EnergyPlusModels,
    /// `equipment/` — HVAC equipment performance curves (Issue #1933).
    Equipment,
    /// `gauge/` — diagnostic gauge reference data.
    Gauge,
    /// Uncategorised (files directly under the root, e.g. `README.md`).
    Other,
}

impl ReferenceCategory {
    /// Map a top-level subdirectory name to a category.
    pub fn from_dir_name(name: &str) -> Self {
        match name {
            "weather" => Self::Weather,
            "solar" => Self::Solar,
            "conduction" => Self::Conduction,
            "ventilation" => Self::Ventilation,
            "zone_balance" => Self::ZoneBalance,
            "ashrae140" => Self::Ashrae140,
            "energyplus_models" => Self::EnergyPlusModels,
            "equipment" => Self::Equipment,
            "gauge" => Self::Gauge,
            _ => Self::Other,
        }
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Weather => "weather",
            Self::Solar => "solar",
            Self::Conduction => "conduction",
            Self::Ventilation => "ventilation",
            Self::ZoneBalance => "zone_balance",
            Self::Ashrae140 => "ashrae140",
            Self::EnergyPlusModels => "energyplus_models",
            Self::Equipment => "equipment",
            Self::Gauge => "gauge",
            Self::Other => "other",
        }
    }
}

/// How a reference dataset was generated. Parsed from the leading
/// `# Status:` header line when present.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReferenceSource {
    /// Direct EnergyPlus 25.2.0 simulation output.
    EnergyPlus,
    /// Analytically derived from published engineering correlations
    /// (AHRI / ASHRAE / EnergyPlus TSD coefficients). Issue #1933.
    Analytical,
    /// Interim / placeholder pending direct simulation output.
    Interim,
    /// Authoritative published reference band (ASHRAE 140-2023 Annex B).
    Authoritative,
    /// Provenance not stated in the file header.
    Unknown,
}

impl ReferenceSource {
    /// Parse the `Status:` tag from a provenance header block.
    ///
    /// Recognises the keywords used by the existing CSVs:
    /// `EnergyPlus`, `ANALYTICAL`, `INTERIM`, `Authoritative`.
    pub fn from_provenance(provenance: &str) -> Self {
        // Look for a "Status:" line first; fall back to keyword scan.
        for line in provenance.lines() {
            let lower = line.to_ascii_lowercase();
            if lower.contains("status:") {
                if lower.contains("analytical") {
                    return Self::Analytical;
                }
                if lower.contains("interim") || lower.contains("placeholder") {
                    return Self::Interim;
                }
                if lower.contains("energyplus") {
                    return Self::EnergyPlus;
                }
                if lower.contains("authoritative") {
                    return Self::Authoritative;
                }
            }
        }
        // Fallback: keyword scan over the whole header.
        let lower = provenance.to_ascii_lowercase();
        if lower.contains("energyplus") || lower.contains("e+ 25.") || lower.contains("epw:") {
            Self::EnergyPlus
        } else if lower.contains("analytical") {
            Self::Analytical
        } else if lower.contains("interim") || lower.contains("placeholder") {
            Self::Interim
        } else if lower.contains("ashrae 140-2023 annex") || lower.contains("authoritative") {
            Self::Authoritative
        } else {
            Self::Unknown
        }
    }
}

/// A single reference dataset entry.
#[derive(Debug, Clone)]
pub struct ReferenceEntry {
    /// Category derived from the top-level subdirectory.
    pub category: ReferenceCategory,
    /// File name (e.g. `case_600_energy_reference.csv`).
    pub file_name: String,
    /// Path relative to the cargo manifest root.
    pub relative_path: PathBuf,
    /// Absolute path on disk.
    pub absolute_path: PathBuf,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Parsed leading `#`-comment provenance block (may be empty).
    pub provenance: String,
    /// Detected generation source.
    pub source: ReferenceSource,
    /// SHA-256 content hash (hex). Computed lazily by [`Self::hash`].
    pub sha256: Option<String>,
}

impl ReferenceEntry {
    /// Compute and cache the SHA-256 hash of the file contents.
    pub fn hash(&mut self) -> Result<&str, std::io::Error> {
        if let Some(ref h) = self.sha256 {
            return Ok(h.as_str());
        }
        let bytes = fs::read(&self.absolute_path)?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        self.sha256 = Some(digest);
        Ok(self.sha256.as_ref().expect("just-set").as_str())
    }

    /// Number of data rows in a CSV (non-blank, non-`#` lines after the
    /// header). For non-CSV files this returns `None`.
    pub fn csv_data_rows(&self) -> Option<usize> {
        if self.file_name.ends_with(".csv") {
            let content = fs::read_to_string(&self.absolute_path).ok()?;
            let mut rows = 0usize;
            let mut seen_header = false;
            for line in content.lines() {
                let t = line.trim();
                if t.is_empty() || t.starts_with('#') {
                    continue;
                }
                if !seen_header {
                    seen_header = true; // first non-comment line is the header
                    continue;
                }
                rows += 1;
            }
            Some(rows)
        } else {
            None
        }
    }
}

/// The full catalog of reference datasets.
#[derive(Debug, Clone, Default)]
pub struct ReferenceCatalog {
    entries: Vec<ReferenceEntry>,
}

/// Error returned by catalog discovery.
#[derive(Debug)]
pub enum CatalogError {
    /// The reference data root does not exist.
    RootNotFound(PathBuf),
    /// A file could not be read.
    IoError(PathBuf, std::io::Error),
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CatalogError::RootNotFound(p) => {
                write!(f, "Reference data root not found: {}", p.display())
            }
            CatalogError::IoError(p, e) => {
                write!(f, "IO error reading {}: {}", p.display(), e)
            }
        }
    }
}

impl std::error::Error for CatalogError {}

impl ReferenceCatalog {
    /// Discover the catalog by walking the reference data root.
    ///
    /// `root` defaults to [`REFERENCE_DATA_ROOT`] when called via
    /// [`Self::discover_default`].
    pub fn discover(root: impl AsRef<Path>) -> Result<Self, CatalogError> {
        let root = root.as_ref();
        if !root.exists() {
            return Err(CatalogError::RootNotFound(root.to_path_buf()));
        }
        let mut entries = Vec::new();
        walk_reference_tree(root, root, &mut entries)?;
        // Stable ordering: category, then relative path.
        entries.sort_by(|a, b| {
            a.category
                .cmp(&b.category)
                .then_with(|| a.relative_path.cmp(&b.relative_path))
        });
        Ok(Self { entries })
    }

    /// Discover using the default root ([`REFERENCE_DATA_ROOT`]).
    pub fn discover_default() -> Result<Self, CatalogError> {
        Self::discover(REFERENCE_DATA_ROOT)
    }

    /// All entries, in stable sorted order.
    pub fn entries(&self) -> &[ReferenceEntry] {
        &self.entries
    }

    /// Mutable access to all entries (e.g. to populate the cached hash).
    pub fn entries_mut(&mut self) -> &mut [ReferenceEntry] {
        &mut self.entries
    }

    /// Entries matching a category.
    pub fn by_category(
        &self,
        category: ReferenceCategory,
    ) -> impl Iterator<Item = &ReferenceEntry> {
        self.entries.iter().filter(move |e| e.category == category)
    }

    /// Look up an entry by relative path substring (e.g. `"case_600"`).
    pub fn find(&self, name_substring: &str) -> Option<&ReferenceEntry> {
        self.entries
            .iter()
            .find(|e| e.relative_path.to_string_lossy().contains(name_substring))
    }

    /// Group entries by category, returning a sorted map.
    pub fn grouped(&self) -> BTreeMap<ReferenceCategory, Vec<&ReferenceEntry>> {
        let mut map: BTreeMap<ReferenceCategory, Vec<&ReferenceEntry>> = BTreeMap::new();
        for e in &self.entries {
            map.entry(e.category).or_default().push(e);
        }
        map
    }

    /// Compute a coverage report summarising the catalog.
    pub fn coverage_report(&self) -> CoverageReport {
        let mut report = CoverageReport::default();
        report.total_entries = self.entries.len();
        for e in &self.entries {
            *report.by_category.entry(e.category).or_insert(0) += 1;
            *report.by_source.entry(e.source).or_insert(0) += 1;
        }
        report.equipment_entries_present = self
            .by_category(ReferenceCategory::Equipment)
            .filter(|e| e.file_name.ends_with(".csv"))
            .count();
        report.ashrae140_case_files = self
            .by_category(ReferenceCategory::ZoneBalance)
            .filter(|e| e.file_name.contains("case_") && e.file_name.ends_with("_reference.csv"))
            .count();
        report
    }
}

/// Walk the reference tree recursively, populating `entries`.
fn walk_reference_tree(
    root: &Path,
    current: &Path,
    entries: &mut Vec<ReferenceEntry>,
) -> Result<(), CatalogError> {
    let rd = match fs::read_dir(current) {
        Ok(rd) => rd,
        Err(e) => return Err(CatalogError::IoError(current.to_path_buf(), e)),
    };
    for dirent in rd.flatten() {
        let path = dirent.path();
        let file_type = match dirent.file_type() {
            Ok(t) => t,
            Err(e) => return Err(CatalogError::IoError(path.clone(), e)),
        };
        if file_type.is_dir() {
            walk_reference_tree(root, &path, entries)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|_| path.clone());
        // Category from the FIRST path component (top-level subdir).
        let category = rel
            .components()
            .next()
            .and_then(|c| c.as_os_str().to_str())
            .map(ReferenceCategory::from_dir_name)
            .unwrap_or(ReferenceCategory::Other);
        let meta = dirent
            .metadata()
            .map_err(|e| CatalogError::IoError(path.clone(), e))?;
        let provenance = read_provenance_header(&path).unwrap_or_default();
        let source = ReferenceSource::from_provenance(&provenance);
        entries.push(ReferenceEntry {
            category,
            file_name: dirent.file_name().to_string_lossy().into_owned(),
            relative_path: rel,
            absolute_path: path,
            size_bytes: meta.len(),
            provenance,
            source,
            sha256: None,
        });
    }
    Ok(())
}

/// Read the leading `#`-prefixed lines of a file as the provenance block.
fn read_provenance_header(path: &Path) -> Result<String, std::io::Error> {
    let content = fs::read_to_string(path)?;
    let mut block = String::new();
    for line in content.lines() {
        let t = line.trim_start();
        if t.starts_with('#') {
            // Keep the line without the leading '#' for readability.
            let stripped = t.trim_start_matches('#').trim_start();
            if !block.is_empty() {
                block.push('\n');
            }
            block.push_str(stripped);
        } else if t.is_empty() {
            // Blank lines inside the leading comment block are preserved.
            if block.is_empty() {
                continue;
            }
        } else {
            // First non-comment, non-blank line ends the provenance block.
            break;
        }
    }
    Ok(block)
}

/// Summary coverage statistics over a [`ReferenceCatalog`].
#[derive(Debug, Clone, Default)]
pub struct CoverageReport {
    /// Total number of discovered entries (all file types).
    pub total_entries: usize,
    /// Entry count per category.
    pub by_category: BTreeMap<ReferenceCategory, usize>,
    /// Entry count per detected source.
    pub by_source: BTreeMap<ReferenceSource, usize>,
    /// Number of CSV datasets in the Equipment category (Issue #1933
    /// coverage indicator).
    pub equipment_entries_present: usize,
    /// Number of `case_*_reference.csv` annual-energy files under
    /// `zone_balance/` (one per covered ASHRAE 140 case).
    pub ashrae140_case_files: usize,
}

impl CoverageReport {
    /// True if the equipment reference category has any CSV datasets.
    pub fn has_equipment_data(&self) -> bool {
        self.equipment_entries_present > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_from_dir_name_roundtrip() {
        assert_eq!(
            ReferenceCategory::from_dir_name("equipment"),
            ReferenceCategory::Equipment
        );
        assert_eq!(
            ReferenceCategory::from_dir_name("weather"),
            ReferenceCategory::Weather
        );
        assert_eq!(
            ReferenceCategory::from_dir_name("nope"),
            ReferenceCategory::Other
        );
        assert_eq!(ReferenceCategory::Equipment.label(), "equipment");
    }

    #[test]
    fn source_from_provenance_keywords() {
        assert_eq!(
            ReferenceSource::from_provenance(
                "Status: ANALYTICAL — derived from published fan laws"
            ),
            ReferenceSource::Analytical
        );
        assert_eq!(
            ReferenceSource::from_provenance("STATUS: INTERIM / PLACEHOLDER"),
            ReferenceSource::Interim
        );
        assert_eq!(
            ReferenceSource::from_provenance("EnergyPlus Version: 25.2.0\nEPW: x.epw"),
            ReferenceSource::EnergyPlus
        );
        assert_eq!(
            ReferenceSource::from_provenance("Source: ASHRAE 140-2023 Annex B authoritative"),
            ReferenceSource::Authoritative
        );
        assert_eq!(
            ReferenceSource::from_provenance(""),
            ReferenceSource::Unknown
        );
    }

    #[test]
    fn catalog_discover_returns_entries_or_missing_root() {
        match ReferenceCatalog::discover_default() {
            Ok(cat) => {
                assert!(
                    !cat.entries().is_empty(),
                    "reference data tree should not be empty"
                );
                let report = cat.coverage_report();
                assert!(report.total_entries > 0);
                assert!(report.by_category.contains_key(&ReferenceCategory::Weather));
                assert!(report
                    .by_category
                    .contains_key(&ReferenceCategory::Equipment));
            }
            Err(CatalogError::RootNotFound(_)) => {
                // Tests run from the crate root where tests/reference_data exists,
                // so this branch is only hit if invoked from elsewhere.
            }
            Err(e) => panic!("unexpected catalog error: {e}"),
        }
    }
}
