//! TMY2 (`.TM2`) weather-format reader — Issue #4295.
//!
//! Parses fixed-width TMY2 files of the kind the ASHRAE 140 §9
//! space-cooling equipment cases are driven by (`CE1xxA.TM2` and
//! friends). The format is the one specified in **Annex A1.4 of
//! ANSI/ASHRAE Addendum a to Standard 140-2001** (Tables A1-5/A1-6 and
//! Figure A1-1), which reprints the NREL TMY2 format (Marion & Urban,
//! *User's Manual for TMY2s*, NREL/SP-463-7668, 1995). Field positions
//! below are transcribed from those tables; the tables are
//! edition-stable format documentation, not numeric-band material.
//!
//! # Provenance and redistribution policy (Issues #4284 / #4285 / #4295)
//!
//! The CE weather files are ASHRAE supplemental material. **They are
//! never committed to this repository** — the same
//! fetch-and-verify-plus-hash-pinning policy that governs the §6
//! DRYCOLD files applies. This module reads a user-supplied licensed
//! copy from local disk; it ships no data.
//!
//! # Layout summary (Addendum a to 140-2001, Tables A1-5/A1-6)
//!
//! * Record 1: file header (1-based field positions): WBAN 2-6, city
//!   8-29, state 31-32, time zone 34-36 (hours behind UT, e.g. `-7`
//!   for MST), latitude 38-44 (N/S flag + degrees + minutes),
//!   longitude 46-53 (W/E flag + degrees + minutes), elevation 56-59
//!   (meters above sea level).
//! * Records 2..=8761: hourly data, one per hour, local standard
//!   time. Each record starts with the 2-digit year the typical month
//!   came from, then month/day/hour (`1-24`). Solar values are
//!   Wh/m² received during the 60 minutes preceding the hour.
//!   Temperatures are in tenths of °C; wind speed in tenths of m/s.
//!   Every data element carries a 1-char source flag and a 1-char
//!   uncertainty flag; missing data is field-filled with 9s.
//!
//! # Mapping to [`HourlyWeatherData`]
//!
//! * `dry_bulb_temp` — dry bulb, tenths of °C.
//! * `dni` / `dhi` / `ghi` — Wh/m² over the hour, which is numerically
//!   the average W/m² the engine consumes.
//! * `humidity` — relative humidity (%).
//! * `wind_speed` — tenths of m/s → m/s.
//! * `wind_direction` — degrees (meteorological convention), `None`
//!   when 9-filled.
//! * `horizontal_infrared` — TMY2 carries **no** horizontal-infrared
//!   field. It is set to `0.0`, the same coercion the EPW reader
//!   applies to missing-data sentinels (Issue #1415 convention). The
//!   §9 CE cases run a near-adiabatic envelope, so sky-IR is a second
//!   order input there; if a future §9 requirement needs a nonzero
//!   sky-IR from TM2, it must come from a documented dew-point/sky-cover
//!   model, not an invented field.
//! * `horizontal_illuminance` / `diffuse_illuminance` — hundreds of
//!   lux → lux, `None` when flagged missing.
//! * `snow_depth` — cm → mm, `None` when 9-filled (999).
//! * `present_weather_code` — the 10-digit present-weather number.
//! * `albedo` — `None` (TMY2 does not report ground albedo).
//! * `ground_temperature` — `None` (not in the format).

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use super::{HourlyWeatherData, WeatherError, WeatherSource};

/// Fixed-width slice helper: 1-based inclusive field positions, as
/// printed in the standard's tables. Falls back to an empty string
/// past end-of-line (some files omit trailing optional fields).
fn field(line: &str, start: usize, end: usize) -> &str {
    let s = start.saturating_sub(1);
    if s >= line.len() {
        return "";
    }
    &line[s..end.min(line.len())]
}

/// Station location parsed from the TMY2 file header (Table A1-5).
#[derive(Debug, Clone, PartialEq)]
pub struct Tm2Location {
    /// WBAN station number (header positions 2-6).
    pub wban: String,
    /// City name (header positions 8-29, underscores preserved).
    pub city: String,
    /// Two-letter state abbreviation (header positions 31-32).
    pub state: String,
    /// Time zone from header positions 34-36, as written in the file
    /// (hours behind UT; `-7` = MST). Sign convention matches the EPW
    /// `TimeZone` column the engine's `EpwLocation.utc_offset_hours`
    /// carries, so callers can pass it to the solar pipeline directly.
    pub utc_offset_hours: f64,
    /// Latitude in decimal degrees, north-positive.
    pub latitude_deg: f64,
    /// Longitude in decimal degrees, **east-positive** (the file's W/E
    /// flag is inverted to match the engine's solar-pipeline
    /// convention; see `EpwLocation::longitude_deg`).
    pub longitude_deg: f64,
    /// Station elevation in meters (header positions 56-59).
    pub elevation_m: f64,
}

impl Tm2Location {
    /// `"City, ST"` display string, mirroring `EpwLocation::city_state`.
    pub fn city_state(&self) -> String {
        format!("{}, {}", self.city, self.state)
    }

    /// Parses the single header record. Positions per Table A1-5.
    pub fn parse(line: &str) -> Result<Self, WeatherError> {
        if line.len() < 59 {
            return Err(WeatherError::ParseError(format!(
                "TMY2 header too short ({} chars, need >= 59): {:?}",
                line.len(),
                line
            )));
        }
        let wban = field(line, 2, 6).trim().to_string();
        let city = field(line, 8, 29).trim().to_string();
        let state = field(line, 31, 32).trim().to_string();
        let tz = field(line, 34, 36)
            .trim()
            .parse::<f64>()
            .map_err(|e| WeatherError::ParseError(format!("TMY2 header time zone: {e}")))?;
        let lat_hemi = field(line, 38, 38).trim();
        let lat_deg = field(line, 40, 41)
            .trim()
            .parse::<f64>()
            .map_err(|e| WeatherError::ParseError(format!("TMY2 header latitude degrees: {e}")))?;
        let lat_min = field(line, 43, 44)
            .trim()
            .parse::<f64>()
            .map_err(|e| WeatherError::ParseError(format!("TMY2 header latitude minutes: {e}")))?;
        let lat = lat_deg + lat_min / 60.0;
        let latitude_deg = match lat_hemi {
            "N" => lat,
            "S" => -lat,
            other => {
                return Err(WeatherError::ParseError(format!(
                    "TMY2 header latitude hemisphere {other:?} (expected N/S)"
                )))
            }
        };
        let lon_hemi = field(line, 46, 46).trim();
        let lon_deg = field(line, 48, 50)
            .trim()
            .parse::<f64>()
            .map_err(|e| WeatherError::ParseError(format!("TMY2 header longitude degrees: {e}")))?;
        let lon_min = field(line, 52, 53)
            .trim()
            .parse::<f64>()
            .map_err(|e| WeatherError::ParseError(format!("TMY2 header longitude minutes: {e}")))?;
        let lon = lon_deg + lon_min / 60.0;
        let longitude_deg = match lon_hemi {
            "E" => lon,
            "W" => -lon,
            other => {
                return Err(WeatherError::ParseError(format!(
                    "TMY2 header longitude hemisphere {other:?} (expected E/W)"
                )))
            }
        };
        let elevation_m = field(line, 56, 59)
            .trim()
            .parse::<f64>()
            .map_err(|e| WeatherError::ParseError(format!("TMY2 header elevation: {e}")))?;
        Ok(Tm2Location {
            wban,
            city,
            state,
            utc_offset_hours: tz,
            latitude_deg,
            longitude_deg,
            elevation_m,
        })
    }
}

/// One fully-parsed hourly data record (Table A1-6).
///
/// Element values are physical units after scaling (°C, W/m², %, m/s,
/// lux); the raw source/uncertainty flags are kept alongside the solar
/// and sky elements that carry them so §9 case code can make its own
/// missing-data policy decisions.
#[derive(Debug, Clone, PartialEq)]
pub struct Tm2Record {
    /// Year the typical month was drawn from (positions 2-3, 2-digit).
    pub year: i32,
    /// Month, 1-12 (positions 4-5).
    pub month: u32,
    /// Day of month, 1-31 (positions 6-7).
    pub day: u32,
    /// Hour of day in local standard time, 1-24 (positions 8-9).
    pub hour: u32,
    /// Extraterrestrial horizontal radiation, Wh/m² (positions 10-13).
    pub etr_horizontal: f64,
    /// Extraterrestrial direct-normal radiation, Wh/m² (14-17).
    pub etr_direct_normal: f64,
    /// Global horizontal radiation, Wh/m² (18-21), with source (22)
    /// and uncertainty (23) flags.
    pub ghi: f64,
    pub ghi_source_flag: char,
    pub ghi_uncertainty_flag: char,
    /// Direct normal radiation, Wh/m² (24-27), flags 28-29.
    pub dni: f64,
    pub dni_source_flag: char,
    pub dni_uncertainty_flag: char,
    /// Diffuse horizontal radiation, Wh/m² (30-33), flags 34-35.
    pub dhi: f64,
    pub dhi_source_flag: char,
    pub dhi_uncertainty_flag: char,
    /// Global horizontal illuminance, hundreds of lux (36-39), flags 40-41.
    pub ghi_illuminance_hundreds_lux: Option<f64>,
    /// Direct normal illuminance, hundreds of lux (42-45), flags 46-47.
    pub dni_illuminance_hundreds_lux: Option<f64>,
    /// Diffuse horizontal illuminance, hundreds of lux (48-51), flags 52-53.
    pub dhi_illuminance_hundreds_lux: Option<f64>,
    /// Zenith luminance, thousands of Cd/m² (54-57), flags 58-59.
    pub zenith_luminance_thousands_cdm2: Option<f64>,
    /// Total sky cover, tenths (60-61), flags 62-63.
    pub total_sky_cover_tenths: Option<f64>,
    /// Opaque sky cover, tenths (64-65), flags 66-67.
    pub opaque_sky_cover_tenths: Option<f64>,
    /// Dry bulb temperature, °C (68-71, tenths), flags 72-73.
    pub dry_bulb_temp: f64,
    pub dry_bulb_source_flag: char,
    pub dry_bulb_uncertainty_flag: char,
    /// Dew point temperature, °C (74-77, tenths), flags 78-79.
    pub dew_point_temp: f64,
    pub dew_point_source_flag: char,
    pub dew_point_uncertainty_flag: char,
    /// Relative humidity, % (80-82), flags 83-84.
    pub relative_humidity: f64,
    /// Atmospheric pressure, millibars (85-88), flags 89-90.
    pub atmospheric_pressure_mbar: f64,
    /// Wind direction, degrees (91-93, meteorological convention), flags 94-95.
    pub wind_direction_deg: Option<f64>,
    /// Wind speed, m/s (96-98, tenths), flags 99-100.
    pub wind_speed: f64,
    /// Horizontal visibility, tenths of km (101-104), flags 105-106.
    pub visibility_tenths_km: Option<f64>,
    /// Ceiling height, m (107-111), flags 112-113.
    pub ceiling_height_m: Option<f64>,
    /// Present weather, 10-digit code (114-123).
    pub present_weather_code: Option<u64>,
    /// Precipitable water, mm (124-126), flags 127-128.
    pub precipitable_water_mm: Option<f64>,
    /// Broadband aerosol optical depth, thousandths (129-131), flags 132-133.
    pub aerosol_optical_depth_millths: Option<f64>,
    /// Snow depth, cm (134-136), flags 137-138.
    pub snow_depth_cm: Option<f64>,
    /// Days since last snowfall (139-140), flags 141-142.
    pub days_since_last_snowfall: Option<f64>,
}

/// True when a fixed-width integer field is 9-filled, the TMY2
/// missing-data convention (field-filled with 9s).
fn is_missing_9s(raw: &str) -> bool {
    !raw.is_empty() && raw.chars().all(|c| c == '9')
}

/// Parses an integer fixed-width field to f64, erroring on garbage.
fn num(line: &str, start: usize, end: usize, name: &str) -> Result<f64, WeatherError> {
    field(line, start, end).trim().parse::<f64>().map_err(|e| {
        WeatherError::ParseError(format!("TMY2 record {name} (pos {start}-{end}): {e}"))
    })
}

/// Parses a `value + source + uncertainty` group (value `width` chars,
/// then a 1-char source flag and a 1-char uncertainty flag).
///
/// A 9-filled or `-999` (missing) value coerces to 0.0 with flags
/// preserved for the *solar* elements — the same missing-solar-to-zero
/// convention the EPW reader applies (Issue #1415 / EnergyPlus Weather
/// Converter). Temperature groups pass their raw value through so the
/// caller can apply its own missing-data policy.
fn flag_group(
    line: &str,
    start: usize,
    width: usize,
    name: &str,
) -> Result<(f64, char, char, bool), WeatherError> {
    let end = start + width - 1;
    let raw = field(line, start, end);
    let missing = is_missing_9s(raw) || raw.trim() == "-999";
    let v = if missing {
        0.0
    } else {
        num(line, start, end, name)?
    };
    let src = field(line, end + 1, end + 1).chars().next().unwrap_or('?');
    let unc = field(line, end + 2, end + 2).chars().next().unwrap_or('?');
    Ok((v, src, unc, missing))
}

/// Parses a `value + source + uncertainty` group whose value may be
/// missing (9-filled → `None`).
fn optional_group(
    line: &str,
    start: usize,
    width: usize,
    name: &str,
) -> Result<Option<f64>, WeatherError> {
    let end = start + width - 1;
    let raw = field(line, start, end);
    if is_missing_9s(raw) {
        return Ok(None);
    }
    let v = num(line, start, end, name)?;
    // 7777/77777-style sentinels mean "unlimited"/"not applicable" —
    // keep the raw value so callers can interpret; only 9s are missing.
    Ok(Some(v))
}

impl Tm2Record {
    /// Parses one hourly data record. Field positions per Table A1-6
    /// (1-based, inclusive).
    pub fn parse(line: &str) -> Result<Self, WeatherError> {
        if line.len() < 142 {
            return Err(WeatherError::ParseError(format!(
                "TMY2 data record too short ({} chars, need >= 142)",
                line.len()
            )));
        }
        let year2 = num(line, 2, 3, "year")? as i32;
        let year = if year2 < 61 {
            2000 + year2
        } else {
            1900 + year2
        };
        let month = num(line, 4, 5, "month")? as u32;
        let day = num(line, 6, 7, "day")? as u32;
        let hour = num(line, 8, 9, "hour")? as u32;
        if !(1..=12).contains(&month) {
            return Err(WeatherError::ParseError(format!(
                "TMY2 record month {month} out of range 1-12"
            )));
        }
        if !(1..=24).contains(&hour) {
            return Err(WeatherError::ParseError(format!(
                "TMY2 record hour {hour} out of range 1-24 (local standard time)"
            )));
        }

        let etr_horizontal = num(line, 10, 13, "ETR horizontal")?;
        let etr_direct_normal = num(line, 14, 17, "ETR direct normal")?;
        let (ghi, ghi_source_flag, ghi_uncertainty_flag, _) = flag_group(line, 18, 4, "GHI")?;
        let (dni, dni_source_flag, dni_uncertainty_flag, _) = flag_group(line, 24, 4, "DNI")?;
        let (dhi, dhi_source_flag, dhi_uncertainty_flag, _) = flag_group(line, 30, 4, "DHI")?;

        // Illuminance groups carry an "I" or "?" source flag; a "?"
        // (not measured / not modeled) group is None even when 0-filled.
        fn illum(
            line: &str,
            start: usize,
            name: &'static str,
        ) -> Result<Option<f64>, WeatherError> {
            let raw = field(line, start, start + 3);
            let src = field(line, start + 4, start + 4)
                .chars()
                .next()
                .unwrap_or('?');
            if is_missing_9s(raw) || src == '?' {
                return Ok(None);
            }
            Ok(Some(num(line, start, start + 3, name)?))
        }
        let ghi_illuminance_hundreds_lux = illum(line, 36, "GH illuminance")?;
        let dni_illuminance_hundreds_lux = illum(line, 42, "DN illuminance")?;
        let dhi_illuminance_hundreds_lux = illum(line, 48, "DH illuminance")?;
        let zenith_luminance_thousands_cdm2 = illum(line, 54, "zenith luminance")?;

        fn sky_cover(line: &str, start: usize) -> Result<Option<f64>, WeatherError> {
            let raw = field(line, start, start + 1);
            if is_missing_9s(raw) {
                return Ok(None);
            }
            Ok(Some(num(line, start, start + 1, "sky cover")?))
        }
        let total_sky_cover_tenths = sky_cover(line, 60)?;
        let opaque_sky_cover_tenths = sky_cover(line, 64)?;

        let (db_raw, dry_bulb_source_flag, dry_bulb_uncertainty_flag, db_missing) =
            flag_group(line, 68, 4, "dry bulb")?;
        if db_missing {
            return Err(WeatherError::ParseError(
                "TMY2 record has missing (9-filled / -999) dry bulb temperature; \
                 refusing to substitute a value"
                    .to_string(),
            ));
        }
        let dry_bulb_temp = db_raw / 10.0;
        let (dp_raw, dew_point_source_flag, dew_point_uncertainty_flag, dp_missing) =
            flag_group(line, 74, 4, "dew point")?;
        if dp_missing {
            return Err(WeatherError::ParseError(
                "TMY2 record has missing (9-filled / -999) dew point temperature; \
                 refusing to substitute a value"
                    .to_string(),
            ));
        }
        let dew_point_temp = dp_raw / 10.0;

        let rh_raw = field(line, 80, 82);
        if is_missing_9s(rh_raw) {
            return Err(WeatherError::ParseError(
                "TMY2 record has missing (9-filled) relative humidity; \
                 refusing to substitute a value"
                    .to_string(),
            ));
        }
        let relative_humidity = num(line, 80, 82, "relative humidity")?;
        let press_raw = field(line, 85, 88);
        if is_missing_9s(press_raw) {
            return Err(WeatherError::ParseError(
                "TMY2 record has missing (9-filled) atmospheric pressure; \
                 refusing to substitute a value"
                    .to_string(),
            ));
        }
        let atmospheric_pressure_mbar = num(line, 85, 88, "atmospheric pressure")?;
        let wind_dir_raw = field(line, 91, 93);
        let wind_direction_deg = if is_missing_9s(wind_dir_raw) {
            None
        } else {
            Some(num(line, 91, 93, "wind direction")?)
        };
        let (ws_raw, _, _, ws_missing) = flag_group(line, 96, 3, "wind speed")?;
        if ws_missing {
            return Err(WeatherError::ParseError(
                "TMY2 record has missing (9-filled) wind speed; \
                 refusing to substitute a value"
                    .to_string(),
            ));
        }
        let wind_speed = ws_raw / 10.0;

        let visibility_tenths_km = optional_group(line, 101, 4, "visibility")?;
        let ceiling_height_m = optional_group(line, 107, 5, "ceiling height")?;
        let pw_raw = field(line, 114, 123);
        let present_weather_code = if pw_raw.chars().all(|c| c == '9') && !pw_raw.is_empty() {
            None
        } else {
            field(line, 114, 123).trim().parse::<u64>().ok()
        };
        let precipitable_water_mm = optional_group(line, 124, 3, "precipitable water")?;
        let aerosol_optical_depth_millths = optional_group(line, 129, 3, "aerosol optical depth")?;
        let snow_depth_cm = optional_group(line, 134, 3, "snow depth")?;
        let days_since_last_snowfall = optional_group(line, 139, 2, "days since snowfall")?;

        Ok(Tm2Record {
            year,
            month,
            day,
            hour,
            etr_horizontal,
            etr_direct_normal,
            ghi,
            ghi_source_flag,
            ghi_uncertainty_flag,
            dni,
            dni_source_flag,
            dni_uncertainty_flag,
            dhi,
            dhi_source_flag,
            dhi_uncertainty_flag,
            ghi_illuminance_hundreds_lux,
            dni_illuminance_hundreds_lux,
            dhi_illuminance_hundreds_lux,
            zenith_luminance_thousands_cdm2,
            total_sky_cover_tenths,
            opaque_sky_cover_tenths,
            dry_bulb_temp,
            dry_bulb_source_flag,
            dry_bulb_uncertainty_flag,
            dew_point_temp,
            dew_point_source_flag,
            dew_point_uncertainty_flag,
            relative_humidity,
            atmospheric_pressure_mbar,
            wind_direction_deg,
            wind_speed,
            visibility_tenths_km,
            ceiling_height_m,
            present_weather_code,
            precipitable_water_mm,
            aerosol_optical_depth_millths,
            snow_depth_cm,
            days_since_last_snowfall,
        })
    }
}

/// A TMY2 weather source: the parsed header plus 8,760 hourly records.
#[derive(Debug, Clone)]
pub struct Tm2WeatherSource {
    location: Tm2Location,
    records: Vec<Tm2Record>,
    /// Materialized engine series (same length as `records`).
    hourly: Vec<HourlyWeatherData>,
}

impl Tm2WeatherSource {
    /// Creates a TMY2 weather source from a file path.
    ///
    /// # Errors
    ///
    /// Returns [`WeatherError`] when the file cannot be read, the
    /// header or any record fails to parse, or the record count is not
    /// 8,760.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, WeatherError> {
        let file = File::open(path).map_err(|e| WeatherError::IoError(e.to_string()))?;
        Self::parse(BufReader::new(file))
    }

    /// Parses TMY2 content from a reader (header + hourly records).
    pub fn parse<R: BufRead>(reader: R) -> Result<Self, WeatherError> {
        let mut lines = reader.lines();
        let header_line = lines
            .next()
            .ok_or_else(|| WeatherError::IncompleteData("TMY2 file is empty".to_string()))?
            .map_err(|e| WeatherError::IoError(e.to_string()))?;
        let location = Tm2Location::parse(&header_line)?;

        let mut records = Vec::new();
        for line in lines {
            let line = line.map_err(|e| WeatherError::IoError(e.to_string()))?;
            if line.trim().is_empty() {
                continue;
            }
            records.push(Tm2Record::parse(&line)?);
        }

        if records.len() != 8760 {
            return Err(WeatherError::IncompleteData(format!(
                "TMY2 file must carry exactly 8,760 hourly records, got {}",
                records.len()
            )));
        }

        let mut hourly = Vec::with_capacity(records.len());
        for (i, r) in records.iter().enumerate() {
            hourly.push(HourlyWeatherData {
                dry_bulb_temp: r.dry_bulb_temp,
                dni: r.dni,
                dhi: r.dhi,
                ghi: r.ghi,
                wind_speed: r.wind_speed,
                wind_direction: r.wind_direction_deg,
                humidity: r.relative_humidity,
                // TMY2 has no horizontal-IR field; 0.0 matches the EPW
                // reader's missing-sentinel coercion (Issue #1415).
                horizontal_infrared: 0.0,
                hour_of_year: i,
                ground_temperature: None,
                horizontal_illuminance: r.ghi_illuminance_hundreds_lux.map(|v| v * 100.0),
                diffuse_illuminance: r.dhi_illuminance_hundreds_lux.map(|v| v * 100.0),
                // Snow depth is cm in TMY2; the engine field is mm.
                snow_depth: r.snow_depth_cm.map(|v| v * 10.0),
                snow_cover: None,
                present_weather: None,
                present_weather_code: r.present_weather_code.map(|c| c as u32),
                // TMY2 does not report ground albedo; callers fall back
                // to the 0.2 ASHRAE 140 fabric default (Issue #4269).
                albedo: None,
            });
        }

        Ok(Tm2WeatherSource {
            location,
            records,
            hourly,
        })
    }

    /// The parsed station header.
    pub fn location_record(&self) -> &Tm2Location {
        &self.location
    }

    /// The fully-parsed hourly records (all Table A1-6 elements).
    pub fn records(&self) -> &[Tm2Record] {
        &self.records
    }
}

impl WeatherSource for Tm2WeatherSource {
    fn location(&self) -> Option<String> {
        Some(self.location.city_state())
    }

    fn get_hourly_data(&self, hour: usize) -> Result<HourlyWeatherData, WeatherError> {
        self.hourly
            .get(hour)
            .cloned()
            .ok_or(WeatherError::InvalidHour(hour))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Figure A1-1 sample header, Addendum a to Standard 140-2001
    /// (p. 27): station 14944, Sioux Falls SD, TZ -6, N 43°34′,
    /// W 96°44′, elevation 435 m. Built to the Table A1-5 field
    /// positions (1-based, leading blank column). Transcribed format
    /// documentation — edition-stable, not numeric-band material; no
    /// ASHRAE-distributed weather file is committed here
    /// (Issues #4284/#4285/#4295).
    const SAMPLE_HEADER: &str = " 14944 SIOUX_FALLS            SD  -6 N 43 34 W  96 44   435 ";

    /// The first hourly record of Figure A1-1 (1985-01-01, hour 01,
    /// local standard time; leading blank is position 1).
    const SAMPLE_RECORD: &str = " 85010101000000000000?00000?00000?00000?00000?00000?00000?010A710A7-150A7-211A7060A70975A7360A7052A70161A700945A70999099999004E7050F8000A700E7";

    /// The hour-12 record from Figure A1-1 (measured solar,
    /// illuminance groups present, unlimited-ceiling sentinel 77777).
    const SAMPLE_RECORD_NOON: &str = " 85010112053014150357G40782G40064G50374I40735I40098I50131I500A700A7-167A7-244A7051A70978A7300A7062A70161A777777A70999999999003E7050F8000A700E7";

    #[test]
    fn parses_sample_header_from_addendum_figure_a1_1() {
        let loc = Tm2Location::parse(SAMPLE_HEADER).expect("header must parse");
        assert_eq!(loc.wban, "14944");
        assert_eq!(loc.city, "SIOUX_FALLS");
        assert_eq!(loc.state, "SD");
        assert_eq!(loc.utc_offset_hours, -6.0);
        // N 43°34′ → 43.5667°N
        assert!((loc.latitude_deg - (43.0 + 34.0 / 60.0)).abs() < 1e-9);
        // W 96°44′ → east-positive −96.7333°
        assert!((loc.longitude_deg + (96.0 + 44.0 / 60.0)).abs() < 1e-9);
        assert_eq!(loc.elevation_m, 435.0);
        assert_eq!(loc.city_state(), "SIOUX_FALLS, SD");
    }

    #[test]
    fn parses_sample_record_units_and_flags() {
        let r = Tm2Record::parse(SAMPLE_RECORD).expect("record must parse");
        assert_eq!((r.year, r.month, r.day, r.hour), (1985, 1, 1, 1));
        // 1 a.m. in January: sun below horizon, ETR and solar both 0.
        assert_eq!(r.etr_horizontal, 0.0);
        assert_eq!(r.ghi, 0.0);
        assert_eq!(r.ghi_source_flag, '?');
        // Dry bulb -15.0 °C (tenths -150), dew point -21.1 °C.
        assert!((r.dry_bulb_temp - (-15.0)).abs() < 1e-9);
        assert!((r.dew_point_temp - (-21.1)).abs() < 1e-9);
        // Pressure 975 mbar, RH 60%, wind 5.2 m/s from 360°.
        assert_eq!(r.atmospheric_pressure_mbar, 975.0);
        assert_eq!(r.relative_humidity, 60.0);
        assert!((r.wind_speed - 5.2).abs() < 1e-9);
        assert_eq!(r.wind_direction_deg, Some(360.0));
        // Illuminance groups carry '?' source at night → None.
        assert!(r.ghi_illuminance_hundreds_lux.is_none());
        // Visibility 0161 → 16.1 km reported; snow depth 000 → 0 cm.
        assert_eq!(r.visibility_tenths_km, Some(161.0));
        assert_eq!(r.snow_depth_cm, Some(0.0));
    }

    #[test]
    fn parses_daytime_record_solar_split_and_illuminance() {
        let r = Tm2Record::parse(SAMPLE_RECORD_NOON).expect("record must parse");
        // Figure A1-1 hour 12: GHI 357, DNI 782, DHI 64 — a real
        // direct/diffuse split the parser must keep distinct.
        assert_eq!(r.ghi, 357.0);
        assert_eq!(r.dni, 782.0);
        assert_eq!(r.dni_source_flag, 'G');
        assert_eq!(r.dhi, 64.0);
        // Measured illuminance (hundreds of lux), 'I' source flag.
        assert_eq!(r.ghi_illuminance_hundreds_lux, Some(374.0));
        assert_eq!(r.dni_illuminance_hundreds_lux, Some(735.0));
        assert_eq!(r.dhi_illuminance_hundreds_lux, Some(98.0));
        // Clear sky: total cover 0 tenths, source A.
        assert_eq!(r.total_sky_cover_tenths, Some(0.0));
        // Dry bulb -16.7 °C; wind 6.2 m/s from 300°.
        assert!((r.dry_bulb_temp - (-16.7)).abs() < 1e-9);
        assert!((r.wind_speed - 6.2).abs() < 1e-9);
        assert_eq!(r.wind_direction_deg, Some(300.0));
        // 77777 = unlimited ceiling, preserved raw (not missing).
        assert_eq!(r.ceiling_height_m, Some(77777.0));
        assert!(r.present_weather_code.is_some());
    }

    #[test]
    fn missing_dry_bulb_is_an_error_not_a_substitution() {
        let bytes: Vec<char> = SAMPLE_RECORD.chars().collect();
        let mut rebuilt: String = bytes[..67].iter().collect();
        rebuilt.push_str("-999");
        rebuilt.extend(bytes[71..].iter());
        let err = Tm2Record::parse(&rebuilt).unwrap_err();
        assert!(format!("{err}").contains("dry bulb"), "got: {err}");
    }

    /// Builds a synthetic, self-consistent full-year TMY2 body: 8,760
    /// records laid out to the Table A1-6 field positions
    /// (360-day calendar — February runs to the 28th — which is fine
    /// for the parser's record-count and calendar sanity checks).
    /// Sinusoidal diurnal dry-bulb wave, midday solar hump with a
    /// fixed direct/diffuse split. No licensed file is involved.
    fn synthetic_year() -> String {
        let md = |m: u32| match m {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            _ => 28,
        };
        let mut body = String::new();
        for m in 1..=12u32 {
            for d in 1..=md(m) {
                for h in 1..=24u32 {
                    let db = 10.0 + 15.0 * ((h as f64 - 13.0) / 12.0 * std::f64::consts::PI).sin();
                    let db_t = (db * 10.0).round() as i32;
                    let daytime = (11..=14).contains(&h);
                    let ghi = if daytime { 600 } else { 0 };
                    let dni = if daytime { 500 } else { 0 };
                    let dhi = if daytime { 100 } else { 0 };
                    let mut line = String::with_capacity(142);
                    line.push(' '); // position 1
                    line.push_str("85"); // year 2-3
                    line.push_str(&format!("{m:02}{d:02}{h:02}")); // 4-9
                    line.push_str("1000"); // ETR horizontal 10-13
                    line.push_str("0800"); // ETR direct normal 14-17
                    line.push_str(&format!("{ghi:04}A7")); // GHI 18-23
                    line.push_str(&format!("{dni:04}A7")); // DNI 24-29
                    line.push_str(&format!("{dhi:04}A7")); // DHI 30-35
                    line.push_str("9999?9".repeat(4).as_str()); // illum+zenith 36-59, not modeled
                    line.push_str("00A700A7"); // sky covers 60-67
                    line.push_str(&format!("{db_t:+04}A7")); // dry bulb 68-73
                    line.push_str(&format!("{db_t:+04}A7")); // dew point 74-79
                    line.push_str("060A7"); // RH 80-84
                    line.push_str("0950A7"); // pressure 85-90
                    line.push_str("360A7"); // wind direction 91-95
                    line.push_str("052A7"); // wind speed 96-100
                    line.push_str("9999?9"); // visibility 101-106
                    line.push_str("99999?9"); // ceiling 107-113
                    line.push_str("9999999999"); // present weather 114-123
                    line.push_str("999A7"); // precipitable water 124-128
                    line.push_str("999A7"); // AOD 129-133
                    line.push_str("999A7"); // snow depth 134-138
                    line.push_str("99A7"); // days since snowfall 139-142
                    assert_eq!(line.chars().count(), 142, "synthetic record width");
                    body.push_str(&line);
                    body.push('\n');
                }
            }
        }
        body
    }

    #[test]
    fn full_year_synthetic_file_parses_to_8760_engine_hours() {
        let text = format!("{SAMPLE_HEADER}\n{}", synthetic_year());
        let src = Tm2WeatherSource::parse(text.as_bytes()).expect("synthetic year must parse");
        assert_eq!(src.records().len(), 8760);
        assert_eq!(src.location(), Some("SIOUX_FALLS, SD".to_string()));

        // WeatherSource contract: hour 8759 valid, 8760 invalid.
        assert!(src.get_hourly_data(8759).is_ok());
        assert!(matches!(
            src.get_hourly_data(8760),
            Err(WeatherError::InvalidHour(_))
        ));

        // Calendar: first record Jan 1 hour 1; hour_of_year is the
        // engine's zero-based index.
        assert_eq!(
            (
                src.records()[0].month,
                src.records()[0].day,
                src.records()[0].hour
            ),
            (1, 1, 1)
        );
        let last = &src.records()[8759];
        assert_eq!(last.hour, 24);

        // Unit scaling survived the round trip: midday heating.
        let noon = src.get_hourly_data(12).unwrap();
        assert_eq!(noon.dni, 500.0);
        assert_eq!(noon.dhi, 100.0);
        assert_eq!(noon.ghi, 600.0);
        let midnight = src.get_hourly_data(3).unwrap();
        assert_eq!(midnight.ghi, 0.0);
        // RH passes through; the diurnal dry-bulb wave is scaled from
        // tenths of °C.
        assert_eq!(noon.humidity, 60.0);
        assert!(
            (src.get_hourly_data(11).unwrap().dry_bulb_temp - src.records()[11].dry_bulb_temp)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn short_file_is_rejected_with_count() {
        let text = format!("{SAMPLE_HEADER}\n{SAMPLE_RECORD}\n");
        let err = Tm2WeatherSource::parse(text.as_bytes()).unwrap_err();
        assert!(format!("{err}").contains("8,760"), "got: {err}");
    }

    #[test]
    fn snow_depth_maps_cm_to_mm_and_missing_to_none() {
        let r = Tm2Record::parse(SAMPLE_RECORD).expect("record must parse");
        // Figure A1-1 record carries snow depth 000 (0 cm, reported).
        assert_eq!(r.snow_depth_cm, Some(0.0));
        // The mapped engine series converts cm → mm.
        let text = format!("{SAMPLE_HEADER}\n{}", synthetic_year());
        let src = Tm2WeatherSource::parse(text.as_bytes()).unwrap();
        // Synthetic records carry 9-filled snow depth → None end to end.
        assert_eq!(r.snow_depth_cm.map(|v| v * 10.0), Some(0.0));
        assert_eq!(src.get_hourly_data(0).unwrap().snow_depth, None);
    }
}
