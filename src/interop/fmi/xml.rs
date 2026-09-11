// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Shared `modelDescription.xml` generation helpers used by both the
//! standard ([`super::export`]) and FFD ([`super::ffd`]) FMU writers.

use quick_xml::events::{BytesEnd, BytesStart, Event};
use quick_xml::Writer;
use std::io::Write;

use super::common::FmiError;
use super::export::{FmiVariables, ZoneVariables};

/// Per-zone variable name: for zone 0 (the legacy single-zone case)
/// use the bare template name so existing FMUs stay name-compatible;
/// for `zone_idx >= 1` prefix with `zone{N}_` to avoid name clashes.
pub(super) fn zone_variable_name(
    variables: &FmiVariables,
    zone: &ZoneVariables,
    zone_idx: usize,
    base: &str,
) -> String {
    if zone_idx == 0 {
        // For the legacy single-zone case the template name is already
        // the user-facing name; the zone's own `name` is informational
        // only (`"zone"`).  This preserves the #1125 spike interface.
        if zone.name == "zone" {
            return variables.suffixed_name(base, 0);
        }
        return format!("{}_{}", sanitize_xml_name(&zone.name), base);
    }
    format!("{}_{}", sanitize_xml_name(&zone.name), base)
}

/// Strip characters that are illegal in FMI variable names.  FMI 2.0
/// variables follow C identifier rules: `[A-Za-z_][A-Za-z0-9_]*`.
pub(super) fn sanitize_xml_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for (i, ch) in name.chars().enumerate() {
        let ok = if i == 0 {
            ch.is_ascii_alphabetic() || ch == '_'
        } else {
            ch.is_ascii_alphanumeric() || ch == '_'
        };
        if ok {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push('z');
    }
    out
}

/// `(start, min, max, unit, description)` for an input variable template.
pub(super) fn input_meta(name: &str) -> (f64, f64, f64, &'static str, &'static str) {
    match name {
        "outdoor_temperature" => (280.0, 200.0, 320.0, "K", "Outdoor dry bulb temperature"),
        "direct_normal_solar" => (0.0, 0.0, 1200.0, "W/m2", "Direct normal solar radiation"),
        "diffuse_horizontal_solar" => (
            0.0,
            0.0,
            800.0,
            "W/m2",
            "Diffuse horizontal solar radiation",
        ),
        "internal_gains" => (0.0, 0.0, 10000.0, "W", "Total internal heat gains"),
        _ => (0.0, 0.0, 0.0, "", ""),
    }
}

/// `(start, min, max, unit, description)` for an output variable template.
pub(super) fn output_meta(name: &str) -> (f64, f64, f64, &'static str, &'static str) {
    match name {
        "zone_temperature" => (293.15, 200.0, 320.0, "K", "Zone air temperature"),
        "heating_load" => (0.0, 0.0, 100000.0, "W", "Heating load (positive)"),
        "cooling_load" => (0.0, 0.0, 100000.0, "W", "Cooling load (positive)"),
        _ => (0.0, 0.0, 0.0, "", ""),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_real_variable<W: Write>(
    writer: &mut Writer<W>,
    name: &str,
    description: &str,
    causality: &str,
    variability: &str,
    start: f64,
    min: f64,
    max: f64,
    unit: &str,
    value_reference: u32,
) -> Result<(), FmiError> {
    let mut sv = BytesStart::new("ScalarVariable");
    sv.push_attribute(("name", name));
    sv.push_attribute(("valueReference", value_reference.to_string().as_str()));
    sv.push_attribute(("description", description));
    sv.push_attribute(("causality", causality));
    sv.push_attribute(("variability", variability));
    writer
        .write_event(Event::Start(sv))
        .map_err(|e| FmiError::ExportFailed(format!("ScalarVariable {name}: {e}")))?;

    let mut real = BytesStart::new("Real");
    // declaredType is omitted unless a TypeDefinitions/SimpleType is referenced.
    real.push_attribute(("quantity", ""));
    real.push_attribute(("unit", unit));
    real.push_attribute(("displayUnit", ""));
    real.push_attribute(("relativeQuantity", "false"));
    real.push_attribute(("min", format_float(min).as_str()));
    real.push_attribute(("max", format_float(max).as_str()));
    real.push_attribute(("nominal", "0.0"));
    real.push_attribute(("unbounded", "false"));
    real.push_attribute(("start", format_float(start).as_str()));
    real.push_attribute(("reinit", "false"));

    writer
        .write_event(Event::Empty(real))
        .map_err(|e| FmiError::ExportFailed(format!("Real {name}: {e}")))?;

    writer
        .write_event(Event::End(BytesEnd::new("ScalarVariable")))
        .map_err(|e| FmiError::ExportFailed(format!("ScalarVariable end {name}: {e}")))?;
    Ok(())
}

/// FMI 2.0 attribute defaults are strings; format f64 compactly.
pub(super) fn format_float(v: f64) -> String {
    // `v == v.trunc()` is a zero-difference integer test (Issue #3357):
    // both operands are computed once each, and `fast-math` reassociation
    // cannot affect a comparison against zero.
    #[allow(clippy::float_cmp)]
    let is_integer = v == v.trunc();
    if is_integer && v.abs() < 1.0e15 {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

/// ISO 8601 UTC timestamp for `generationDateAndTime`.
pub(super) fn generation_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    // Convert epoch seconds to UTC calendar date without pulling in
    // `chrono` (which is already in Cargo.toml as a dependency but
    // avoiding it here keeps this helper dependency-light).
    let days = secs.div_euclid(86_400);
    let secs_of_day = secs.rem_euclid(86_400);
    let (h, m, s) = (
        secs_of_day / 3600,
        (secs_of_day % 3600) / 60,
        secs_of_day % 60,
    );
    let (y, mo, d) = days_to_ymd(days);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, d, h, m, s)
}

/// Convert days-since-epoch to (year, month, day).  Uses the
/// proleptic Gregorian calendar; accurate enough for a `generationDateAndTime`.
pub(super) fn days_to_ymd(days_since_epoch: i64) -> (i32, u32, u32) {
    // Algorithm by Howard Hinnant (public domain).
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = (yoe as i32) + (era as i32) * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
