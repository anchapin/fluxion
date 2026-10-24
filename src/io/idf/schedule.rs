// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! EnergyPlus IDF **schedule** import — `Schedule:Compact`,
//! `Schedule:Constant` and `ScheduleTypeLimits`.
//!
//! Issue #3991 (the *schedules* slice of the IDF import epic; fenestration,
//! HVAC objects and IDF export are separate slices).
//!
//! # Target representation
//!
//! Imported schedules land in the schema's **existing** schedule type —
//! [`crate::sim::schedule::DailySchedule`], the type behind
//! [`crate::api::schema::ScheduleSet`]. No parallel schedule
//! representation is introduced here: an IDF schedule becomes a
//! [`DailySchedule`] whose [`crate::sim::schedule::ScheduleValues::Weekly`]
//! payload holds the 7 × 24 hourly values (Monday first, matching
//! [`crate::sim::schedule::DayType`]).
//!
//! # Structure mapping
//!
//! | EnergyPlus | Fluxion target |
//! |------------|----------------|
//! | `Schedule:Constant` value | `ScheduleType::Constant`, 24 identical hours |
//! | `Schedule:Compact` with `For: <day type>` | weekly profile, day type → weekday(s) |
//! | `Schedule:Compact` with `Through: <date>` | weekly profile + explicit loss note (see below) |
//!
//! # Lossy conversions
//!
//! The target is hourly and week-shaped; EnergyPlus schedules may be
//! sub-hourly and calendar-shaped. Anything the target cannot express is
//! either reported as an explicit [`ScheduleLoss`] note (never silently
//! dropped) or rejected with a precise [`IdfError`]:
//!
//! - **Annual (`Through:`) schedules** — the calendar-date dimension has
//!   no counterpart in a week-shaped target. The weekly profile is built
//!   from the last matching block per weekday and an explicit loss note
//!   records the collapse.
//! - **Sub-hourly `Until:` boundaries** — truncated to the hour with an
//!   explicit loss note.
//! - **Design-day / holiday day types** — have no weekday counterpart;
//!   dropped with an explicit loss note.
//! - **Incomplete day coverage** — a day type that does not cover the
//!   full 24 h is a hard error (EnergyPlus rejects it too).
//! - **Values outside the declared `ScheduleTypeLimits` range** — a hard
//!   error.

use crate::sim::schedule::{DailySchedule, ScheduleType, ScheduleValues};

use super::error::IdfError;
use super::parser::{IdfObject, IdfValue};

/// Numeric type declared by a `ScheduleTypeLimits` object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericType {
    /// `Continuous` — fractional values are allowed.
    Continuous,
    /// `Discrete` — the value is a count-like integer.
    Discrete,
}

/// Parsed `ScheduleTypeLimits` object.
///
/// Field layout (EnergyPlus IDD):
/// `Name, Lower Limit, Upper Limit, Numeric Type, Unit Type`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleTypeLimits {
    /// Schedule-type-limits name (referenced by schedules).
    pub name: String,
    /// Lower limit, when declared.
    pub lower_limit: Option<f64>,
    /// Upper limit, when declared.
    pub upper_limit: Option<f64>,
    /// `Continuous` or `Discrete`.
    pub numeric_type: NumericType,
}

impl ScheduleTypeLimits {
    /// Validate a schedule value against the declared limits.
    ///
    /// # Errors
    ///
    /// Returns [`IdfError::Conversion`] when `value` is outside
    /// `[lower_limit, upper_limit]`.
    pub fn validate_value(&self, schedule_name: &str, value: f64) -> Result<(), IdfError> {
        if let Some(lo) = self.lower_limit {
            if value < lo {
                return Err(IdfError::conversion_error(format!(
                    "Schedule '{schedule_name}': value {value} is below the lower limit {lo} declared by ScheduleTypeLimits '{}'",
                    self.name
                )));
            }
        }
        if let Some(hi) = self.upper_limit {
            if value > hi {
                return Err(IdfError::conversion_error(format!(
                    "Schedule '{schedule_name}': value {value} is above the upper limit {hi} declared by ScheduleTypeLimits '{}'",
                    self.name
                )));
            }
        }
        Ok(())
    }
}

/// An explicit note about information that could **not** be represented
/// exactly in the target schedule representation.
///
/// Loss notes are produced by the conversion layer
/// ([`super::convert::import_named_schedules`]) so that a lossy import is
/// always visible to the caller instead of silently dropping values.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleLoss {
    /// Name of the schedule the note refers to.
    pub schedule: String,
    /// Human-readable description of the loss.
    pub detail: String,
}

/// The result of importing a single IDF schedule object.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedSchedule {
    /// Schedule name (the IDF object name).
    pub name: String,
    /// The schedule in the target representation.
    pub schedule: DailySchedule,
    /// Explicit notes about anything that could not be represented exactly.
    pub losses: Vec<ScheduleLoss>,
}

// ---------------------------------------------------------------------------
// Small field helpers
// ---------------------------------------------------------------------------

/// Render an [`IdfValue`] as a trimmed string (`None` for empty fields).
fn value_as_text(value: &IdfValue) -> Option<String> {
    value
        .to_display_string()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Render an [`IdfValue`] as a number (`None` when the field is empty or
/// not numeric).
fn value_as_number(value: &IdfValue) -> Option<f64> {
    match value {
        IdfValue::Real(f) => Some(*f),
        IdfValue::Integer(i) => Some(*i as f64),
        IdfValue::String(s) => s.trim().parse::<f64>().ok(),
        IdfValue::Empty => None,
    }
}

/// The object name (field 1) of a schedule object.
fn require_name(obj: &IdfObject) -> Result<String, IdfError> {
    obj.fields.first().and_then(value_as_text).ok_or_else(|| {
        IdfError::conversion_error(format!(
            "{} object is missing its Name field",
            obj.object_type
        ))
    })
}

/// The `ScheduleTypeLimits` name referenced by a schedule object (field 2).
pub fn referenced_limits_name(obj: &IdfObject) -> Option<String> {
    obj.fields.get(1).and_then(value_as_text)
}

// ---------------------------------------------------------------------------
// ScheduleTypeLimits
// ---------------------------------------------------------------------------

/// Parse a `ScheduleTypeLimits` object.
///
/// # Errors
///
/// Returns [`IdfError::Conversion`] when the name is missing, the numeric
/// type is unknown, or the declared limits are inverted.
pub fn parse_schedule_type_limits(obj: &IdfObject) -> Result<ScheduleTypeLimits, IdfError> {
    let name = require_name(obj)?;
    let lower_limit = obj.fields.get(1).and_then(value_as_number);
    let upper_limit = obj.fields.get(2).and_then(value_as_number);
    let numeric_type = match obj.fields.get(3).and_then(value_as_text) {
        None => NumericType::Continuous,
        Some(text) if text.eq_ignore_ascii_case("continuous") => NumericType::Continuous,
        Some(text) if text.eq_ignore_ascii_case("discrete") => NumericType::Discrete,
        Some(other) => {
            return Err(IdfError::conversion_error(format!(
                "ScheduleTypeLimits '{name}': unknown Numeric Type '{other}' (expected Continuous or Discrete)"
            )))
        }
    };
    if let (Some(lo), Some(hi)) = (lower_limit, upper_limit) {
        if lo > hi {
            return Err(IdfError::conversion_error(format!(
                "ScheduleTypeLimits '{name}': Lower Limit {lo} is greater than Upper Limit {hi}"
            )));
        }
    }
    Ok(ScheduleTypeLimits {
        name,
        lower_limit,
        upper_limit,
        numeric_type,
    })
}

// ---------------------------------------------------------------------------
// Schedule:Compact
// ---------------------------------------------------------------------------

/// One `Until: <time>, <value>` data point of a `Schedule:Compact`.
#[derive(Debug, Clone)]
struct Segment {
    /// The `For: <day type>` context in effect (empty = all days).
    day_type: Option<String>,
    /// The `Until:` boundary in hours after midnight (24.0 = end of day).
    until: f64,
    /// The scheduled value.
    value: f64,
}

/// A `Through: <date>`-delimited block of a `Schedule:Compact`.
#[derive(Debug, Clone, Default)]
struct Block {
    /// The `Through:` date of this block, when present.
    date: Option<(u32, u32)>,
    /// The data points of this block, in file order.
    segments: Vec<Segment>,
}

/// Which weekdays a `For: <day type>` selector covers.
#[derive(Debug, Clone)]
enum Selector {
    /// Every weekday (0 = Monday … 6 = Sunday).
    All,
    /// An explicit set of weekdays.
    Days(Vec<usize>),
    /// A day type with no weekday counterpart (holiday / design days).
    Dropped(String),
}

/// Parse an `HH:MM` boundary into hours after midnight.
///
/// # Errors
///
/// Returns [`IdfError::Conversion`] for malformed times.
fn parse_time(schedule: &str, text: &str) -> Result<f64, IdfError> {
    let (hour_text, minute_text) = text.split_once(':').ok_or_else(|| {
        IdfError::conversion_error(format!(
            "Schedule '{schedule}': malformed time '{text}' (expected 'HH:MM')"
        ))
    })?;
    let hours: u32 = hour_text.trim().parse().map_err(|_| {
        IdfError::conversion_error(format!(
            "Schedule '{schedule}': malformed hour in time '{text}' (expected 'HH:MM')"
        ))
    })?;
    let minutes: u32 = minute_text.trim().parse().map_err(|_| {
        IdfError::conversion_error(format!(
            "Schedule '{schedule}': malformed minute in time '{text}' (expected 'HH:MM')"
        ))
    })?;
    if hours > 24 || minutes > 59 || (hours == 24 && minutes != 0) {
        return Err(IdfError::conversion_error(format!(
            "Schedule '{schedule}': time '{text}' is out of range (expected 00:00 … 24:00)"
        )));
    }
    Ok(hours as f64 + minutes as f64 / 60.0)
}

/// Parse a `MM/DD` date into `(month, day)`.
///
/// # Errors
///
/// Returns [`IdfError::Conversion`] for malformed or out-of-range dates.
fn parse_date(schedule: &str, text: &str) -> Result<(u32, u32), IdfError> {
    let (month_text, day_text) = text.split_once('/').ok_or_else(|| {
        IdfError::conversion_error(format!(
            "Schedule '{schedule}': malformed date '{text}' (expected 'MM/DD')"
        ))
    })?;
    let month: u32 = month_text.trim().parse().map_err(|_| {
        IdfError::conversion_error(format!(
            "Schedule '{schedule}': malformed month in date '{text}' (expected 'MM/DD')"
        ))
    })?;
    let day: u32 = day_text.trim().parse().map_err(|_| {
        IdfError::conversion_error(format!(
            "Schedule '{schedule}': malformed day in date '{text}' (expected 'MM/DD')"
        ))
    })?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return Err(IdfError::conversion_error(format!(
            "Schedule '{schedule}': date '{text}' is out of range (expected MM/DD)"
        )));
    }
    Ok((month, day))
}

/// Strip a case-insensitive ASCII prefix, returning the remainder with its
/// original casing intact.
fn strip_prefix_ci<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    let head = text.get(..prefix.len())?;
    if head.eq_ignore_ascii_case(prefix) {
        text.get(prefix.len()..)
    } else {
        None
    }
}

/// Split a `Schedule:Compact` object into its `Through:`-delimited blocks.
///
/// # Errors
///
/// Returns [`IdfError::Conversion`] for malformed field sequences.
fn parse_compact_blocks(obj: &IdfObject, name: &str) -> Result<Vec<Block>, IdfError> {
    let mut blocks: Vec<Block> = Vec::new();
    let mut current = Block::default();
    let mut current_day_type: Option<String> = None;
    let mut pending_until: Option<f64> = None;
    let mut pending_until_text = String::new();

    for (index, field) in obj.fields.iter().enumerate().skip(2) {
        if pending_until.is_some() {
            // The field right after an `Until:` boundary must be the value.
            let value = value_as_number(field).ok_or_else(|| {
                IdfError::conversion_error(format!(
                    "Schedule '{name}': expected a numeric value after '{}' but found '{}'",
                    pending_until_text,
                    field.to_display_string().unwrap_or_default()
                ))
            })?;
            current.segments.push(Segment {
                day_type: current_day_type.clone(),
                until: pending_until.unwrap(),
                value,
            });
            pending_until = None;
            pending_until_text.clear();
            continue;
        }

        let text = match value_as_text(field) {
            Some(t) => t,
            None => continue,
        };
        if let Some(rest) = strip_prefix_ci(&text, "through:") {
            let date = parse_date(name, rest.trim())?;
            if !current.segments.is_empty() {
                blocks.push(current);
            }
            current = Block {
                date: Some(date),
                segments: Vec::new(),
            };
            current_day_type = None;
        } else if let Some(rest) = strip_prefix_ci(&text, "for:") {
            let day_type = rest.trim().to_string();
            if day_type.is_empty() {
                return Err(IdfError::conversion_error(format!(
                    "Schedule '{name}': 'For:' is missing a day type"
                )));
            }
            current_day_type = Some(day_type);
        } else if let Some(rest) = strip_prefix_ci(&text, "until:") {
            pending_until = Some(parse_time(name, rest.trim())?);
            pending_until_text = format!("Until: {}", rest.trim());
        } else {
            return Err(IdfError::conversion_error(format!(
                "Schedule '{name}': unexpected field '{text}' (expected 'Through:', 'For:', 'Until:' or a numeric value)"
            )));
        }
        let _ = index;
    }

    if pending_until.is_some() {
        return Err(IdfError::conversion_error(format!(
            "Schedule '{name}': missing value after '{}'",
            pending_until_text
        )));
    }
    if !current.segments.is_empty() {
        blocks.push(current);
    }
    if blocks.is_empty() {
        return Err(IdfError::conversion_error(format!(
            "Schedule '{name}': no schedule data found"
        )));
    }
    Ok(blocks)
}

/// Map an EnergyPlus day-type name onto the weekdays it covers.
///
/// # Errors
///
/// Returns [`IdfError::Conversion`] for unknown day types.
fn selector_for(name: &str, day_type: Option<&str>) -> Result<Selector, IdfError> {
    let text = match day_type {
        None => return Ok(Selector::All),
        Some(t) => t.trim(),
    };
    let original = text.to_string();
    match text.to_ascii_lowercase().as_str() {
        "alldays" => Ok(Selector::All),
        "weekdays" => Ok(Selector::Days((0..5).collect())),
        "weekends" | "weekend" => Ok(Selector::Days(vec![5, 6])),
        "monday" => Ok(Selector::Days(vec![0])),
        "tuesday" => Ok(Selector::Days(vec![1])),
        "wednesday" => Ok(Selector::Days(vec![2])),
        "thursday" => Ok(Selector::Days(vec![3])),
        "friday" => Ok(Selector::Days(vec![4])),
        "saturday" => Ok(Selector::Days(vec![5])),
        "sunday" => Ok(Selector::Days(vec![6])),
        "holiday" | "summerdesignday" | "winterdesignday" | "springdesignday" | "falldesignday" => {
            Ok(Selector::Dropped(original))
        }
        _ => Err(IdfError::conversion_error(format!(
            "Schedule '{name}': unknown day type '{original}'"
        ))),
    }
}

/// Build the 24-hour profile of one day-type group.
///
/// Returns the profile plus a flag telling whether a sub-hourly boundary
/// had to be truncated to the hour.
///
/// # Errors
///
/// Returns [`IdfError::Conversion`] when the boundaries do not increase or
/// when the group does not cover the full day.
fn profile_from_segments(
    name: &str,
    segments: &[(f64, f64)],
) -> Result<([f64; 24], bool), IdfError> {
    let mut profile = [0.0_f64; 24];
    let mut covered = [false; 24];
    let mut start = 0.0_f64;
    let mut fractional = false;

    for (until, value) in segments {
        if *until <= start {
            return Err(IdfError::conversion_error(format!(
                "Schedule '{name}': 'Until:' boundaries must increase (got {until} after {start})"
            )));
        }
        if (until - until.floor()).abs() > f64::EPSILON {
            fractional = true;
        }
        for hour in 0..24 {
            let hour_start = hour as f64;
            if hour_start >= start && hour_start < *until {
                profile[hour] = *value;
                covered[hour] = true;
            }
        }
        start = *until;
    }

    if let Some(hour) = covered.iter().position(|c| !c) {
        return Err(IdfError::conversion_error(format!(
            "Schedule '{name}': no value defined for hour {hour} (a schedule must cover the whole day, i.e. end with 'Until: 24:00')"
        )));
    }
    Ok((profile, fractional))
}

/// Import a `Schedule:Compact` object into the target representation.
///
/// # Errors
///
/// Returns [`IdfError::Conversion`] for malformed schedules and for values
/// outside the referenced `ScheduleTypeLimits` range.
pub fn import_schedule_compact(
    obj: &IdfObject,
    limits: Option<&ScheduleTypeLimits>,
) -> Result<ImportedSchedule, IdfError> {
    let name = require_name(obj)?;
    let blocks = parse_compact_blocks(obj, &name)?;

    let mut weekly = [[0.0_f64; 24]; 7];
    let mut losses: Vec<ScheduleLoss> = Vec::new();
    let has_dates = blocks.iter().any(|b| b.date.is_some());

    for block in &blocks {
        // Group consecutive segments that share the same `For:` context.
        let mut groups: Vec<(Option<String>, Vec<(f64, f64)>)> = Vec::new();
        for segment in &block.segments {
            if let Some(last) = groups.last_mut() {
                if last.0 == segment.day_type {
                    last.1.push((segment.until, segment.value));
                    continue;
                }
            }
            groups.push((
                segment.day_type.clone(),
                vec![(segment.until, segment.value)],
            ));
        }

        for (day_type, segments) in groups {
            let (profile, fractional) = profile_from_segments(&name, &segments)?;
            if fractional {
                losses.push(ScheduleLoss {
                    schedule: name.clone(),
                    detail: "sub-hourly 'Until:' boundary truncated to the hour".to_string(),
                });
            }
            match selector_for(&name, day_type.as_deref())? {
                Selector::All => {
                    for day in weekly.iter_mut() {
                        *day = profile;
                    }
                }
                Selector::Days(days) => {
                    for day in days {
                        weekly[day] = profile;
                    }
                }
                Selector::Dropped(dropped) => {
                    losses.push(ScheduleLoss {
                        schedule: name.clone(),
                        detail: format!(
                            "day type '{dropped}' has no weekday equivalent in the weekly target; its values were dropped"
                        ),
                    });
                }
            }
        }
    }

    if has_dates {
        losses.push(ScheduleLoss {
            schedule: name.clone(),
            detail: "annual schedule ('Through:' dates): the calendar-date dimension cannot be represented by the weekly target; the last matching block per weekday was used"
                .to_string(),
        });
    }

    // Validate every value against the referenced limits.
    if let Some(limits) = limits {
        for block in &blocks {
            for segment in &block.segments {
                limits.validate_value(&name, segment.value)?;
            }
        }
    }

    Ok(ImportedSchedule {
        name: name.clone(),
        schedule: DailySchedule {
            name,
            schedule_type: ScheduleType::Weekly,
            values: ScheduleValues::Weekly(weekly),
        },
        losses,
    })
}

// ---------------------------------------------------------------------------
// Schedule:Constant
// ---------------------------------------------------------------------------

/// Import a `Schedule:Constant` object into the target representation.
///
/// # Errors
///
/// Returns [`IdfError::Conversion`] when the value is missing or outside
/// the referenced `ScheduleTypeLimits` range.
pub fn import_schedule_constant(
    obj: &IdfObject,
    limits: Option<&ScheduleTypeLimits>,
) -> Result<ImportedSchedule, IdfError> {
    let name = require_name(obj)?;
    let value = obj.fields.get(2).and_then(value_as_number).ok_or_else(|| {
        IdfError::conversion_error(format!(
            "Schedule:Constant '{name}': expected a numeric value in field 3"
        ))
    })?;
    if let Some(limits) = limits {
        limits.validate_value(&name, value)?;
    }
    Ok(ImportedSchedule {
        name: name.clone(),
        schedule: DailySchedule {
            name,
            schedule_type: ScheduleType::Constant,
            values: ScheduleValues::Daily([value; 24]),
        },
        losses: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::idf::parser::IdfParser;

    /// Parse a one-object IDF snippet and return the single object.
    fn single_object(src: &str) -> IdfObject {
        let idf = IdfParser::from_str(src).expect("IDF parses");
        assert_eq!(idf.objects.len(), 1, "expected exactly one object");
        idf.objects.into_iter().next().unwrap()
    }

    /// The weekly profile of an imported schedule (Monday first).
    fn weekly(schedule: &DailySchedule) -> &[[f64; 24]; 7] {
        match &schedule.values {
            ScheduleValues::Weekly(weekly) => weekly,
            other => panic!("expected a weekly schedule, got {other:?}"),
        }
    }

    /// The daily profile of an imported schedule.
    fn daily(schedule: &DailySchedule) -> &[f64; 24] {
        match &schedule.values {
            ScheduleValues::Daily(daily) => daily,
            other => panic!("expected a daily schedule, got {other:?}"),
        }
    }

    #[test]
    fn compact_schedule_with_day_types_maps_to_weekly_profile() {
        let obj = single_object(
            "Schedule:Compact, Occupancy, Fraction,\
             \n  For: Weekdays, Until: 08:00, 0.1, Until: 18:00, 0.9, Until: 24:00, 0.2,\
             \n  For: Weekends, Until: 24:00, 0.5;\n",
        );
        let imported = import_schedule_compact(&obj, None).expect("schedule imports");

        assert_eq!(imported.name, "Occupancy");
        assert_eq!(imported.schedule.schedule_type, ScheduleType::Weekly);

        let week = weekly(&imported.schedule);
        // Monday … Friday carry the weekday profile.
        for day in 0..5 {
            assert!(
                (week[day][0] - 0.1).abs() < 1e-9,
                "weekday hour 0 = {}",
                week[day][0]
            );
            assert!(
                (week[day][7] - 0.1).abs() < 1e-9,
                "weekday hour 7 = {}",
                week[day][7]
            );
            assert!(
                (week[day][8] - 0.9).abs() < 1e-9,
                "weekday hour 8 = {}",
                week[day][8]
            );
            assert!(
                (week[day][17] - 0.9).abs() < 1e-9,
                "weekday hour 17 = {}",
                week[day][17]
            );
            assert!(
                (week[day][18] - 0.2).abs() < 1e-9,
                "weekday hour 18 = {}",
                week[day][18]
            );
            assert!(
                (week[day][23] - 0.2).abs() < 1e-9,
                "weekday hour 23 = {}",
                week[day][23]
            );
        }
        // Saturday and Sunday carry the weekend profile.
        for day in 5..7 {
            for hour in 0..24 {
                assert!(
                    (week[day][hour] - 0.5).abs() < 1e-9,
                    "weekend hour {hour} = {}",
                    week[day][hour]
                );
            }
        }
        assert!(
            imported.losses.is_empty(),
            "unexpected losses: {:?}",
            imported.losses
        );
    }

    #[test]
    fn compact_schedule_with_through_dates_records_an_explicit_loss() {
        let obj = single_object(
            "Schedule:Compact, Heating, Temperature,\
             \n  Through: 01/31, For: AllDays, Until: 24:00, 20.0,\
             \n  Through: 12/31, For: AllDays, Until: 24:00, 22.0;\n",
        );
        let imported = import_schedule_compact(&obj, None).expect("schedule imports");

        // The last matching block wins (documented collapse of the
        // calendar-date dimension) — and the collapse is reported.
        let week = weekly(&imported.schedule);
        assert!((week[0][0] - 22.0).abs() < 1e-9, "hour 0 = {}", week[0][0]);
        assert!(
            imported
                .losses
                .iter()
                .any(|l| l.detail.contains("Through:") && l.schedule == "Heating"),
            "expected an explicit annual-schedule loss note, got {:?}",
            imported.losses
        );
    }

    #[test]
    fn constant_schedule_maps_to_a_constant_profile() {
        let obj = single_object("Schedule:Constant, AlwaysOn, Fraction, 1.0;\n");
        let imported = import_schedule_constant(&obj, None).expect("schedule imports");

        assert_eq!(imported.name, "AlwaysOn");
        assert_eq!(imported.schedule.schedule_type, ScheduleType::Constant);
        let values = daily(&imported.schedule);
        for hour in 0..24 {
            assert!(
                (values[hour] - 1.0).abs() < 1e-9,
                "hour {hour} = {}",
                values[hour]
            );
        }
        assert!(imported.losses.is_empty());
    }

    #[test]
    fn malformed_compact_schedule_reports_a_precise_error() {
        // A value is missing after the "Until:" boundary.
        let obj =
            single_object("Schedule:Compact, Broken, Fraction, For: AllDays, Until: 24:00;\n");
        let err = import_schedule_compact(&obj, None).expect_err("malformed schedule is rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("Broken")
                && (msg.contains("expected a numeric value")
                    || msg.contains("missing value after")),
            "error should name the schedule and the problem, got: {msg}"
        );
    }

    #[test]
    fn malformed_compact_schedule_reports_unknown_day_type() {
        let obj = single_object(
            "Schedule:Compact, Broken, Fraction, For: NotADayType, Until: 24:00, 1.0;\n",
        );
        let err = import_schedule_compact(&obj, None).expect_err("unknown day type is rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("unknown day type") && msg.contains("NotADayType"),
            "error should name the unknown day type, got: {msg}"
        );
    }

    #[test]
    fn malformed_compact_schedule_reports_non_increasing_boundaries() {
        let obj = single_object(
            "Schedule:Compact, Broken, Fraction, Until: 12:00, 1.0, Until: 06:00, 2.0, Until: 24:00, 3.0;\n",
        );
        let err = import_schedule_compact(&obj, None).expect_err("decreasing boundary is rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("must increase"),
            "error should explain the boundary problem, got: {msg}"
        );
    }

    #[test]
    fn compact_schedule_missing_the_end_of_day_is_rejected() {
        let obj = single_object("Schedule:Compact, Broken, Fraction, Until: 12:00, 1.0;\n");
        let err = import_schedule_compact(&obj, None).expect_err("incomplete day is rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("no value defined for hour 12"),
            "error should point at the uncovered hour, got: {msg}"
        );
    }

    #[test]
    fn schedule_type_limits_are_parsed_and_enforced() {
        let obj = single_object("ScheduleTypeLimits, Fraction, 0.0, 1.0, Continuous;\n");
        let limits = parse_schedule_type_limits(&obj).expect("limits parse");
        assert_eq!(limits.name, "Fraction");
        assert_eq!(limits.numeric_type, NumericType::Continuous);
        assert_eq!(limits.lower_limit, Some(0.0));
        assert_eq!(limits.upper_limit, Some(1.0));

        // A value above the declared upper limit is rejected.
        let schedule =
            single_object("Schedule:Compact, TooHot, Fraction, For: AllDays, Until: 24:00, 1.5;\n");
        let err =
            import_schedule_compact(&schedule, Some(&limits)).expect_err("out-of-range rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("TooHot") && msg.contains("above the upper limit"),
            "error should explain the range violation, got: {msg}"
        );
    }

    #[test]
    fn schedule_type_limits_with_an_unknown_numeric_type_are_rejected() {
        let obj = single_object("ScheduleTypeLimits, Weird, 0.0, 1.0, NotAType;\n");
        let err = parse_schedule_type_limits(&obj).expect_err("unknown numeric type is rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("unknown Numeric Type"),
            "error should explain the numeric type problem, got: {msg}"
        );
    }

    #[test]
    fn sub_hourly_boundaries_are_truncated_with_a_loss_note() {
        let obj = single_object(
            "Schedule:Compact, Fractional, Fraction, For: AllDays, Until: 08:30, 1.0, Until: 24:00, 0.5;\n",
        );
        let imported = import_schedule_compact(&obj, None).expect("schedule imports");
        let week = weekly(&imported.schedule);
        // The 08:30 boundary is truncated to the hour: hour 8 belongs to
        // the first segment.
        assert!((week[0][8] - 1.0).abs() < 1e-9, "hour 8 = {}", week[0][8]);
        assert!((week[0][9] - 0.5).abs() < 1e-9, "hour 9 = {}", week[0][9]);
        assert!(
            imported
                .losses
                .iter()
                .any(|l| l.detail.contains("sub-hourly")),
            "expected a sub-hourly loss note, got {:?}",
            imported.losses
        );
    }

    #[test]
    fn holiday_day_types_are_dropped_with_a_loss_note() {
        let obj = single_object(
            "Schedule:Compact, WithHoliday, Fraction, For: AllDays, Until: 24:00, 1.0, For: Holiday, Until: 24:00, 0.0;\n",
        );
        let imported = import_schedule_compact(&obj, None).expect("schedule imports");
        assert!(
            imported
                .losses
                .iter()
                .any(|l| l.detail.contains("Holiday") && l.detail.contains("dropped")),
            "expected a holiday loss note, got {:?}",
            imported.losses
        );
        // The "AllDays" profile is untouched by the dropped holiday block.
        let week = weekly(&imported.schedule);
        assert!((week[0][0] - 1.0).abs() < 1e-9, "hour 0 = {}", week[0][0]);
    }
}
