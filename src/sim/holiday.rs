//! Holiday calendar module for US federal holidays.
//!
//! Provides day type lookup for identifying weekdays, weekends, and holidays.

use std::collections::HashSet;
use std::sync::OnceLock;

use crate::sim::schedule::DayType;

static US_FEDERAL_HOLIDAYS: OnceLock<HashSet<usize>> = OnceLock::new();

/// Calculate US federal holidays for a given year.
///
/// Returns HashSet of day_of_year (1-366) for non-leap years.
/// This implementation uses year-agnostic formulas that work for typical years.
/// For leap years, some holidays may shift by one day after February.
fn calculate_holidays_for_year() -> HashSet<usize> {
    let mut holidays = HashSet::new();

    // Year-agnostic formulas for US federal holidays
    // New Year's Day (Jan 1)
    holidays.insert(1);

    // Martin Luther King Jr Day (3rd Monday in January)
    // Days 1-7: Jan 1-7, check for Monday (day_of_week = 0 for Monday)
    for day in 1..31 {
        let day_of_week = (day - 1) % 7; // 0=Monday
        if day_of_week == 0 && (1..=7).contains(&day) && is_nth_monday(day, 3) {
            holidays.insert(day);
        }
    }

    // Presidents' Day (3rd Monday in February)
    // Days 32-59: Feb 1-28
    for day in 32..60 {
        let day_of_week = (day - 32) % 7;
        if day_of_week == 0 && is_nth_monday(day - 31, 3) {
            holidays.insert(day);
        }
    }

    // Memorial Day (last Monday in May)
    // Days 122-152: May 1-31
    for day in (122..152).rev() {
        let day_of_week = (day - 122) % 7;
        if day_of_week == 0 {
            holidays.insert(day);
            break;
        }
    }

    // Juneteenth (June 19) - Day 170
    holidays.insert(170);

    // Independence Day (July 4) - Day 185
    holidays.insert(185);

    // Labor Day (1st Monday in September)
    // Days 244-273: Sep 1-30
    for day in 244..274 {
        let day_of_week = (day - 244) % 7;
        if day_of_week == 0 && is_nth_monday(day - 243, 1) {
            holidays.insert(day);
        }
    }

    // Columbus Day (2nd Monday in October)
    // Days 274-304: Oct 1-31
    for day in 274..305 {
        let day_of_week = (day - 274) % 7;
        if day_of_week == 0 && is_nth_monday(day - 273, 2) {
            holidays.insert(day);
        }
    }

    // Veterans Day (November 11) - Day 315
    holidays.insert(315);

    // Thanksgiving (4th Thursday in November)
    // Days 306-335: Nov 1-30
    for day in 306..336 {
        let day_of_week = (day - 306) % 7;
        if day_of_week == 3 && is_nth_thursday(day - 305, 4) {
            holidays.insert(day);
        }
    }

    // Christmas (December 25) - Day 359
    holidays.insert(359);

    holidays
}

/// Check if a day is the nth Monday of its month.
fn is_nth_monday(day: usize, n: usize) -> bool {
    ((day - 1) / 7 + 1) == n
}

/// Check if a day is the nth Thursday of its month.
fn is_nth_thursday(day: usize, n: usize) -> bool {
    ((day - 1) / 7 + 1) == n
}

/// Get day type for a given day of year.
///
/// # Arguments
/// * `day_of_year` - Day of year (1-366)
///
/// # Returns
/// * `DayType::Holiday` if the day is a US federal holiday
/// * `DayType::Weekday` for Monday-Friday (non-holiday)
/// * `DayType::Weekend` for Saturday-Sunday (non-holiday)
///
/// # Note
/// This implementation assumes day_of_year 1 (Jan 1) is Monday for simplicity.
/// For accurate year-specific day-of-week calculations, use a calendar library.
///
/// # Examples
/// ```
/// use fluxion::sim::holiday::get_day_type;
/// use fluxion::sim::schedule::DayType;
///
/// // New Year's Day is a holiday
/// assert_eq!(get_day_type(1), DayType::Holiday);
///
/// // Monday Jan 2 is a weekday
/// assert_eq!(get_day_type(2), DayType::Weekday);
///
/// // Saturday Jan 7 is a weekend
/// assert_eq!(get_day_type(7), DayType::Weekend);
/// ```
pub fn get_day_type(day_of_year: usize) -> DayType {
    let day_of_week = (day_of_year - 1) % 7; // 0=Monday, ..., 6=Sunday

    // Check if holiday
    let holidays = US_FEDERAL_HOLIDAYS.get_or_init(calculate_holidays_for_year);
    if holidays.contains(&day_of_year) {
        return DayType::Holiday;
    }

    // Weekday or Weekend
    match day_of_week {
        0..=4 => DayType::Weekday, // Monday-Friday
        5..=6 => DayType::Weekend, // Saturday-Sunday
        _ => DayType::Weekday,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_years_day_is_holiday() {
        assert_eq!(get_day_type(1), DayType::Holiday);
    }

    #[test]
    fn test_weekday_is_weekday() {
        // Day 1 is Monday (holiday - New Year's Day)
        assert_eq!(get_day_type(1), DayType::Holiday);
        // Day 2 is Tuesday (weekday)
        assert_eq!(get_day_type(2), DayType::Weekday);
        // Day 3 is Wednesday (weekday)
        assert_eq!(get_day_type(3), DayType::Weekday);
    }

    #[test]
    fn test_weekend_is_weekend() {
        // Day 6 is Saturday (weekend)
        assert_eq!(get_day_type(6), DayType::Weekend);
        // Day 7 is Sunday (weekend)
        assert_eq!(get_day_type(7), DayType::Weekend);
    }

    #[test]
    fn test_independence_day_is_holiday() {
        assert_eq!(get_day_type(185), DayType::Holiday);
    }

    #[test]
    fn test_christmas_is_holiday() {
        assert_eq!(get_day_type(359), DayType::Holiday);
    }

    #[test]
    fn test_juneteenth_is_holiday() {
        assert_eq!(get_day_type(170), DayType::Holiday);
    }

    #[test]
    fn test_holiday_count() {
        let holidays = calculate_holidays_for_year();
        // Should have 10 federal holidays
        assert_eq!(holidays.len(), 10);
    }
}
