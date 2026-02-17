//! Demand Response and Load Shifting
//!
//! This module provides demand response and load shifting capabilities
//! for grid-interactive building operations.

use serde::{Deserialize, Serialize};

/// Demand response event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DREventType {
    /// Emergency demand response (high priority)
    Emergency,
    /// Voluntary demand response
    Voluntary,
    /// Ancillary services (fast responding)
    AncillaryServices,
    /// Peak shaving event
    PeakShaving,
    /// Load shifting
    LoadShifting,
}

/// Demand response signal source
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DRSignalSource {
    /// Manual/utility signal
    Manual,
    /// OpenADR signal
    OpenADR,
    /// Real-time pricing
    RealTimePricing,
    /// Time-of-use pricing
    TimeOfUse,
}

/// Priority levels for load shedding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum LoadPriority {
    /// Critical loads (never shed)
    Critical = 3,
    /// High priority loads
    High = 2,
    /// Medium priority loads
    Medium = 1,
    /// Low priority loads (shed first)
    Low = 0,
}

/// Represents a demand response event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DREvent {
    /// Event identifier
    pub id: String,
    /// Event type
    pub event_type: DREventType,
    /// Signal source
    pub source: DRSignalSource,
    /// Event start time (hour of week 0-167)
    pub start_hour: usize,
    /// Event duration (hours)
    pub duration_hours: usize,
    /// Target load reduction (kW)
    pub target_reduction: f64,
    /// Minimum reduction required (kW)
    pub min_reduction: f64,
    /// Event is currently active
    pub is_active: bool,
    /// Actual load reduction achieved (kW)
    pub achieved_reduction: f64,
}

impl DREvent {
    /// Create a new demand response event
    pub fn new(
        id: String,
        event_type: DREventType,
        start_hour: usize,
        duration_hours: usize,
        target_reduction: f64,
    ) -> Self {
        Self {
            id,
            event_type,
            source: DRSignalSource::Manual,
            start_hour,
            duration_hours,
            target_reduction,
            min_reduction: target_reduction * 0.8, // 80% of target
            is_active: false,
            achieved_reduction: 0.0,
        }
    }

    /// Check if event is active at a given hour
    pub fn is_active_at(&self, hour_of_week: usize) -> bool {
        let end_hour = self.start_hour + self.duration_hours;
        hour_of_week >= self.start_hour && hour_of_week < end_hour
    }

    /// Start the event
    pub fn start(&mut self) {
        self.is_active = true;
    }

    /// End the event
    pub fn end(&mut self) {
        self.is_active = false;
    }

    /// Check if target was met
    pub fn target_met(&self) -> bool {
        self.achieved_reduction >= self.min_reduction
    }
}

/// Load shedding controller
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadSheddingController {
    /// Available loads with their priorities
    pub loads: Vec<Load>,
    /// Total shedding capacity (kW)
    pub total_shedding_capacity: f64,
    /// Currently shed loads (kW)
    pub current_shed_load: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Load {
    /// Load identifier
    pub id: String,
    /// Load name
    pub name: String,
    /// Load power consumption (kW)
    pub power_kw: f64,
    /// Priority level
    pub priority: LoadPriority,
    /// Whether this load can be shed
    pub can_shed: bool,
    /// Whether this load is currently shed
    pub is_shed: bool,
}

impl LoadSheddingController {
    /// Create a new load shedding controller
    pub fn new() -> Self {
        Self {
            loads: Vec::new(),
            total_shedding_capacity: 0.0,
            current_shed_load: 0.0,
        }
    }

    /// Add a load to the controller
    pub fn add_load(&mut self, load: Load) {
        if load.can_shed {
            self.total_shedding_capacity += load.power_kw;
        }
        self.loads.push(load);
    }

    /// Calculate available shedding capacity
    pub fn available_capacity(&self) -> f64 {
        self.total_shedding_capacity - self.current_shed_load
    }

    /// Shed loads to meet target reduction
    pub fn shed_loads(&mut self, target_kw: f64) -> f64 {
        let mut remaining = target_kw;
        self.current_shed_load = 0.0;

        // Sort loads by priority (lowest first)
        let mut sheddable: Vec<_> = self.loads.iter_mut()
            .filter(|l| l.can_shed && !l.is_shed)
            .collect();
        sheddable.sort_by_key(|l| l.priority);

        for load in sheddable {
            if remaining <= 0.0 {
                break;
            }

            let shed_amount = load.power_kw.min(remaining);
            load.is_shed = true;
            self.current_shed_load += shed_amount;
            remaining -= shed_amount;
        }

        // Return actual load shed
        target_kw - remaining
    }

    /// Restore all loads
    pub fn restore_all(&mut self) {
        for load in &mut self.loads {
            load.is_shed = false;
        }
        self.current_shed_load = 0.0;
    }

    /// Get current total load
    pub fn current_load(&self) -> f64 {
        self.loads.iter()
            .map(|l| if l.is_shed { 0.0 } else { l.power_kw })
            .sum()
    }
}

impl Default for LoadSheddingController {
    fn default() -> Self {
        Self::new()
    }
}

/// Real-time pricing data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimePricing {
    /// Price by hour of day ($/kWh)
    pub hourly_prices: [f64; 24],
    /// Current price ($/kWh)
    pub current_price: f64,
    /// Average price ($/kWh)
    pub average_price: f64,
    /// Peak price ($/kWh)
    pub peak_price: f64,
    /// Off-peak price ($/kWh)
    pub off_peak_price: f64,
}

impl RealTimePricing {
    /// Create new RTP data
    pub fn new() -> Self {
        Self {
            hourly_prices: [0.10; 24], // Default $0.10/kWh
            current_price: 0.10,
            average_price: 0.10,
            peak_price: 0.10,
            off_peak_price: 0.10,
        }
    }

    /// Set time-of-use pricing
    pub fn set_time_of_use(&mut self, off_peak: f64, mid_peak: f64, peak: f64) {
        for hour in 0..24 {
            self.hourly_prices[hour] = match hour {
                0..=6 => off_peak,     // Night
                7..=9 => mid_peak,     // Morning ramp
                10..=16 => peak,       // Mid-day peak
                17..=19 => peak,       // Evening peak
                20..=22 => mid_peak,   // Evening ramp down
                23 => off_peak,        // Night
                _ => mid_peak,
            };
        }
        self.update_statistics();
    }

    /// Update price for current hour
    pub fn update(&mut self, hour: usize) {
        self.current_price = self.hourly_prices[hour % 24];
    }

    /// Update price statistics
    fn update_statistics(&mut self) {
        self.average_price = self.hourly_prices.iter().sum::<f64>() / 24.0;
        self.peak_price = self.hourly_prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        self.off_peak_price = self.hourly_prices.iter().cloned().fold(f64::INFINITY, f64::min);
    }
}

impl Default for RealTimePricing {
    fn default() -> Self {
        Self::new()
    }
}

/// Demand response manager coordinating all DR activities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DRManager {
    /// Active demand response events
    pub events: Vec<DREvent>,
    /// Load shedding controller
    pub load_shedding: LoadSheddingController,
    /// Real-time pricing
    pub rtp: RealTimePricing,
    /// DR enabled flag
    pub enabled: bool,
    /// Grid signal callback URL (for OpenADR)
    pub signal_url: Option<String>,
}

impl DRManager {
    /// Create a new DR manager
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            load_shedding: LoadSheddingController::new(),
            rtp: RealTimePricing::new(),
            enabled: false,
            signal_url: None,
        }
    }

    /// Add a demand response event
    pub fn add_event(&mut self, event: DREvent) {
        self.events.push(event);
    }

    /// Get active events at current hour
    pub fn active_events(&self, hour_of_week: usize) -> Vec<&DREvent> {
        self.events.iter()
            .filter(|e| e.is_active_at(hour_of_week))
            .collect()
    }

    /// Calculate total load reduction target from active events
    pub fn total_reduction_target(&self, hour_of_week: usize) -> f64 {
        self.active_events(hour_of_week)
            .iter()
            .map(|e| e.target_reduction)
            .sum()
    }

    /// Check if DR is active at current time
    pub fn is_dr_active(&self, hour_of_week: usize) -> bool {
        !self.active_events(hour_of_week).is_empty()
    }

    /// Set time-of-use pricing
    pub fn set_pricing(&mut self, off_peak: f64, mid_peak: f64, peak: f64) {
        self.rtp.set_time_of_use(off_peak, mid_peak, peak);
    }
}

impl Default for DRManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dr_event() {
        let event = DREvent::new(
            "DR-1".to_string(),
            DREventType::PeakShaving,
            14,   // 2pm
            4,    // 4 hours
            100.0, // 100 kW
        );

        assert!(event.is_active_at(15)); // During event
        assert!(!event.is_active_at(10)); // Before event
    }

    #[test]
    fn test_load_shedding() {
        let mut controller = LoadSheddingController::new();

        controller.add_load(Load {
            id: "L1".to_string(),
            name: "HVAC".to_string(),
            power_kw: 50.0,
            priority: LoadPriority::Low,
            can_shed: true,
            is_shed: false,
        });

        controller.add_load(Load {
            id: "L2".to_string(),
            name: "Lighting".to_string(),
            power_kw: 30.0,
            priority: LoadPriority::High,
            can_shed: true,
            is_shed: false,
        });

        // Try to shed 60kW - should shed low priority first
        let shed = controller.shed_loads(60.0);
        assert!(shed >= 50.0); // Should shed at least 50kW
    }

    #[test]
    fn test_rtp_pricing() {
        let mut rtp = RealTimePricing::new();
        rtp.set_time_of_use(0.05, 0.10, 0.25);

        assert!(rtp.peak_price > rtp.off_peak_price);
        assert!(rtp.average_price > 0.0);
    }

    #[test]
    fn test_dr_manager() {
        let mut manager = DRManager::new();
        manager.enabled = true;

        let event = DREvent::new(
            "DR-1".to_string(),
            DREventType::Emergency,
            10,
            2,
            50.0,
        );
        manager.add_event(event);

        let target = manager.total_reduction_target(11);
        assert!(target > 0.0);
    }
}
