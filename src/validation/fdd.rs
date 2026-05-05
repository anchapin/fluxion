//! Fault Detection and Diagnostics (FDD) for building energy models.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FaultSeverity {
    Info,
    Warning,
    Moderate,
    Critical,
}

impl FaultSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            FaultSeverity::Info => "INFO",
            FaultSeverity::Warning => "WARNING",
            FaultSeverity::Moderate => "MODERATE",
            FaultSeverity::Critical => "CRITICAL",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FaultType {
    SensorStuck,
    SensorDrift,
    SensorBias,
    SensorNoise,
    DeadbandViolation,
    TemperatureAnomaly,
    ControlFault,
    EquipmentDegradation,
    UnexpectedLoad,
}

impl FaultType {
    pub fn as_str(&self) -> &'static str {
        match self {
            FaultType::SensorStuck => "Sensor Stuck",
            FaultType::SensorDrift => "Sensor Drift",
            FaultType::SensorBias => "Sensor Bias",
            FaultType::SensorNoise => "Excessive Sensor Noise",
            FaultType::DeadbandViolation => "Deadband Violation",
            FaultType::TemperatureAnomaly => "Temperature Anomaly",
            FaultType::ControlFault => "Control Fault",
            FaultType::EquipmentDegradation => "Equipment Degradation",
            FaultType::UnexpectedLoad => "Unexpected Load",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fault {
    pub id: String,
    pub fault_type: FaultType,
    pub severity: FaultSeverity,
    pub message: String,
    pub source: String,
    pub _timestamp: usize,
    pub confidence: f64,
    pub recommended_action: String,
}

impl Fault {
    pub fn new(
        fault_type: FaultType,
        severity: FaultSeverity,
        message: String,
        source: String,
        timestamp: usize,
        confidence: f64,
        recommended_action: String,
    ) -> Self {
        let id = format!("{}-{}-{}", source, timestamp, rand::random::<u32>());
        Self {
            id,
            fault_type,
            severity,
            message,
            source,
            _timestamp: timestamp,
            confidence,
            recommended_action,
        }
    }
}

#[derive(Default)]
pub struct FaultDetector {
    faults: Vec<Fault>,
    anomalies: HashMap<String, AnomalyDetector>,
    trackers: HashMap<String, DegradationTracker>,
    sensor_history: HashMap<String, Vec<f64>>,
}

impl FaultDetector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_faults(&self) -> &[Fault] {
        &self.faults
    }

    pub fn get_faults_by_severity(&self, severity: &FaultSeverity) -> Vec<Fault> {
        self.faults
            .iter()
            .filter(|f| f.severity == *severity)
            .cloned()
            .collect()
    }

    pub fn get_faults_by_type(&self, fault_type: &FaultType) -> Vec<Fault> {
        self.faults
            .iter()
            .filter(|f| f.fault_type == *fault_type)
            .cloned()
            .collect()
    }

    pub fn clear_faults(&mut self) {
        self.faults.clear();
    }

    pub fn detect_temperature_anomaly(&mut self, zone_id: &str, temp: f64, hour: usize) {
        let detector = self
            .anomalies
            .entry(zone_id.to_string())
            .or_insert_with(|| AnomalyDetector::new(30));
        detector.add_value(temp);
        if detector.is_anomalous(temp, 3.0) {
            self.faults.push(Fault::new(
                FaultType::TemperatureAnomaly,
                FaultSeverity::Moderate,
                format!("Temperature anomaly detected in {}: {:.1}°C", zone_id, temp),
                zone_id.to_string(),
                hour,
                0.8,
                "Check zone insulation and HVAC operation".to_string(),
            ));
        }
    }

    pub fn detect_control_faults(
        &mut self,
        zone_id: &str,
        temp: f64,
        _heat_sp: f64,
        cool_sp: f64,
        mode: &str,
        hour: usize,
    ) {
        if mode == "heating" && temp > cool_sp {
            self.faults.push(Fault::new(
                FaultType::ControlFault,
                FaultSeverity::Critical,
                format!("Simultaneous heating and cooling in {}", zone_id),
                zone_id.to_string(),
                hour,
                0.95,
                "Check HVAC controller logic".to_string(),
            ));
        }
    }

    pub fn detect_consumption_anomaly(&mut self, id: &str, value: f64, hour: usize) {
        let detector = self
            .anomalies
            .entry(format!("consumption-{}", id))
            .or_insert_with(|| AnomalyDetector::new(30));
        detector.add_value(value);
        if detector.is_anomalous(value, 3.0) {
            self.faults.push(Fault::new(
                FaultType::UnexpectedLoad,
                FaultSeverity::Warning,
                format!("Excessive consumption detected for {}: {:.1} W", id, value),
                id.to_string(),
                hour,
                0.7,
                "Audit equipment efficiency".to_string(),
            ));
        }
    }

    pub fn detect_sensor_faults(&mut self, id: &str, value: f64, range: (f64, f64), hour: usize) {
        if value < range.0 || value > range.1 {
            self.faults.push(Fault::new(
                FaultType::SensorDrift,
                FaultSeverity::Warning,
                format!("Sensor {} out of range: {:.1}", id, value),
                id.to_string(),
                hour,
                0.9,
                "Recalibrate sensor".to_string(),
            ));
            return;
        }

        let history = self.sensor_history.entry(id.to_string()).or_default();
        history.push(value);
        if history.len() > 10 {
            history.remove(0);
            let first = history[0];
            if history.iter().all(|&v| (v - first).abs() < 1e-6) {
                self.faults.push(Fault::new(
                    FaultType::SensorStuck,
                    FaultSeverity::Critical,
                    format!("Sensor {} appears to be stuck at {:.1}", id, value),
                    id.to_string(),
                    hour,
                    0.95,
                    "Replace sensor".to_string(),
                ));
            }
        }
    }

    pub fn track_equipment_performance(
        &mut self,
        id: &str,
        value: f64,
        rated: f64,
        threshold: f64,
        hour: usize,
    ) {
        let tracker = self
            .trackers
            .entry(id.to_string())
            .or_insert_with(|| DegradationTracker::new(rated, 168));
        tracker.update(value);
        if tracker.is_degraded(threshold) {
            self.faults.push(Fault::new(
                FaultType::EquipmentDegradation,
                FaultSeverity::Moderate,
                format!(
                    "Equipment {} efficiency degraded below {:.1}%",
                    id,
                    rated * (1.0 - threshold)
                ),
                id.to_string(),
                hour,
                0.85,
                "Perform equipment maintenance".to_string(),
            ));
        }
    }

    pub fn generate_report(&self) -> String {
        if self.faults.is_empty() {
            return "No faults detected.".to_string();
        }
        let mut report = format!("FDD Report: {} faults detected\n", self.faults.len());
        for fault in &self.faults {
            report.push_str(&format!(
                "[{}] {}: {} ({})\n",
                fault.severity.as_str(),
                fault.source,
                fault.message,
                fault.fault_type.as_str()
            ));
        }
        report
    }
}

pub struct AnomalyDetector {
    pub window_size: usize,
    pub history: Vec<f64>,
}

impl AnomalyDetector {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            history: Vec::new(),
        }
    }

    pub fn add_value(&mut self, value: f64) {
        self.history.push(value);
        if self.history.len() > self.window_size {
            self.history.remove(0);
        }
    }

    pub fn get_z_score(&self, value: f64) -> f64 {
        if self.history.len() < 5 {
            return 0.0;
        }
        let n = self.history.len() as f64;
        let mean = self.history.iter().sum::<f64>() / n;
        let variance = self
            .history
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / n;
        let std_dev = variance.sqrt();
        if std_dev < 1e-6 {
            0.0
        } else {
            (value - mean) / std_dev
        }
    }

    pub fn is_anomalous(&self, value: f64, threshold: f64) -> bool {
        self.get_z_score(value).abs() > threshold
    }
}

pub struct DegradationTracker {
    pub initial_value: f64,
    pub window_size: usize,
    pub history: Vec<f64>,
}

impl DegradationTracker {
    pub fn new(initial_value: f64, window_size: usize) -> Self {
        Self {
            initial_value,
            window_size,
            history: Vec::new(),
        }
    }

    pub fn update(&mut self, value: f64) {
        self.history.push(value);
        if self.history.len() > self.window_size {
            self.history.remove(0);
        }
    }

    pub fn is_degraded(&self, threshold: f64) -> bool {
        if self.history.is_empty() {
            return false;
        }
        let current_avg = self.history.iter().sum::<f64>() / self.history.len() as f64;
        (self.initial_value - current_avg) / self.initial_value > threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fault_severity_as_str() {
        assert_eq!(FaultSeverity::Info.as_str(), "INFO");
        assert_eq!(FaultSeverity::Warning.as_str(), "WARNING");
        assert_eq!(FaultSeverity::Moderate.as_str(), "MODERATE");
        assert_eq!(FaultSeverity::Critical.as_str(), "CRITICAL");
    }

    #[test]
    fn test_fault_severity_ordering() {
        assert!(FaultSeverity::Info < FaultSeverity::Warning);
        assert!(FaultSeverity::Warning < FaultSeverity::Moderate);
        assert!(FaultSeverity::Moderate < FaultSeverity::Critical);
    }

    #[test]
    fn test_fault_type_as_str() {
        assert_eq!(FaultType::SensorStuck.as_str(), "Sensor Stuck");
        assert_eq!(FaultType::SensorDrift.as_str(), "Sensor Drift");
        assert_eq!(FaultType::SensorBias.as_str(), "Sensor Bias");
        assert_eq!(FaultType::SensorNoise.as_str(), "Excessive Sensor Noise");
        assert_eq!(FaultType::DeadbandViolation.as_str(), "Deadband Violation");
        assert_eq!(
            FaultType::TemperatureAnomaly.as_str(),
            "Temperature Anomaly"
        );
        assert_eq!(FaultType::ControlFault.as_str(), "Control Fault");
        assert_eq!(
            FaultType::EquipmentDegradation.as_str(),
            "Equipment Degradation"
        );
    }

    #[test]
    fn test_fault_new() {
        let fault = Fault::new(
            FaultType::SensorStuck,
            FaultSeverity::Warning,
            "Temperature sensor stuck".to_string(),
            "sensor_1".to_string(),
            0,
            0.75,
            "Investigate sensor".to_string(),
        );
        assert!(matches!(fault.fault_type, FaultType::SensorStuck));
        assert!(matches!(fault.severity, FaultSeverity::Warning));
        assert!(!fault.id.is_empty());
    }

    #[test]
    fn test_degradation_tracker_significant_degradation() {
        let mut tracker = DegradationTracker::new(100.0, 168);
        for _ in 0..50 {
            tracker.update(80.0);
        }
        assert!(tracker.is_degraded(0.1));
    }

    #[test]
    fn test_anomaly_detector_detects_outlier() {
        let mut detector = AnomalyDetector::new(30);
        // Establishing mean/std with some variation
        for _ in 0..10 {
            detector.add_value(50.0);
        }
        detector.add_value(51.0);
        detector.add_value(49.0);

        assert!(detector.is_anomalous(200.0, 3.0));
    }

    #[test]
    fn test_fault_detector_new() {
        let detector = FaultDetector::new();
        assert!(detector.get_faults().is_empty());
    }

    #[test]
    fn test_fault_detector_detect_control_faults() {
        let mut detector = FaultDetector::new();
        // Temp (25) is above cooling setpoint (24) while in heating mode
        detector.detect_control_faults("zone_1", 25.0, 20.0, 24.0, "heating", 0);
        let faults = detector.get_faults();
        assert!(!faults.is_empty());
    }

    #[test]
    fn test_fault_detector_clear_faults() {
        let mut detector = FaultDetector::new();
        detector.detect_control_faults("zone_1", 25.0, 20.0, 24.0, "heating", 0);
        assert!(!detector.get_faults().is_empty());
        detector.clear_faults();
        assert!(detector.get_faults().is_empty());
    }

    #[test]
    fn test_fault_detector_getters() {
        let mut detector = FaultDetector::new();
        detector.detect_control_faults("zone_1", 25.0, 20.0, 24.0, "heating", 0);

        let critical_faults = detector.get_faults_by_severity(&FaultSeverity::Critical);
        assert_eq!(critical_faults.len(), 1);

        let warning_faults = detector.get_faults_by_severity(&FaultSeverity::Warning);
        assert_eq!(warning_faults.len(), 0);

        let control_faults = detector.get_faults_by_type(&FaultType::ControlFault);
        assert_eq!(control_faults.len(), 1);
    }

    #[test]
    fn test_detect_temperature_anomaly() {
        let mut detector = FaultDetector::new();
        // Establish baseline
        for i in 0..10 {
            detector.detect_temperature_anomaly("zone_1", 20.0, i);
        }
        // Add anomalous value
        detector.detect_temperature_anomaly("zone_1", 100.0, 10);

        let faults = detector.get_faults_by_type(&FaultType::TemperatureAnomaly);
        assert_eq!(faults.len(), 1);
    }

    #[test]
    fn test_detect_consumption_anomaly() {
        let mut detector = FaultDetector::new();
        for i in 0..10 {
            detector.detect_consumption_anomaly("hvac_1", 1000.0, i);
        }
        detector.detect_consumption_anomaly("hvac_1", 5000.0, 10);

        let faults = detector.get_faults_by_type(&FaultType::UnexpectedLoad);
        assert_eq!(faults.len(), 1);
    }

    #[test]
    fn test_track_equipment_performance() {
        let mut detector = FaultDetector::new();
        for i in 0..10 {
            detector.track_equipment_performance("boiler_1", 0.9, 1.0, 0.1, i);
        }
        // Drastic drop
        detector.track_equipment_performance("boiler_1", 0.5, 1.0, 0.1, 10);

        let faults = detector.get_faults_by_type(&FaultType::EquipmentDegradation);
        assert_eq!(faults.len(), 1);
    }

    #[test]
    fn test_generate_report() {
        let mut detector = FaultDetector::new();
        assert_eq!(detector.generate_report(), "No faults detected.");

        detector.detect_control_faults("zone_1", 25.0, 20.0, 24.0, "heating", 0);
        let report = detector.generate_report();
        assert!(report.contains("FDD Report"));
        assert!(report.contains("CRITICAL"));
    }

    #[test]
    fn test_anomaly_detector_z_score_small_history() {
        let mut detector = AnomalyDetector::new(30);
        detector.add_value(10.0);
        assert_eq!(detector.get_z_score(20.0), 0.0); // Less than 5 values
    }

    #[test]
    fn test_degradation_tracker_empty_history() {
        let tracker = DegradationTracker::new(100.0, 168);
        assert!(!tracker.is_degraded(0.1));
    }
}
