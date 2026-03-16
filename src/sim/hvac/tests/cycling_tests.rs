//! Unit tests for cycling loss tracking

use fluxion::sim::hvac::cycling::CyclingTracker;

#[test]
fn test_cycling_losses() {
    let mut tracker = CyclingTracker::new();

    // Initial state
    assert_eq!(tracker.startup_count, 0);
    assert_eq!(tracker.cumulative_runtime_hours, 0.0);

    // First startup
    let (_mult, penalty) = tracker.calculate_cycling_loss(true, 0.5);
    assert_eq!(tracker.startup_count, 1); // Detected startup
    assert_eq!(penalty, 0.1); // Startup penalty 0.1 kWh
    assert_eq!(tracker.current_runtime_timesteps, 1);

    // No startup (still on) - within minimum runtime, no PLR degradation
    let (mult, penalty) = tracker.calculate_cycling_loss(true, 0.5);
    assert_eq!(tracker.startup_count, 1); // No new startup
    assert_eq!(penalty, 0.0); // No penalty
    assert_eq!(mult, 1.0); // No PLR degradation (within minimum runtime)
    assert_eq!(tracker.current_runtime_timesteps, 2);

    // Shutdown
    let (_mult, penalty) = tracker.calculate_cycling_loss(false, 0.0);
    assert_eq!(tracker.was_on, false);
    assert_eq!(penalty, 0.0);
    assert_eq!(tracker.current_runtime_timesteps, 2); // Not incremented

    // Second startup
    let (_mult, penalty) = tracker.calculate_cycling_loss(true, 0.5);
    assert_eq!(tracker.startup_count, 2); // New startup detected
    assert_eq!(penalty, 0.1); // Startup penalty again
    assert_eq!(tracker.current_runtime_timesteps, 1); // Reset on startup
}

#[test]
fn test_minimum_runtime_enforcement() {
    let mut tracker = CyclingTracker::new();

    // Startup
    tracker.calculate_cycling_loss(true, 0.5);
    assert_eq!(tracker.current_runtime_timesteps, 1);
    assert!(tracker.must_run()); // Must run for 5 timesteps

    // Within minimum runtime (timesteps 2-5)
    for _ in 2..6 {
        tracker.calculate_cycling_loss(true, 0.5);
        // Runtime is now: 2, 3, 4, 5
        // Minimum runtime is 5, so must_run should be true for runtime <= 5
        assert!(tracker.must_run());
    }

    // After loop, we're at timestep 5 (within minimum runtime)
    assert_eq!(tracker.current_runtime_timesteps, 5);
    assert!(tracker.must_run()); // Still must run at timestep 5

    // One more timestep to pass minimum runtime
    tracker.calculate_cycling_loss(true, 0.5);
    assert_eq!(tracker.current_runtime_timesteps, 6);
    assert!(!tracker.must_run()); // Can shut down now (past minimum runtime)
}

#[test]
fn test_startup_penalty() {
    let mut tracker = CyclingTracker::new();

    // First call: is_on=true, plr=0.5
    let (_mult, penalty) = tracker.calculate_cycling_loss(true, 0.5);
    assert_eq!(tracker.startup_count, 1);
    assert_eq!(penalty, 0.1); // Startup penalty applied

    // Second call: is_on=false, plr=0.0
    let (_mult, penalty) = tracker.calculate_cycling_loss(false, 0.0);
    assert_eq!(tracker.startup_count, 1); // No new startup
    assert_eq!(penalty, 0.0); // No penalty when turning off

    // Third call: is_on=true, plr=0.5 (new startup)
    let (_mult, penalty) = tracker.calculate_cycling_loss(true, 0.5);
    assert_eq!(tracker.startup_count, 2); // New startup
    assert_eq!(penalty, 0.1); // Startup penalty applied again
}

#[test]
fn test_plr_degradation() {
    // Test with default degradation factor (0.2)
    let mut tracker = CyclingTracker::new();

    // After minimum runtime, test various PLR values
    for _ in 0..6 {
        tracker.calculate_cycling_loss(true, 0.5);
    }

    // PLR=0.5 → multiplier = 1.0 + 0.2 * (1.0 - 0.5) = 1.1
    let (mult_50, _) = tracker.calculate_cycling_loss(true, 0.5);
    assert!((mult_50 - 1.1).abs() < 0.01);

    // PLR=1.0 → multiplier = 1.0 + 0.2 * 0.0 = 1.0 (no degradation at full load)
    let (mult_100, _) = tracker.calculate_cycling_loss(true, 1.0);
    assert_eq!(mult_100, 1.0);

    // PLR=0.0 → multiplier = 1.0 + 0.2 * 1.0 = 1.2 (20% degradation at no load)
    let (mult_0, _) = tracker.calculate_cycling_loss(true, 0.0);
    assert_eq!(mult_0, 1.2);

    // Test with different degradation factor (0.3)
    let mut tracker2 = CyclingTracker {
        startup_penalty_kwh: 0.1,
        plr_degradation_factor: 0.3,
        ..CyclingTracker::new()
    };

    for _ in 0..6 {
        tracker2.calculate_cycling_loss(true, 0.3);
    }

    // PLR=0.3, degradation=0.3 → multiplier = 1.0 + 0.3 * 0.7 = 1.21
    let (mult_custom, _) = tracker2.calculate_cycling_loss(true, 0.3);
    assert!((mult_custom - 1.21).abs() < 0.01);
}
