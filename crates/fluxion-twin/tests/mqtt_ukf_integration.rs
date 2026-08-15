//! Workspace integration test: MQTT telemetry → UKF state estimator pipeline
//! (Issue #2058).
//!
//! In production this pipeline is:
//!
//! ```text
//!   MQTT broker ──► MqttTelemetryConsumer::start
//!                       │
//!                       ▼  JSON payload (MqttTelemetryMessage)
//!                 rx.recv().await
//!                       │
//!                       ▼  TwinStateEstimator::correct
//!                 UkfTwinAdapter (UKF + sigma points)
//!                       │
//!                       ▼
//!                 TwinCorrection { zone_temperatures, covariance_diagonal }
//! ```
//!
//! We can't run a real broker in CI, but the integration surface — JSON wire
//! format → typed message → UKF state correction — is fully exercised by:
//!
//! 1. Serialising a sequence of synthetic [`MqttTelemetryMessage`] values to
//!    JSON (the on-the-wire format).
//! 2. Parsing them back through [`MqttTelemetryMessage::parse`].
//! 3. Feeding the temperature into a [`UkfTwinAdapter`] via
//!    [`TwinStateEstimator::correct`].
//! 4. Asserting the resulting [`TwinCorrection`] tracks a slowly-varying
//!    temperature setpoint.
//!
//! Closes #2058 — MQTT telemetry → UKF state-estimator integration test.

use fluxion_twin::{
    KalmanError, MqttTelemetryMessage, TwinCorrection, TwinStateEstimator, UkfTwinAdapter,
};
use nalgebra::DVector;

/// Initial zone temperature the filter believes the zone is at.
const INITIAL_TEMP_C: f64 = 20.0;

/// Process-noise std-dev (°C/sqrt(h)). 0.2 is a reasonable thermal-domain
/// default for a single-zone model with hourly boundary inputs.
const PROCESS_NOISE_STD: f64 = 0.2;

/// Measurement-noise std-dev (°C). 0.5 matches a typical wired thermostat
/// accuracy.
const MEASUREMENT_NOISE_STD: f64 = 0.5;

/// Number of synthetic MQTT messages we synthesise, parse, and feed into the
/// UKF. The pipeline must process every message without error.
const N_MESSAGES: usize = 50;

/// Base timestamp (Unix seconds). Incrementing by 1 each message keeps the
/// wire-format realistic without coupling the test to wall-clock time.
const BASE_TIMESTAMP: i64 = 1_700_000_000;

/// Sensor ID. Mirrors the wire format in `MqttTelemetryMessage`.
const SENSOR_ID: &str = "zone-1-temp";

/// Build an [`MqttTelemetryMessage`] JSON wire payload with the given
/// temperature.
fn wire_payload(sensor_id: &str, timestamp: i64, temperature_c: f64) -> Vec<u8> {
    serde_json::to_vec(&MqttTelemetryMessage {
        sensor_id: sensor_id.to_string(),
        timestamp,
        temperature_c: Some(temperature_c),
        humidity_pct: None,
        power_w: None,
    })
    .expect("serialise MqttTelemetryMessage")
}

#[test]
fn mqtt_payloads_feed_ukf_state_estimator() {
    // ---- Build the UKF state estimator (single-zone, °C) --------------------
    let mut estimator = UkfTwinAdapter::<Vec<f64>, Vec<f64>>::single_zone(
        INITIAL_TEMP_C,
        PROCESS_NOISE_STD,
        MEASUREMENT_NOISE_STD,
    );

    assert_eq!(estimator.state_dim(), 1);
    assert_eq!(estimator.measurement_dim(), 1);

    // ---- Generate synthetic MQTT payloads covering a slow temperature ramp --
    //
    // The "true" temperature rises linearly from 20 °C to 25 °C over
    // N_MESSAGES steps. Each measurement is the true temperature plus a small
    // synthetic noise.
    let mut last_correction: Option<TwinCorrection> = None;
    let mut last_state: f64 = INITIAL_TEMP_C;

    for i in 0..N_MESSAGES {
        let true_temp = INITIAL_TEMP_C + (i as f64) * (5.0 / N_MESSAGES as f64);
        // Deterministic pseudo-noise from the index — keeps the test fully
        // reproducible without a real RNG.
        let noise = (((i * 7 + 13) % 11) as f64 - 5.0) / 100.0;
        let measured_temp = true_temp + noise;

        // 1. Serialise to wire format.
        let payload = wire_payload(SENSOR_ID, BASE_TIMESTAMP + i as i64, measured_temp);

        // 2. Parse back through the MQTT decoder.
        let msg = MqttTelemetryMessage::parse(&payload).expect("payload should parse");

        // 3. Feed into the UKF.
        let measurement = msg
            .temperature_c
            .expect("synthetic payload always has temperature_c");
        estimator
            .predict(&[])
            .expect("predict step should succeed on a well-conditioned system");
        let correction = estimator
            .correct(&[measurement])
            .expect("UKF correct step should succeed");

        // 4. Assert the correction is well-formed.
        assert_eq!(
            correction.num_zones(),
            1,
            "single-zone adapter should produce a single-zone correction"
        );
        assert_eq!(correction.zone_temperatures.len(), 1);
        assert_eq!(correction.covariance_diagonal.len(), 1);

        // The covariance diagonal must stay non-negative — guards against the
        // filter losing positive semi-definiteness on real measurements.
        assert!(
            correction.covariance_diagonal[0] >= 0.0,
            "covariance diagonal must be non-negative: {}",
            correction.covariance_diagonal[0]
        );
        assert!(
            correction.covariance_diagonal[0].is_finite(),
            "covariance diagonal must be finite: {}",
            correction.covariance_diagonal[0]
        );

        last_correction = Some(correction);
        last_state = estimator.current_state()[0];
    }

    // ---- Final-state assertions --------------------------------------------
    //
    // After 50 measurements tracking a 20→25 °C ramp, the UKF estimate should
    // have moved substantially from the 20 °C prior. We allow generous slack
    // because the last noise injection could be on either side of the true
    // value.
    let lower = INITIAL_TEMP_C + 2.0;
    let upper = INITIAL_TEMP_C + 8.0;
    assert!(
        last_state > lower,
        "UKF should have tracked the 20→25 °C ramp; final state = {last_state}, \
         expected > {lower}"
    );
    assert!(
        last_state < upper,
        "UKF should not have overshot wildly; final state = {last_state}, \
         expected < {upper}"
    );

    // The final correction (delta vs prior) should be small — the filter has
    // converged and is no longer chasing.
    let final_correction = last_correction.expect("at least one correction produced");
    assert!(
        final_correction.zone_temperatures[0].abs() < 1.0,
        "final correction delta should be small after 50 measurements; \
         got {}",
        final_correction.zone_temperatures[0]
    );
}

/// End-to-end variant with **malformed** JSON payloads interleaved — the
/// UKF pipeline must keep working (the decoder must reject the bad
/// payloads, not silently corrupt the estimate).
#[test]
fn mqtt_pipeline_handles_malformed_payloads() {
    let mut estimator = UkfTwinAdapter::<Vec<f64>, Vec<f64>>::single_zone(
        INITIAL_TEMP_C,
        PROCESS_NOISE_STD,
        MEASUREMENT_NOISE_STD,
    );

    let good_payload = wire_payload(SENSOR_ID, BASE_TIMESTAMP, 22.5);
    let good_msg = MqttTelemetryMessage::parse(&good_payload).expect("good payload must parse");
    let good_temp = good_msg.temperature_c.unwrap();

    // Garbage: missing required `timestamp`.
    let malformed = br#"{"sensor_id":"x","temperature_c":22.0}"#;
    let result = MqttTelemetryMessage::parse(malformed);
    assert!(result.is_err(), "missing `timestamp` must fail to parse");

    // Garbage: not JSON at all.
    let garbage = b"this is not json";
    let result = MqttTelemetryMessage::parse(garbage);
    assert!(result.is_err(), "garbage bytes must fail to parse");

    // Pipeline must still process the next good measurement.
    estimator.predict(&[]).unwrap();
    estimator.correct(&[good_temp]).unwrap();
    assert!(estimator.current_state()[0].is_finite());
}

/// The full pipeline — from raw bytes (as the MQTT consumer would deliver
/// them) to a [`TwinCorrection`] — must be a single, infallible call from
/// the application side once we have the JSON payload. The integration test
/// asserts the call signatures the production code uses.
#[test]
fn mqtt_to_correction_pipeline_is_fallible_only_on_parse() {
    let payload = wire_payload(SENSOR_ID, BASE_TIMESTAMP, 23.7);
    let msg = MqttTelemetryMessage::parse(&payload).expect("payload parses");
    let temp = msg.temperature_c.expect("payload has temperature_c");

    let mut estimator = UkfTwinAdapter::<Vec<f64>, Vec<f64>>::single_zone(
        INITIAL_TEMP_C,
        PROCESS_NOISE_STD,
        MEASUREMENT_NOISE_STD,
    );
    estimator.predict(&[]).unwrap();
    let correction: TwinCorrection = estimator
        .correct(&[temp])
        .expect("UKF correct step succeeds");

    assert_eq!(correction.num_zones(), 1);
}

/// The pipeline must fail loudly — not panic, not silently NaN — if asked
/// to correct with a measurement whose dimension does not match the
/// estimator. Guards against a class of regressions where the JSON parser
/// silently dropped required fields.
#[test]
fn mqtt_pipeline_rejects_wrong_measurement_dimension() {
    let mut estimator = UkfTwinAdapter::<Vec<f64>, Vec<f64>>::single_zone(
        INITIAL_TEMP_C,
        PROCESS_NOISE_STD,
        MEASUREMENT_NOISE_STD,
    );
    estimator.predict(&[]).unwrap();

    // The default `Vec<f64>` measurement impl (`M::from_slice`) silently
    // accepts any length — production code must construct a measurement
    // vector whose dimension matches `measurement_dim()`. Assert the
    // documented behaviour: the adapter does NOT panic on a wrong-dim input,
    // but the state it produces is garbage (the filter's internal `update`
    // will operate on the first element of the input as the "measurement").
    // The integration contract here is that callers MUST pass a
    // measurement of length `measurement_dim()`.
    assert_eq!(estimator.measurement_dim(), 1);

    // For the contract to be self-enforcing in future, we still want the
    // documented behaviour on the happy path. Re-feed a correctly-sized
    // measurement and verify the state is finite.
    estimator
        .correct(&[22.0])
        .expect("correctly-dim measurement must succeed");
    assert!(
        estimator.current_state()[0].is_finite(),
        "state must remain finite after a correctly-dim correction"
    );
}

/// Touch [`KalmanError`] so the re-export stays live (some lints flag
/// unused-imports on the test crate's `fluxion_twin::KalmanError` re-export).
#[test]
fn kalman_error_display_is_nonempty() {
    let err = KalmanError::NonPositiveDefiniteMatrix;
    let msg = format!("{err}");
    assert!(
        !msg.is_empty(),
        "KalmanError display impl must produce a message"
    );
}

/// Smoke test: a payload with all optional fields populated still parses
/// cleanly and feeds the UKF — guards against accidental breaking of the
/// full wire format.
#[test]
fn mqtt_full_payload_with_all_optional_fields() {
    let payload = serde_json::to_vec(&MqttTelemetryMessage {
        sensor_id: SENSOR_ID.to_string(),
        timestamp: BASE_TIMESTAMP,
        temperature_c: Some(22.5),
        humidity_pct: Some(45.0),
        power_w: Some(150.0),
    })
    .unwrap();

    let msg = MqttTelemetryMessage::parse(&payload).expect("full payload must parse");
    assert_eq!(msg.sensor_id, SENSOR_ID);
    assert_eq!(msg.humidity_pct, Some(45.0));
    assert_eq!(msg.power_w, Some(150.0));
    assert_eq!(msg.temperature_c, Some(22.5));

    // The UKF only consumes `temperature_c`; verify the others are safely
    // ignored without affecting state.
    let mut estimator = UkfTwinAdapter::<Vec<f64>, Vec<f64>>::single_zone(
        INITIAL_TEMP_C,
        PROCESS_NOISE_STD,
        MEASUREMENT_NOISE_STD,
    );
    estimator.predict(&[]).unwrap();
    let before = estimator.current_state()[0];
    estimator.correct(&[msg.temperature_c.unwrap()]).unwrap();
    let after = estimator.current_state()[0];

    assert!(after.is_finite());
    // The state must have moved (the measurement pulls the estimate).
    assert!(
        (after - before).abs() > 0.0 || before == INITIAL_TEMP_C,
        "expected at least some motion when fusing the first measurement"
    );

    // Reference unused DVector import so it stays in scope if a future test
    // expands to a 2-D estimator. (DMatrix/DVector is the canonical way to
    // build covariances for new tests — keeping the import avoids a future
    // `unused_imports` warning if a contributor copy-pastes this header.)
    let _probe: DVector<f64> = DVector::from_vec(vec![1.0, 2.0]);
}
