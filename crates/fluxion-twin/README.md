# fluxion-twin

Digital twin core for Fluxion: an **Unscented Kalman Filter (UKF)** for non-linear
thermal state estimation, paired with a backpressure-aware **telemetry pipeline**
that ingests sensor readings over MQTT.

The UKF uses the sigma-point approach to propagate mean and covariance through
non-linear state and measurement functions, avoiding the linearization error
inherent in the Extended Kalman Filter (EKF). Telemetry arrives over MQTT
(MQTT-over-TLS by default) and is delivered to the filter as typed
`MqttTelemetryMessage` values through a bounded channel with out-of-order
deduplication.

## Contents

- [Install](#install)
- [Unscented Kalman Filter](#unscented-kalman-filter)
  - [State model](#state-model)
  - [Predict / update loop](#predict--update-loop)
  - [TwinStateEstimator adapter](#twinstateestimator-adapter)
- [Telemetry](#telemetry)
  - [MQTT broker setup](#mqtt-broker-setup)
  - [Sample subscriber](#sample-subscriber)
  - [Sample publisher](#sample-publisher)
  - [Bounded in-process consumer](#bounded-in-process-consumer)
- [Observability](#observability)
- [Crate layout](#crate-layout)
- [References](#references)

## Install

```toml
[dependencies]
fluxion-twin = { path = "…" }   # or version = "0.1" once published
```

Re-exports of interest live at the crate root: `UnscentedKalmanFilter`,
`UkfTwinAdapter`, `TwinStateEstimator`, `TwinCorrection`, `KalmanError`, and the
telemetry types `MqttTelemetryConsumer`, `MqttTelemetryMessage`,
`TelemetryConsumer`, `TelemetryMsg`, `Sender`.

## Unscented Kalman Filter

### State model

[`UnscentedKalmanFilter<S, M>`] is generic over a state vector type `S` and a
measurement vector type `M`. Both must implement the [`StateVector`] /
[`MeasurementVector`] traits, which `Vec<f64>` satisfies out of the box.

For a single-zone thermal model the state is the zone temperature in °C and the
measurement is an observed zone temperature:

| Quantity | Symbol | Example |
|----------|--------|---------|
| State vector `x` | `x[0]` | zone air temperature (°C) |
| Control input `u` | `u[0..]` | timestep, weather, heat gains, … |
| Measurement `z` | `z[0]` | sensor reading (°C) |
| Process noise `Q` | diag | model uncertainty (°C²) |
| Measurement noise `R` | diag | sensor uncertainty (°C²) |

The filter stores the posterior state estimate and its covariance, both public
fields you can inspect after each step:

```rust
pub struct UnscentedKalmanFilter<S, M> {
    pub state: S,                       // posterior mean estimate
    pub p_covariance: DMatrix<f64>,     // posterior covariance
    pub process_noise: DMatrix<f64>,    // Q
    pub measurement_noise: DMatrix<f64>,// R
    // …
}
```

### Predict / update loop

Each timestep runs a **predict** (propagate state + covariance through the
state-transition function `f(x, u)`) followed by an **update** (fuse a
measurement through the observation function `h(x)`):

```rust
use fluxion_twin::{UnscentedKalmanFilter, KalmanError};
use nalgebra::{DMatrix, DVector};

// Single-zone thermal model: state = [zone_temp_c], measurement = [zone_temp_c].
let initial_state: Vec<f64> = vec![20.0];
let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0]));      // prior uncertainty
let q  = DMatrix::from_diagonal(&DVector::from_vec(vec![0.05_f64.powi(2)])); // process noise
let r  = DMatrix::from_diagonal(&DVector::from_vec(vec![0.5_f64.powi(2)]));  // sensor noise

// f(x, u): trivial decay model (replace with your physics integration).
// h(x):    we observe the zone temperature directly.
let mut ukf = UnscentedKalmanFilter::new(
    initial_state,
    p0,
    q,
    r,
    |x: &Vec<f64>, _u: &[f64]| vec![x[0] * 0.99],  // f(x, u)
    |x: &Vec<f64>| vec![x[0]],                      // h(x)
);

// --- one timestep ---
let u: Vec<f64> = vec![];                          // control inputs (none here)
ukf.predict(&u)?;                                   // propagate
ukf.update(&vec![21.3])?;                           // fuse sensor reading

let estimated_temp = ukf.state[0];                  // corrected posterior
let uncertainty    = ukf.p_covariance[(0, 0)];      // posterior variance (°C^2)
# let _ = (estimated_temp, uncertainty);
# Ok::<(), KalmanError>(())
```

`predict` and `update` each return `Result<(), KalmanError>` —
`KalmanError::NonPositiveDefiniteMatrix` is raised when the covariance cannot be
Cholesky-factorised (e.g. a degenerate zero covariance).

### TwinStateEstimator adapter

For integration with the rest of Fluxion, [`UkfTwinAdapter`] wraps a UKF in the
[`TwinStateEstimator`] trait, whose `correct` returns a [`TwinCorrection`]
(per-zone temperature deltas + a covariance diagonal for trust weighting). A
single-zone thermal default is one constructor call:

```rust
use fluxion_twin::{UkfTwinAdapter, TwinStateEstimator};

// state [zone_temp_c], initial 20.0 °C, process noise σ=0.2, sensor noise σ=0.5
let mut twin = UkfTwinAdapter::<Vec<f64>, Vec<f64>>::single_zone(20.0, 0.2, 0.5);

twin.predict(&[])?;                       // advance the model
let correction = twin.correct(&[21.3])?;  // fuse measurement → TwinCorrection

println!("Δtemp = {:.3} °C", correction.zone_temperatures[0]);
# Ok::<(), fluxion_twin::KalmanError>(())
```

Multi-zone models pass the full state/measurement vectors directly to the
generic `UnscentedKalmanFilter::new`.

## Telemetry

### MQTT broker setup

[`MqttTelemetryConsumer::connect`] subscribes to a broker topic and forwards
parsed [`MqttTelemetryMessage`] values. The transport is **MQTT-over-TLS by
default** (`mqtts://`, port 8883) using rustls with the platform trust store;
server certificates are **validated**.

| Broker URL | Transport | Port |
|------------|-----------|------|
| `mqtts://broker.local` | TLS (certs validated) | 8883 |
| `mqtts://broker.local:8883` | TLS (certs validated) | 8883 |
| `broker.local` (bare host) | TLS (certs validated) | 8883 |
| `mqtt://broker.local:1883` | **plaintext** — rejected unless opt-in | 1883 |
| `tcp://10.0.0.5:1883` | **plaintext** — rejected unless opt-in | 1883 |

Two environment variables control the escape hatches (both **local-dev only**):

| Variable | Effect |
|----------|--------|
| `FLUXION_MQTT_ALLOW_INSECURE` | When truthy (`1`/`true`/`yes`/`on`), permits plaintext (`mqtt://` / `tcp://`) broker URLs. Also the **release-boot-guard opt-in** (Issue #2703). |
| `FLUXION_MQTT_INSECURE` | When truthy, **skips TLS server-certificate validation** (e.g. self-signed brokers). Dangerous — disables all cert checking; logged as a warning. |

Plaintext URLs are rejected with `MqttTelemetryError::InvalidConfig` unless
`FLUXION_MQTT_ALLOW_INSECURE` is set. rumqttc handles automatic reconnection on
transient disconnects.

**Release boot guard (Issue #2703, parity with `fluxion-rest`):** in `--release`
builds, `connect()` refuses to start when the resolved transport is insecure —
plaintext broker URL **or** disabled certificate validation
(`FLUXION_MQTT_INSECURE=1`) — unless the operator has set
`FLUXION_MQTT_ALLOW_INSECURE=1` to explicitly opt in. Debug builds skip the
guard so local dev against self-signed brokers keeps working.

### Sample subscriber

`connect` returns the consumer plus the receiving end of a bounded (capacity
1024) `tokio::sync::mpsc` channel. Spawn `start()` to drive the event loop, then
drain messages from the receiver — or call `next()` directly for lower-level
control:

```rust
use fluxion_twin::MqttTelemetryConsumer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // mqtts://, port 8883, certs validated. For a self-signed local broker
    // set FLUXION_MQTT_INSECURE=1 in the environment first.
    let (consumer, mut rx) =
        MqttTelemetryConsumer::connect("mqtts://broker.local", "fluxion/sensors/#").await?;

    // Drive the event loop on a background task.
    tokio::spawn(async move { let _ = consumer.start().await; });

    while let Some(msg) = rx.recv().await {
        if let Some(temp) = msg.temperature_c {
            println!("sensor={} temp={:.2} °C", msg.sensor_id, temp);
        }
    }
    Ok(())
}
```

For manual polling (one message at a time, with full control over the event
loop), use `MqttTelemetryConsumer::next`, which returns `None` only on a fatal,
unrecoverable condition:

```rust
# use fluxion_twin::MqttTelemetryConsumer;
# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let (mut consumer, _rx) =
    MqttTelemetryConsumer::connect("mqtts://broker.local", "fluxion/sensors/#").await?;
while let Some(msg) = consumer.next().await {
    println!("{msg:?}");
}
# Ok(())
# }
```

### Sample publisher

`fluxion-twin` focuses on the **subscriber/consumer** side. To publish the JSON
payloads it expects, use `rumqttc` directly — `MqttTelemetryMessage` derives
`serde::Serialize`, so serialise it to JSON and publish:

```rust
use fluxion_twin::MqttTelemetryMessage;
use rumqttc::{AsyncClient, MqttOptions, QoS, Transport};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut opts = MqttOptions::new("fluxion-publisher", "broker.local", 8883);
    opts.set_transport(Transport::tls_with_default_config());
    let (client, mut eventloop) = AsyncClient::new(opts, 10);

    let reading = MqttTelemetryMessage {
        sensor_id: "zone-1-temp".to_string(),
        timestamp: 1_700_000_000,
        temperature_c: Some(22.5),
        humidity_pct: Some(45.0),
        power_w: None,
    };
    let payload = serde_json::to_vec(&reading)?;

    client
        .publish("fluxion/sensors/zone-1", QoS::AtLeastOnce, false, payload)
        .await?;

    // Poll the event loop so the publish actually goes out (rumqttc is async;
    // requests are only dispatched while the EventLoop is being polled).
    let _ = eventloop.poll().await;
    Ok(())
}
```

The wire format is plain JSON so any MQTT client (mosquitto_pub, a Python
script, …) can publish compatible messages:

```json
{
    "sensor_id": "zone-1-temp",
    "timestamp": 1700000000,
    "temperature_c": 22.5,
    "humidity_pct": 45.0,
    "power_w": 150.0
}
```

Only `sensor_id` and `timestamp` are required; `temperature_c`, `humidity_pct`,
and `power_w` are optional so heterogeneous sensors can share one topic.

### Bounded in-process consumer

When telemetry comes from within the same process (e.g. a simulator), the
transport-agnostic [`TelemetryConsumer`] offers a bounded `crossbeam-channel`
(capacity 1024) with per-sensor **sequence-number deduplication** and
out-of-order buffering. It reorders late packets and drops stale/duplicate
sequence numbers, surfacing backpressure via
[`TelemetryError::BufferFull`]:

```rust
use fluxion_twin::{TelemetryConsumer, TelemetryMsg};
use uuid::Uuid;

let (sender, mut consumer) = TelemetryConsumer::new();
let sensor = Uuid::new_v4();

// Producer side (e.g. another thread).
sender.send(TelemetryMsg::new(sensor, 1, vec![22.5])).unwrap();
sender.send(TelemetryMsg::new(sensor, 2, vec![22.7])).unwrap();

// Consumer side: blocks until an in-order, non-duplicate message is ready.
let msg = consumer.recv_with_backpressure()?;
assert_eq!(msg.sequence, 1);
# Ok::<(), fluxion_twin::TelemetryError>(())
```

## Observability

Every `predict`/`update` call carries a `tracing` span and records a latency
histogram; the MQTT path records a per-message counter and the bounded consumer
records a slow-consumer counter (Issue #2519). Wire these into a metrics sink
(e.g. Prometheus) in your host binary — the crate only emits, it does not
install a global recorder.

| Metric | When emitted |
|--------|--------------|
| `fluxion_twin_ukf_predict_duration_seconds` | every `predict()` call |
| `fluxion_twin_ukf_update_duration_seconds` | every `update()` call |
| `fluxion_twin_mqtt_messages_total{outcome}` | per MQTT message (`received` / `error`) |
| `fluxion_twin_slow_consumer_events_total` | bounded-channel depth exceeds 900 |

## Crate layout

```
crates/fluxion-twin/
├── Cargo.toml
└── src/
    ├── lib.rs                  # UnscentedKalmanFilter, UkfTwinAdapter, TwinStateEstimator
    ├── error.rs                # KalmanError
    └── telemetry/
        ├── mod.rs              # telemetry re-exports
        ├── consumer.rs         # TelemetryConsumer (bounded + dedup)
        ├── message.rs          # TelemetryMsg
        ├── mqtt.rs             # MqttTelemetryConsumer, MqttTelemetryMessage
        └── error.rs            # TelemetryError, MqttTelemetryError
```

## References

- Wan, E.A. & van der Merwe, R. (2000). *The Unscented Kalman Filter for
  Nonlinear Estimation.* Proc. IEEE AS-SPCC.
- Issue #2508 — add user-facing READMEs for `fluxion-twin` and `fluxion-toon`.
- The companion format crate [`fluxion-toon`](../fluxion-toon/README.md) is used
  elsewhere in Fluxion for compact, LLM-friendly telemetry snapshots.
