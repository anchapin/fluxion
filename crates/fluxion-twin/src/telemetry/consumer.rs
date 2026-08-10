//! Bounded channel consumer with out-of-order packet deduplication.
//!
//! Implements:
//! - Bounded channel (1024 capacity) to prevent memory exhaustion
//! - Out-of-order buffering per sensor (up to 100 messages)
//! - Sequence number deduplication per sensor
//! - Slow consumer warning when queue depth > 900

use crossbeam_channel::{self as channel, Receiver, RecvError, SendError};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

use super::error::TelemetryError;
use super::message::TelemetryMsg;

const MAX_CHANNEL_CAPACITY: usize = 1024;
const SLOW_CONSUMER_THRESHOLD: usize = 900;

pub struct TelemetryConsumer {
    rx: Receiver<TelemetryMsg>,
    buffer: HashMap<Uuid, VecDeque<TelemetryMsg>>,
    last_sequence: HashMap<Uuid, u64>,
}

impl TelemetryConsumer {
    pub fn new() -> (Sender<TelemetryMsg>, Self) {
        let (tx, rx) = channel::bounded(MAX_CHANNEL_CAPACITY);
        (
            Sender { inner: tx },
            Self {
                rx,
                buffer: HashMap::new(),
                last_sequence: HashMap::new(),
            },
        )
    }

    pub fn with_receiver(rx: Receiver<TelemetryMsg>) -> Self {
        Self {
            rx,
            buffer: HashMap::new(),
            last_sequence: HashMap::new(),
        }
    }

    pub fn process_message(&mut self, msg: TelemetryMsg) -> Option<TelemetryMsg> {
        let sensor_id = msg.sensor_id;
        let seq = msg.sequence;

        if let Some(&last) = self.last_sequence.get(&sensor_id) {
            if seq <= last {
                return None;
            }
        }

        self.last_sequence.insert(sensor_id, seq);
        Some(msg)
    }

    fn try_drain_buffer(
        &mut self,
        sensor_id: Uuid,
    ) -> Option<Result<TelemetryMsg, TelemetryError>> {
        let expected_seq = self.last_sequence.get(&sensor_id).copied().unwrap_or(0) + 1;

        if let Some(buf) = self.buffer.get(&sensor_id) {
            if buf.len() >= 100 {
                return Some(Err(TelemetryError::BufferFull(sensor_id)));
            }
            for (i, msg) in buf.iter().enumerate() {
                if msg.sequence == expected_seq {
                    let msg = self.buffer.get_mut(&sensor_id).unwrap().remove(i).unwrap();
                    return Some(
                        self.process_message(msg)
                            .ok_or(TelemetryError::Recv("Channel closed".to_string())),
                    );
                }
            }
        }
        None
    }

    pub fn recv_with_backpressure(&mut self) -> Result<TelemetryMsg, TelemetryError> {
        let queue_len = self.rx.len();
        if queue_len > SLOW_CONSUMER_THRESHOLD {
            tracing::warn!(
                queue_depth = queue_len,
                capacity = MAX_CHANNEL_CAPACITY,
                threshold = SLOW_CONSUMER_THRESHOLD,
                "Telemetry channel slow consumer: queue depth exceeds backpressure threshold"
            );
            // Issue #2519 — surface slow-consumer pressure as a metric so it
            // can be alerted on without grepping logs. Incremented once per
            // observed breach (the caller invokes recv per message, so this
            // tracks the count of messages received while over threshold).
            metrics::counter!("fluxion_twin_slow_consumer_events_total").increment(1);
        }

        match self.rx.recv() {
            Ok(msg) => {
                let sensor_id = msg.sensor_id;
                let expected_seq = self.last_sequence.get(&sensor_id).copied().unwrap_or(0) + 1;

                if msg.sequence == expected_seq {
                    if let Some(processed) = self.process_message(msg) {
                        Ok(processed)
                    } else {
                        self.recv_with_backpressure()
                    }
                } else if msg.sequence > expected_seq {
                    self.buffer.entry(sensor_id).or_default().push_back(msg);
                    if let Some(result) = self.try_drain_buffer(sensor_id) {
                        return result;
                    }
                    self.recv_with_backpressure()
                } else {
                    self.buffer.entry(sensor_id).or_default().push_back(msg);
                    self.recv_with_backpressure()
                }
            }
            Err(RecvError) => Err(TelemetryError::Recv("Channel closed".to_string())),
        }
    }

    pub fn len(&self) -> usize {
        self.rx.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rx.is_empty()
    }
}

impl Default for TelemetryConsumer {
    fn default() -> Self {
        Self::new().1
    }
}

#[derive(Debug, Clone)]
pub struct Sender<T> {
    inner: channel::Sender<T>,
}

impl<T> Sender<T> {
    pub fn send(&self, msg: T) -> Result<(), SendError<T>> {
        self.inner.send(msg)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity().unwrap_or(MAX_CHANNEL_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn create_test_msg(sensor_id: Uuid, sequence: u64) -> TelemetryMsg {
        TelemetryMsg {
            sensor_id,
            sequence,
            timestamp: chrono::Utc::now(),
            payload: vec![1.0],
        }
    }

    #[test]
    fn test_duplicate_sequence_dropped() {
        let (tx, rx) = channel::bounded(10);
        let mut consumer = TelemetryConsumer::with_receiver(rx);
        let sensor_id = Uuid::new_v4();

        let msg1 = create_test_msg(sensor_id, 1);
        tx.send(msg1.clone()).unwrap();
        let result1 = consumer.recv_with_backpressure();
        assert!(result1.is_ok());

        let msg2 = create_test_msg(sensor_id, 1);
        tx.send(msg2).unwrap();
        let result2 = consumer.recv_with_backpressure();
        assert!(result2.is_err());
    }

    #[test]
    fn test_out_of_order_buffering() {
        let (tx, rx) = channel::bounded(10);
        let mut consumer = TelemetryConsumer::with_receiver(rx);
        let sensor_id = Uuid::new_v4();

        tx.send(create_test_msg(sensor_id, 3)).unwrap();
        tx.send(create_test_msg(sensor_id, 1)).unwrap();
        tx.send(create_test_msg(sensor_id, 2)).unwrap();

        let result1 = consumer.recv_with_backpressure();
        assert!(result1.is_ok());
        assert_eq!(result1.unwrap().sequence, 1);

        let result2 = consumer.recv_with_backpressure();
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap().sequence, 2);

        let result3 = consumer.recv_with_backpressure();
        assert!(result3.is_ok());
        assert_eq!(result3.unwrap().sequence, 3);
    }

    #[test]
    fn test_old_sequence_dropped() {
        let (tx, rx) = channel::bounded(10);
        let mut consumer = TelemetryConsumer::with_receiver(rx);
        let sensor_id = Uuid::new_v4();

        tx.send(create_test_msg(sensor_id, 5)).unwrap();
        let result1 = consumer.recv_with_backpressure();
        assert!(result1.is_ok());
        assert_eq!(result1.unwrap().sequence, 5);

        tx.send(create_test_msg(sensor_id, 3)).unwrap();
        let result2 = consumer.recv_with_backpressure();
        assert!(result2.is_err());
    }

    #[test]
    fn test_multi_sensor_deduplication() {
        let (tx, rx) = channel::bounded(10);
        let mut consumer = TelemetryConsumer::with_receiver(rx);
        let sensor_a = Uuid::new_v4();
        let sensor_b = Uuid::new_v4();

        tx.send(create_test_msg(sensor_a, 1)).unwrap();
        tx.send(create_test_msg(sensor_b, 1)).unwrap();
        tx.send(create_test_msg(sensor_a, 1)).unwrap();
        tx.send(create_test_msg(sensor_b, 1)).unwrap();

        let result1 = consumer.recv_with_backpressure();
        assert!(result1.is_ok());

        let result2 = consumer.recv_with_backpressure();
        assert!(result2.is_ok());

        let result3 = consumer.recv_with_backpressure();
        assert!(result3.is_err());

        let result4 = consumer.recv_with_backpressure();
        assert!(result4.is_err());
    }

    // ---- Observability (Issue #2519): slow-consumer counter ----

    /// When the channel depth exceeds the backpressure threshold,
    /// `recv_with_backpressure` must increment
    /// `fluxion_twin_slow_consumer_events_total`. Uses a thread-local
    /// `DebuggingRecorder` so the assertion is isolated from the process-global
    /// recorder (same pattern as the main crate, Issue #2498).
    #[test]
    fn slow_consumer_counter_increments_past_threshold() {
        use metrics_util::debugging::DebuggingRecorder;

        // A bounded channel with capacity just above the slow-consumer
        // threshold lets us push it past the threshold deterministically.
        let cap = SLOW_CONSUMER_THRESHOLD + 5;
        let (tx, rx) = channel::bounded(cap);
        let mut consumer = TelemetryConsumer::with_receiver(rx);
        let sensor_id = Uuid::new_v4();

        // Fill the channel above the threshold with a monotonic sequence so
        // each recv succeeds instead of recursing on duplicates.
        for seq in 1..=(cap as u64) {
            tx.send(create_test_msg(sensor_id, seq)).unwrap();
        }

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            // Drain one message while the queue is still over threshold.
            let _ = consumer.recv_with_backpressure();
        });

        let map = snapshotter.snapshot().into_hashmap();
        let found = map
            .keys()
            .any(|ck| ck.key().name() == "fluxion_twin_slow_consumer_events_total");
        assert!(
            found,
            "expected fluxion_twin_slow_consumer_events_total to be emitted when queue depth > threshold"
        );
    }
}
