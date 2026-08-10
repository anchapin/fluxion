//! Bounded channel consumer with out-of-order packet deduplication.
//!
//! Implements:
//! - Bounded channel (1024 capacity) to prevent memory exhaustion
//! - Out-of-order buffering per sensor (up to 100 messages)
//! - Sequence number deduplication per sensor
//! - Slow consumer warning when queue depth > 900

use crossbeam_channel::{self as channel, Receiver, RecvError, SendError, TryRecvError};
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

/// Outcome of classifying a freshly-received message against the per-sensor
/// sequence frontier. Kept private — it is an internal control-flow signal
/// for [`TelemetryConsumer::recv_with_backpressure`].
enum RecvOutcome {
    /// In-order message ready to hand to `process_message` and return.
    Deliver(TelemetryMsg),
    /// Duplicate / stale sequence, or out-of-order buffer full.
    Error(TelemetryError),
    /// Out-of-order message buffered; keep draining the channel.
    Buffered,
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

    /// Pop the next in-order buffered message, if one is ready.
    ///
    /// A buffered message is "ready" when it is exactly the next sequence
    /// expected for its sensor (`last_sequence[sensor] + 1`). Scanning every
    /// sensor keeps multi-sensor consumers correct: any sensor whose gap has
    /// been filled emits its buffered message before the channel is touched
    /// again.
    ///
    /// Emitting ready buffered messages *before* reading the channel is the
    /// core fix for Issue #2617: without this step a `recv` that follows an
    /// out-of-order `recv` blocks on the (now empty) channel even though the
    /// next message is already sitting in the buffer.
    fn pop_ready_buffered(&mut self) -> Option<TelemetryMsg> {
        let last_sequence = &self.last_sequence;
        let target = self
            .buffer
            .iter()
            .filter_map(|(sensor_id, buf)| {
                let front = buf.front()?;
                let last = last_sequence.get(sensor_id).copied().unwrap_or(0);
                (front.sequence == last + 1).then_some((*sensor_id, front.sequence))
            })
            // Smallest ready sequence first; tie-break by sensor id so the
            // pick is deterministic regardless of HashMap iteration order.
            .min_by_key(|&(sensor_id, seq)| (seq, sensor_id))
            .map(|(sensor_id, _)| sensor_id)?;

        let buf = self.buffer.get_mut(&target)?;
        let msg = buf.pop_front();
        if buf.is_empty() {
            self.buffer.remove(&target);
        }
        msg
    }

    /// Fallback for when the channel is empty but out-of-order messages are
    /// still buffered and the gap cannot be filled from the channel. Emits
    /// the buffered message closest to the per-sensor frontier so the
    /// consumer keeps making progress instead of blocking forever — the
    /// second half of the #2617 deadlock fix.
    fn pop_best_buffered(&mut self) -> Option<TelemetryMsg> {
        let target = self
            .buffer
            .iter()
            .filter_map(|(sensor_id, buf)| {
                let front = buf.front()?;
                Some((*sensor_id, front.sequence))
            })
            .min_by_key(|&(sensor_id, seq)| (seq, sensor_id))
            .map(|(sensor_id, _)| sensor_id)?;

        let buf = self.buffer.get_mut(&target)?;
        let msg = buf.pop_front();
        if buf.is_empty() {
            self.buffer.remove(&target);
        }
        msg
    }

    /// Run an in-order message through `process_message`, converting the
    /// (impossible-in-practice, since the caller already verified the
    /// sequence is strictly greater than the last delivered) `None` into a
    /// channel-closed error so the signature stays `Result`.
    fn deliver(&mut self, msg: TelemetryMsg) -> Result<TelemetryMsg, TelemetryError> {
        self.process_message(msg)
            .ok_or_else(|| TelemetryError::Recv("Channel closed".to_string()))
    }

    /// Classify a freshly-pulled message against the per-sensor sequence
    /// frontier. Duplicates / stale packets and a full out-of-order buffer
    /// are surfaced as errors; an in-order packet is delivered; an
    /// out-of-order packet is buffered and the caller keeps draining.
    fn classify(&mut self, msg: TelemetryMsg) -> RecvOutcome {
        let sensor_id = msg.sensor_id;
        let seq = msg.sequence;
        let last = self.last_sequence.get(&sensor_id).copied().unwrap_or(0);
        if seq <= last {
            RecvOutcome::Error(TelemetryError::Recv(format!(
                "duplicate or stale sequence {seq} for sensor {sensor_id} (last delivered {last})"
            )))
        } else if seq == last + 1 {
            RecvOutcome::Deliver(msg)
        } else {
            let buf = self.buffer.entry(sensor_id).or_default();
            if buf.len() >= 100 {
                RecvOutcome::Error(TelemetryError::BufferFull(sensor_id))
            } else {
                buf.push_back(msg);
                RecvOutcome::Buffered
            }
        }
    }

    pub fn recv_with_backpressure(&mut self) -> Result<TelemetryMsg, TelemetryError> {
        // Slow-consumer backpressure signal (Issue #2519). Sampled once per
        // recv invocation: if the queue depth has grown past the threshold
        // we warn and bump the slow-consumer counter so it can be alerted on
        // without grepping logs.
        let queue_len = self.rx.len();
        if queue_len > SLOW_CONSUMER_THRESHOLD {
            tracing::warn!(
                queue_depth = queue_len,
                capacity = MAX_CHANNEL_CAPACITY,
                threshold = SLOW_CONSUMER_THRESHOLD,
                "Telemetry channel slow consumer: queue depth exceeds backpressure threshold"
            );
            // Incremented once per observed breach (the caller invokes recv
            // per message, so this tracks the count of messages received
            // while over threshold).
            metrics::counter!("fluxion_twin_slow_consumer_events_total").increment(1);
        }

        loop {
            // (1) Emit any buffered message that has become the
            // next-expected for its sensor. This MUST happen before touching
            // the channel — otherwise a buffered message could hide behind
            // an empty channel and stall the consumer forever (Issue #2617).
            if let Some(ready) = self.pop_ready_buffered() {
                return self.deliver(ready);
            }

            // (2) Pull the next message. A non-blocking `try_recv` first so
            // we can reorder across several already-queued packets *and*
            // notice when the channel runs dry — which is exactly when we
            // must fall back to the buffer to avoid deadlocking.
            let msg = match self.rx.try_recv() {
                Ok(msg) => msg,
                Err(TryRecvError::Disconnected) => {
                    return Err(TelemetryError::Recv("Channel closed".to_string()));
                }
                Err(TryRecvError::Empty) => {
                    if self.buffer.is_empty() {
                        // Nothing buffered and nothing queued — block for the
                        // next message (normal backpressure semantics), then
                        // classify it on the next loop iteration.
                        match self.rx.recv() {
                            Ok(msg) => msg,
                            Err(RecvError) => {
                                return Err(TelemetryError::Recv("Channel closed".to_string()));
                            }
                        }
                    } else {
                        // Channel empty but the buffer still holds
                        // out-of-order messages whose gap cannot be filled.
                        // Emit the closest buffered message so the consumer
                        // makes progress instead of blocking forever (#2617).
                        match self.pop_best_buffered() {
                            Some(best) => return self.deliver(best),
                            None => match self.rx.recv() {
                                Ok(msg) => msg,
                                Err(RecvError) => {
                                    return Err(TelemetryError::Recv("Channel closed".to_string()));
                                }
                            },
                        }
                    }
                }
            };

            // (3) Classify the freshly-pulled message.
            match self.classify(msg) {
                RecvOutcome::Deliver(msg) => return self.deliver(msg),
                RecvOutcome::Error(e) => return Err(e),
                RecvOutcome::Buffered => continue,
            }
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
