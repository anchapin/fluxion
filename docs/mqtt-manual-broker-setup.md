# Manual MQTT Broker Setup — `test_mqtt_consumer_integration`

> **TL;DR**: How to run the quarantined MQTT integration test by hand against a local
> TLS broker (mosquitto + self-signed CA), because the test is MANUAL-ONLY.
> **Key decisions**: The test stays `#[ignore]`d; provisioning a TLS broker in CI is out
> of scope for Issue #4211 | `crates/fluxion-twin/src/telemetry/mqtt.rs`.
> **Owned by**: fluxion-twin maintainers (tracking issue #4211).
> **Reviewed**: 2026-10-23

## Decision: manual-only (Issue #4211)

`test_mqtt_consumer_integration` is **not** a CI candidate and is **not** being
un-ignored. The decision was scoped by the issue owner:

* The test needs a real MQTT-over-TLS broker that the client can validate against
  the platform trust store. Provisioning such a broker in CI (containers, service
  containers, CI jobs, CI secret material for a CA) is **out of scope for Issue
  #4211**.
* Therefore the test remains `#[ignore = "requires a running TLS MQTT broker at
  mqtts://localhost:8883"]` (see
  [`crates/fluxion-twin/src/telemetry/mqtt.rs`](../crates/fluxion-twin/src/telemetry/mqtt.rs),
  line 823) and is run **by hand** using the recipe below.
* The registry row for this test lives in [`tests/QUARANTINE.md`](../tests/QUARANTINE.md)
  under *Category: Other / Unclassified* and points back at this document.

No CI file, workflow, or Cargo manifest is changed by this document.

## What the test actually requires (extracted from the code)

Every row below is read directly from the test and the consumer implementation.

| Requirement | Value | Code evidence |
|---|---|---|
| Broker URL (hard-coded) | `mqtts://localhost:8883` | `mqtt.rs:827` — `let broker = "mqtts://localhost:8883";` |
| Broker host / port | `localhost` / `8883` | parsed by `parse_broker_url` (`mqtt.rs:398`), `DEFAULT_MQTTS_PORT = 8883` (`mqtt.rs:52`) |
| Transport | TLS only, certificate validation **always on** | `mqtt.rs:199` — `mqttoptions.set_transport(Transport::tls_with_default_config());` |
| Root CA source | the **platform/native trust store** — no CA path, no CA env var is read by Fluxion code | rumqttc's `TlsConfiguration::default()` loads `load_native_certs()`; `mqtt.rs` never passes a CA |
| Client certificate / key | **not used** — the client is built with `with_no_client_auth()` | rumqttc `TlsConfiguration::default()` → `ClientConfig::builder()…with_no_client_auth()`; `mqtt.rs` calls no `Transport::tls(ca, client_auth, …)` |
| Credentials (username/password) | **none** — `connect()` never calls `set_credentials`, so the broker must accept an anonymous client | `mqtt.rs:197-202` (`MqttOptions::new`, `set_keep_alive`, `set_transport`, `client.subscribe`) |
| Client ID | `fluxion-twin-consumer` | `mqtt.rs:197` — `MqttOptions::new("fluxion-twin-consumer", host, port)` |
| Keep-alive | 5 s | `mqtt.rs:198` — `mqttoptions.set_keep_alive(Duration::from_secs(5));` |
| Subscribed topic | `fluxion/test/zone1` (exact, QoS `AtLeastOnce`) | `mqtt.rs:828` — `let topic = "fluxion/test/zone1";`, `mqtt.rs:202` — `client.subscribe(topic, QoS::AtLeastOnce)` |
| Published by the test | **nothing** — the test never publishes; it only subscribes and then aborts the consumer after 2 s | `mqtt.rs:835-846` |
| Payload format (if you publish manually) | JSON with required `sensor_id` + `timestamp`; optional `temperature_c`, `humidity_pct`, `power_w` | `MqttTelemetryMessage` (`mqtt.rs:94-116`), `MqttTelemetryMessage::parse` (`mqtt.rs:119`) |
| Test lifetime | sleeps 2 s, aborts the consumer task, drains one message with `try_recv()` (result discarded) | `mqtt.rs:842-846` |

Two consequences worth knowing before you start:

1. **The broker must serve `localhost` with a certificate whose SAN contains
   `DNS:localhost`** — the client dials host `localhost` and rustls validates the
   name. A certificate with only a CN, or with `IP:127.0.0.1` only, will be rejected.
2. **The test passes even when no broker is running.** `connect()` only fails for a
   *configuration* error (`mqtt.rs:831-834` prints `Skipping integration test — no
   broker at …`), while `client.subscribe()` queues the SUBSCRIBE into rumqttc's
   event loop before the loop is ever polled. So "green" is **not** evidence that the
   handshake happened — use the broker log (Step 5) to prove the client connected and
   subscribed.

## Prerequisites

```bash
# Debian/Ubuntu
sudo apt-get install -y mosquitto mosquitto-clients openssl
```

* `mosquitto` ≥ 2.0 (broker), `mosquitto_pub` / `mosquitto_sub` (clients) from
  `mosquitto-clients`.
* `openssl` (certificate generation).
* `cargo` (the Rust toolchain pinned by `rust-toolchain.toml`).

Ports used: TCP `8883` on `localhost` only.

## Step 1 — generate a test CA and a `localhost` server certificate

All commands below are POSIX-shell and were verified with OpenSSL 3.0. Work in a
scratch directory outside the repository so nothing is accidentally committed:

```bash
export MQTT_TLS_DIR=/tmp/fluxion-mqtt-tls
rm -rf "$MQTT_TLS_DIR"
mkdir -p "$MQTT_TLS_DIR"
cd "$MQTT_TLS_DIR"

# 1a. Test CA (self-signed root).
openssl req -x509 -newkey rsa:2048 -sha256 -days 365 -nodes \
  -keyout ca.key -out ca.crt \
  -subj "/CN=Fluxion MQTT Test CA" \
  -addext "basicConstraints=critical,CA:TRUE" \
  -addext "keyUsage=critical,keyCertSign,cRLSign"

# 1b. Server key + CSR for `localhost`.
openssl req -newkey rsa:2048 -sha256 -nodes \
  -keyout server.key -out server.csr \
  -subj "/CN=localhost"

# 1c. Extension file: SAN is mandatory (rustls/webpki ignores CN).
cat > server.ext <<'EOF'
basicConstraints=CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=DNS:localhost,IP:127.0.0.1
EOF

# 1d. Sign the server certificate with the test CA.
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out server.crt -days 365 -sha256 -extfile server.ext

# 1e. Sanity check: SAN and chain must be right.
openssl x509 -in server.crt -noout -text | grep -A1 "Subject Alternative Name"
openssl verify -CAfile ca.crt server.crt
```

Expected output of the last two commands:

```
X509v3 Subject Alternative Name:
    DNS:localhost, IP Address:127.0.0.1
server.crt: OK
```

Keep the private key readable only by yourself (`chmod 600 ca.key server.key`).

## Step 2 — make the test CA trusted by the Rust client

The client builds its root store from the **platform trust store**
(`rustls-native-certs` via rumqttc's default TLS configuration). The Fluxion code
reads no CA path and no CA environment variable, so the CA has to be trusted by the
platform. Two ways, in order of preference:

### Option A — `SSL_CERT_FILE` (no root required, recommended)

`rustls-native-certs` (0.8.4, pinned in `Cargo.lock`) honours `SSL_CERT_FILE` /
`SSL_CERT_DIR`: when either is set, **only** the certificates from those
locations are loaded. Point `SSL_CERT_FILE` at the test CA and no system-wide
change is needed:

```bash
export SSL_CERT_FILE="$MQTT_TLS_DIR/ca.crt"
```

Remember to unset it during teardown (`unset SSL_CERT_FILE`).

### Option B — install the CA into the OS trust store (needs root)

```bash
# Debian/Ubuntu
sudo cp "$MQTT_TLS_DIR/ca.crt" /usr/local/share/ca-certificates/fluxion-mqtt-test-ca.crt
sudo update-ca-certificates
# remove again with:
# sudo rm /usr/local/share/ca-certificates/fluxion-mqtt-test-ca.crt && sudo update-ca-certificates
```

```bash
# macOS (system keychain)
sudo security add-trusted-cert -d -r trustRoot \
  -k /Library/Keychains/System.keychain "$MQTT_TLS_DIR/ca.crt"
# remove again with:
# sudo security delete-cert -c "Fluxion MQTT Test CA" \
#   -k /Library/Keychains/System.keychain
```

Option B is global to the machine — prefer Option A unless you need the CA trusted
for other tools (e.g. `mosquitto_pub` without `--cafile`).

## Step 3 — write the mosquitto configuration

```bash
cat > "$MQTT_TLS_DIR/mosquitto.conf" <<EOF
# Listener for the TLS-only broker the test expects (mqtts://localhost:8883).
listener 8883

# The Fluxion consumer sends no username/password (mqtt.rs never calls
# set_credentials), so the broker must accept anonymous clients.
allow_anonymous true

# Server certificate + the CA that signed it.
cafile $MQTT_TLS_DIR/ca.crt
certfile $MQTT_TLS_DIR/server.crt
keyfile $MQTT_TLS_DIR/server.key

# The Rust client presents no client certificate (with_no_client_auth).
require_certificate false

tls_version tlsv1.2

persistence false
log_type all
log_dest stdout
EOF
```

Notes:

* Do **not** set `use_identity_as_username true` together with a password file —
  the client sends no credentials at all.
* Without an `acl_file`, mosquitto grants access to all topics, so the
  `fluxion/test/zone1` subscription and publish are allowed by default.
* Run the broker as your own user (not via `sudo`), so it does not drop
  privileges to the `mosquitto` account and lose access to `server.key`.

## Step 4 — start the broker

```bash
mosquitto -c "$MQTT_TLS_DIR/mosquitto.conf" -v
```

Leave it running in its own terminal. On startup you should see the listener
announcement; there is no plaintext listener, so nothing answers on 1883.

Optional smoke test from a second terminal (proves the CA + SAN work):

```bash
mosquitto_pub -h localhost -p 8883 \
  --cafile "$MQTT_TLS_DIR/ca.crt" \
  -t fluxion/test/zone1 \
  -m '{"sensor_id":"zone-1-temp","timestamp":1700000000,"temperature_c":22.5}'
```

A `Connection error: Connection Refused: not authorised` or a TLS error means the
CA/SAN wiring is wrong — re-check Steps 1 and 2 before running the test.

## Step 5 — run the test

The test lives in the library unit-test module of the `fluxion-twin` crate, so it
runs with `--lib` (its full path is
`telemetry::mqtt::tests::test_mqtt_consumer_integration`):

```bash
cd <repo-root>

cargo test -p fluxion-twin --lib \
  telemetry::mqtt::tests::test_mqtt_consumer_integration \
  -- --ignored --exact --nocapture
```

If you are using Option A (`SSL_CERT_FILE`), export it in the same shell:

```bash
SSL_CERT_FILE="$MQTT_TLS_DIR/ca.crt" \
  cargo test -p fluxion-twin --lib \
    telemetry::mqtt::tests::test_mqtt_consumer_integration \
    -- --ignored --exact --nocapture
```

Expected result: the test is reported as `ok` and the run takes ~2 s (the test
sleeps 2 s before aborting the consumer task).

**Verify the handshake in the broker log, not in the test result** (see the
"passes even without a broker" note above). In the terminal running mosquitto you
should see, while the test runs:

```
New connection from 127.0.0.1:… on port 8883.
New client connected … as … (p2, c1, k60).
… Sending SUBSCRIBE (d1, q1, m1) 'fluxion/test/zone1'
```

If instead you see repeated `Error: Socket error on client <unknown>, disconnecting.`
or `Connection error: …`, the TLS handshake failed — most often an untrusted CA
(Step 2) or a missing `DNS:localhost` SAN (Step 1).

### Exercising the message path (optional)

The test itself never publishes, so to also cover the JSON parse path, publish a
message to the subscribed topic while the test is running (the consumer is alive
for the first 2 s of the test):

```bash
mosquitto_pub -h localhost -p 8883 \
  --cafile "$MQTT_TLS_DIR/ca.crt" \
  -t fluxion/test/zone1 \
  -m '{"sensor_id":"zone-1-temp","timestamp":1700000000,"temperature_c":22.5,"humidity_pct":45.0,"power_w":150.0}'
```

The payload must contain `sensor_id` and `timestamp`; without them the consumer
counts the message as `outcome="error"` (`fluxion_twin_mqtt_messages_total`) and
drops it. The test discards the drained message (`let _ = rx.try_recv();`), so the
observable proof is again the broker log showing the PUBLISH reaching the
subscriber.

## Step 6 — tear down

```bash
# 1. Stop the broker (Ctrl-C in its terminal, or:)
pkill -f "mosquitto -c $MQTT_TLS_DIR/mosquitto.conf"

# 2. Drop the trust-store override (Option A).
unset SSL_CERT_FILE

# 3. Remove the CA from the system trust store if you used Option B.
sudo rm -f /usr/local/share/ca-certificates/fluxion-mqtt-test-ca.crt
sudo update-ca-certificates
# macOS: sudo security delete-cert -c "Fluxion MQTT Test CA" -k /Library/Keychains/System.keychain

# 4. Delete the scratch directory (CA, server cert, key, config).
rm -rf "$MQTT_TLS_DIR"
```

Nothing in the repository is modified by this procedure, so the working tree stays
clean after teardown.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `invalid peer certificate: UnknownIssuer` | CA not trusted by the client | Step 2 (Option A `SSL_CERT_FILE` or Option B system store) |
| `invalid peer certificate: NotValidForName` | server cert lacks `DNS:localhost` in the SAN | re-generate with `server.ext` (Step 1c) |
| `could not load platform certs` panic | `SSL_CERT_FILE` points at a missing/empty/invalid PEM | `unset SSL_CERT_FILE` or point it at `ca.crt` |
| `Connection error: Connection Refused: not authorised` | broker requires credentials or a client cert | `allow_anonymous true`, `require_certificate false` (Step 3) |
| test is green but the broker log shows no client | the test never fails without a broker — read the "passes even without a broker" note | publish with `mosquitto_pub` and check the log |
| `error: no test target named …` | `--test <name>` was used | the test is a unit test: use `--lib` (Step 5) |

## Code evidence appendix

The complete test, verbatim from
[`crates/fluxion-twin/src/telemetry/mqtt.rs`](../crates/fluxion-twin/src/telemetry/mqtt.rs)
(lines 821-847):

```rust
#[tokio::test]
#[ignore = "requires a running TLS MQTT broker at mqtts://localhost:8883"]
async fn test_mqtt_consumer_integration() {
    // Requires a validated TLS broker at mqtts://localhost:8883.
    // There is no bypass for certificate verification (Issue #3162).
    let broker = "mqtts://localhost:8883";
    let topic = "fluxion/test/zone1";

    let consumer = MqttTelemetryConsumer::connect(broker, topic).await;
    if consumer.is_err() {
        eprintln!("Skipping integration test — no broker at {broker}");
        return;
    }
    let (consumer, mut rx) = consumer.unwrap();

    let handle = tokio::spawn(async move {
        let _ = consumer.start().await;
    });

    // Give the consumer a moment, then abort.
    tokio::time::sleep(Duration::from_secs(2)).await;
    handle.abort();

    // Drain anything received (likely nothing without a publisher).
    let _ = rx.try_recv();
}
```

And the transport setup it relies on (lines 197-202):

```rust
let mut mqttoptions = MqttOptions::new("fluxion-twin-consumer", host, port);
mqttoptions.set_keep_alive(Duration::from_secs(5));
mqttoptions.set_transport(Transport::tls_with_default_config());

let (client, eventloop) = AsyncClient::new(mqttoptions, CHANNEL_CAPACITY);
client.subscribe(topic, QoS::AtLeastOnce).await?;
```

`Transport::tls_with_default_config()` is rumqttc's default TLS configuration: the
root store is filled from `rustls_native_certs::load_native_certs()` (the platform
trust store) and the client is built with `with_no_client_auth()`. There is no
client certificate, no CA file parameter, and no credential call anywhere in
`crates/fluxion-twin/src/telemetry/mqtt.rs` — hence the broker requirements above.
