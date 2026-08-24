// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! LiveTwin WebSocket server with MessagePack binary serialization (Issue #2062).
//!
//! Provides real-time state streaming via WebSocket at `ws://0.0.0.0:8080/live-twin`.
//! Uses MessagePack (rmp-serde) for compact binary encoding (~2x bandwidth savings vs JSON).
//!
//! # Example
//!
//! ```ignore
//! let broadcaster = LiveTwinBroadcaster::new();
//! let payload = LiveTwinPayload {
//!     timestamp: Utc::now(),
//!     simulation_id: Uuid::new_v4(),
//!     zone_states: vec![],
//! };
//! broadcaster.broadcast(&payload).ok();
//! ```

use std::net::SocketAddr;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use parking_lot::Mutex;
use rmp_serde::Serializer;
use serde::{Deserialize, Serialize};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc};
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;
use uuid::Uuid;

use crate::twin::connection_pool::{ConnectionId, ConnectionPool};

pub const DEFAULT_LIVE_TWIN_PORT: u16 = 8080;
pub const LIVE_TWIN_PATH: &str = "/live-twin";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneState {
    pub zone_id: usize,
    pub t_air: f64,
    pub t_mass: f64,
    pub t_surface: f64,
    pub rh: f64,
    pub heating_setpoint: f64,
    pub cooling_setpoint: f64,
    pub heating_demand: f64,
    pub cooling_demand: f64,
    pub hvac_power_kw: f64,
    pub energy_heating_kwh: f64,
    pub energy_cooling_kwh: f64,
    pub occupancy: f64,
}

impl ZoneState {
    pub fn new(zone_id: usize) -> Self {
        Self {
            zone_id,
            t_air: 20.0,
            t_mass: 20.0,
            t_surface: 20.0,
            rh: 50.0,
            heating_setpoint: 21.0,
            cooling_setpoint: 26.0,
            heating_demand: 0.0,
            cooling_demand: 0.0,
            hvac_power_kw: 0.0,
            energy_heating_kwh: 0.0,
            energy_cooling_kwh: 0.0,
            occupancy: 0.0,
        }
    }

    pub fn with_temperatures(mut self, t_air: f64, t_mass: f64, t_surface: f64) -> Self {
        self.t_air = t_air;
        self.t_mass = t_mass;
        self.t_surface = t_surface;
        self
    }

    pub fn with_setpoints(mut self, heating: f64, cooling: f64) -> Self {
        self.heating_setpoint = heating;
        self.cooling_setpoint = cooling;
        self
    }

    pub fn with_hvac_demand(mut self, heating: f64, cooling: f64, power_kw: f64) -> Self {
        self.heating_demand = heating;
        self.cooling_demand = cooling;
        self.hvac_power_kw = power_kw;
        self
    }

    pub fn with_energy(mut self, heating_kwh: f64, cooling_kwh: f64) -> Self {
        self.energy_heating_kwh = heating_kwh;
        self.energy_cooling_kwh = cooling_kwh;
        self
    }

    pub fn with_rh_occupancy(mut self, rh: f64, occupancy: f64) -> Self {
        self.rh = rh;
        self.occupancy = occupancy;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveTwinPayload {
    pub timestamp: DateTime<Utc>,
    pub simulation_id: Uuid,
    pub zone_states: Vec<ZoneState>,
}

#[derive(Debug, Clone)]
pub enum BroadcastError {
    Serialization(String),
    NoConnections,
}

impl std::fmt::Display for BroadcastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serialization(e) => write!(f, "MessagePack serialization failed: {e}"),
            Self::NoConnections => write!(f, "No active WebSocket connections to broadcast to"),
        }
    }
}

impl std::error::Error for BroadcastError {}

pub struct LiveTwinBroadcaster {
    pool: ConnectionPool,
    connections: Arc<Mutex<Vec<ConnectionEntry>>>,
    payload_tx: broadcast::Sender<LiveTwinPayload>,
}

struct ConnectionEntry {
    id: ConnectionId,
    sender: mpsc::Sender<Message>,
}

impl LiveTwinBroadcaster {
    pub fn new() -> Self {
        let (payload_tx, _) = broadcast::channel(100);
        Self {
            pool: ConnectionPool::default(),
            connections: Arc::new(Mutex::new(Vec::new())),
            payload_tx,
        }
    }

    pub fn with_capacity(max_connections: usize) -> Self {
        let (payload_tx, _) = broadcast::channel(100);
        Self {
            pool: ConnectionPool::new(max_connections),
            connections: Arc::new(Mutex::new(Vec::new())),
            payload_tx,
        }
    }

    pub fn pool(&self) -> &ConnectionPool {
        &self.pool
    }

    pub fn active_connection_count(&self) -> usize {
        self.connections.lock().len()
    }

    pub async fn broadcast(&self, payload: &LiveTwinPayload) -> Result<(), BroadcastError> {
        let mut buf = Vec::new();
        payload
            .serialize(&mut Serializer::new(&mut buf))
            .map_err(|e| BroadcastError::Serialization(e.to_string()))?;

        let msg = Message::Binary(buf);
        let senders: Vec<_> = self
            .connections
            .lock()
            .iter()
            .map(|conn| conn.sender.clone())
            .collect();

        if senders.is_empty() {
            return Err(BroadcastError::NoConnections);
        }

        for sender in senders {
            sender.send(msg.clone()).await.ok();
        }

        Ok(())
    }

    pub async fn serve(
        self,
        addr: SocketAddr,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let listener = TcpListener::bind(addr).await?;
        log::info!("LiveTwin WebSocket server listening on {}", addr);

        let connections = Arc::clone(&self.connections);
        let pool = Arc::new(self.pool);
        let payload_tx = self.payload_tx;

        loop {
            match listener.accept().await {
                Ok((stream, peer_addr)) => {
                    let connections = Arc::clone(&connections);
                    let pool = Arc::clone(&pool);
                    let payload_rx = payload_tx.subscribe();

                    tokio::spawn(async move {
                        if let Err(e) =
                            handle_connection(stream, peer_addr, connections, pool, payload_rx)
                                .await
                        {
                            log::error!("Connection error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    log::error!("Failed to accept connection: {}", e);
                }
            }
        }
    }

    pub fn broadcast_channel(&self) -> broadcast::Sender<LiveTwinPayload> {
        self.payload_tx.clone()
    }
}

async fn handle_connection(
    stream: TcpStream,
    peer_addr: SocketAddr,
    connections: Arc<Mutex<Vec<ConnectionEntry>>>,
    pool: Arc<ConnectionPool>,
    mut payload_rx: broadcast::Receiver<LiveTwinPayload>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let ws_stream = accept_async(stream).await?;
    let (mut write, mut read) = ws_stream.split();
    let conn_id = ConnectionId::new();

    log::info!("New LiveTwin connection from {}: {:?}", peer_addr, conn_id);

    if let Err(e) = pool.add(conn_id) {
        log::warn!("Rejected connection from {}: {}", peer_addr, e);
        return Ok(());
    }

    let (tx, mut rx) = mpsc::channel::<Message>(100);

    {
        let mut conns = connections.lock();
        conns.push(ConnectionEntry {
            id: conn_id,
            sender: tx,
        });
    }

    let connections_clone = Arc::clone(&connections);
    let pool_clone = Arc::clone(&pool);

    tokio::spawn(async move {
        let mut ping_interval = tokio::time::interval(tokio::time::Duration::from_secs(30));

        loop {
            tokio::select! {
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Ping(data))) => {
                            if write.send(Message::Pong(data)).await.is_err() {
                                break;
                            }
                        }
                        Some(Ok(Message::Pong(_))) => {
                            pool_clone.handle_pong(conn_id);
                        }
                        Some(Ok(Message::Close(_))) | None => {
                            log::info!("Connection {:?} closed", conn_id);
                            break;
                        }
                        Some(Ok(_)) => {}
                        Some(Err(e)) => {
                            log::warn!("WebSocket error for {:?}: {}", conn_id, e);
                            break;
                        }
                    }
                }
                _ = ping_interval.tick() => {
                    pool_clone.mark_pending_ping(conn_id);
                    if write.send(Message::Ping(vec![].into())).await.is_err() {
                        break;
                    }
                }
                payload = payload_rx.recv() => {
                    if let Ok(p) = payload {
                        let mut buf = Vec::new();
                        if p.serialize(&mut Serializer::new(&mut buf)).is_ok()
                            && write.send(Message::Binary(buf.into())).await.is_err()
                        {
                            break;
                        }
                    }
                }
                msg = rx.recv() => {
                    if let Some(msg) = msg {
                        if write.send(msg).await.is_err() {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        pool_clone.mark_disconnected(conn_id);
        let mut conns = connections_clone.lock();
        conns.retain(|c| c.id != conn_id);
        log::info!("Connection {:?} removed", conn_id);
    });

    Ok(())
}

impl Default for LiveTwinBroadcaster {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveTwinPayload {
    pub fn new(simulation_id: Uuid) -> Self {
        Self {
            timestamp: Utc::now(),
            simulation_id,
            zone_states: Vec::new(),
        }
    }

    pub fn with_zones(mut self, zone_states: Vec<ZoneState>) -> Self {
        self.zone_states = zone_states;
        self
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn from_zone_telemetry(
        simulation_id: Uuid,
        zone_temps: &[f64],
        zone_setpoints: &[(f64, f64)],
        zone_hvac_power: &[f64],
        zone_energy_heating: &[f64],
        zone_energy_cooling: &[f64],
    ) -> Self {
        let zone_states: Vec<ZoneState> = zone_temps
            .iter()
            .enumerate()
            .map(|(i, &t_air)| {
                let (heating_sp, cooling_sp) = zone_setpoints
                    .get(i)
                    .copied()
                    .unwrap_or((21.0, 26.0));
                let hvac_power = zone_hvac_power.get(i).copied().unwrap_or(0.0);
                let energy_heating = zone_energy_heating.get(i).copied().unwrap_or(0.0);
                let energy_cooling = zone_energy_cooling.get(i).copied().unwrap_or(0.0);

                ZoneState::new(i)
                    .with_temperatures(t_air, t_air - 0.5, t_air - 1.5)
                    .with_setpoints(heating_sp, cooling_sp)
                    .with_hvac_demand(0.0, 0.0, hvac_power)
                    .with_energy(energy_heating, energy_cooling)
            })
            .collect();

        Self {
            timestamp: Utc::now(),
            simulation_id,
            zone_states,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingConfig {
    pub building_id: String,
    pub num_zones: usize,
    pub zone_names: Vec<String>,
    pub zone_areas: Vec<f64>,
    pub heating_setpoints: Vec<f64>,
    pub cooling_setpoints: Vec<f64>,
    pub thermal_masses: Vec<f64>,
}

impl BuildingConfig {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn create_zone_states(&self) -> Vec<ZoneState> {
        self.zone_names
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let heating_sp = self.heating_setpoints.get(i).copied().unwrap_or(21.0);
                let cooling_sp = self.cooling_setpoints.get(i).copied().unwrap_or(26.0);
                ZoneState::new(i)
                    .with_setpoints(heating_sp, cooling_sp)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zone_state_serialization() {
        let zone = ZoneState {
            zone_id: 0,
            t_air: 22.5,
            t_mass: 21.8,
            t_surface: 20.2,
            rh: 45.0,
            heating_setpoint: 21.0,
            cooling_setpoint: 26.0,
            heating_demand: 150.0,
            cooling_demand: 0.0,
            hvac_power_kw: 2.5,
            energy_heating_kwh: 125.5,
            energy_cooling_kwh: 0.0,
            occupancy: 0.75,
        };

        let mut buf = Vec::new();
        zone.serialize(&mut Serializer::new(&mut buf)).unwrap();
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_live_twin_payload_msgpack_size() {
        let payload = LiveTwinPayload::new(Uuid::new_v4()).with_zones(vec![
            ZoneState {
                zone_id: 0,
                t_air: 22.5,
                t_mass: 21.8,
                t_surface: 20.2,
                rh: 45.0,
                heating_setpoint: 21.0,
                cooling_setpoint: 26.0,
                heating_demand: 150.0,
                cooling_demand: 0.0,
                hvac_power_kw: 2.5,
                energy_heating_kwh: 125.5,
                energy_cooling_kwh: 0.0,
                occupancy: 0.75,
            },
            ZoneState {
                zone_id: 1,
                t_air: 23.0,
                t_mass: 22.5,
                t_surface: 21.0,
                rh: 50.0,
                heating_setpoint: 21.0,
                cooling_setpoint: 26.0,
                heating_demand: 0.0,
                cooling_demand: 200.0,
                hvac_power_kw: 3.2,
                energy_heating_kwh: 0.0,
                energy_cooling_kwh: 88.3,
                occupancy: 0.50,
            },
        ]);

        let mut buf = Vec::new();
        payload.serialize(&mut Serializer::new(&mut buf)).unwrap();

        let json_size = serde_json::to_vec(&payload).unwrap().len();
        let msgpack_size = buf.len();

        println!("JSON size: {} bytes", json_size);
        println!("MessagePack size: {} bytes", msgpack_size);
        assert!(
            msgpack_size < json_size,
            "MessagePack should be smaller than JSON"
        );
    }

    #[test]
    fn test_live_twin_payload_new() {
        let simulation_id = Uuid::new_v4();
        let payload = LiveTwinPayload::new(simulation_id);

        assert_eq!(payload.simulation_id, simulation_id);
        assert!(payload.zone_states.is_empty());
    }

    #[test]
    fn test_broadcaster_default() {
        let broadcaster = LiveTwinBroadcaster::default();
        assert_eq!(broadcaster.active_connection_count(), 0);
    }

    #[test]
    fn test_broadcaster_with_capacity() {
        let broadcaster = LiveTwinBroadcaster::with_capacity(50);
        assert_eq!(broadcaster.pool().max_connections(), 50);
    }

    #[tokio::test]
    async fn test_broadcast_no_connections() {
        let broadcaster = LiveTwinBroadcaster::new();
        let payload = LiveTwinPayload::new(Uuid::new_v4());

        let result = broadcaster.broadcast(&payload).await;
        assert!(matches!(result, Err(BroadcastError::NoConnections)));
    }

    #[test]
    fn test_zone_state_builder() {
        let zone = ZoneState::new(0)
            .with_temperatures(22.5, 21.8, 20.2)
            .with_setpoints(21.0, 26.0)
            .with_hvac_demand(150.0, 0.0, 2.5)
            .with_energy(125.5, 0.0)
            .with_rh_occupancy(45.0, 0.75);

        assert_eq!(zone.zone_id, 0);
        assert!((zone.t_air - 22.5).abs() < 1e-10);
        assert!((zone.t_mass - 21.8).abs() < 1e-10);
        assert!((zone.t_surface - 20.2).abs() < 1e-10);
        assert!((zone.heating_setpoint - 21.0).abs() < 1e-10);
        assert!((zone.cooling_setpoint - 26.0).abs() < 1e-10);
        assert!((zone.heating_demand - 150.0).abs() < 1e-10);
        assert!((zone.hvac_power_kw - 2.5).abs() < 1e-10);
        assert!((zone.energy_heating_kwh - 125.5).abs() < 1e-10);
        assert!((zone.rh - 45.0).abs() < 1e-10);
        assert!((zone.occupancy - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_live_twin_payload_from_zone_telemetry() {
        let sim_id = Uuid::new_v4();
        let temps = vec![22.5, 23.0, 21.0];
        let setpoints = vec![(21.0, 26.0), (21.0, 26.0), (20.0, 25.0)];
        let hvac_power = vec![2.5, 3.2, 1.8];
        let energy_heating = vec![125.5, 0.0, 200.0];
        let energy_cooling = vec![0.0, 88.3, 50.0];

        let payload = LiveTwinPayload::from_zone_telemetry(
            sim_id,
            &temps,
            &setpoints,
            &hvac_power,
            &energy_heating,
            &energy_cooling,
        );

        assert_eq!(payload.simulation_id, sim_id);
        assert_eq!(payload.zone_states.len(), 3);
        assert!((payload.zone_states[0].t_air - 22.5).abs() < 1e-10);
        assert!((payload.zone_states[1].t_air - 23.0).abs() < 1e-10);
        assert!((payload.zone_states[2].t_air - 21.0).abs() < 1e-10);
    }

    #[test]
    fn test_building_config_json_roundtrip() {
        let config = BuildingConfig {
            building_id: "test_building".to_string(),
            num_zones: 3,
            zone_names: vec!["Zone A".to_string(), "Zone B".to_string(), "Zone C".to_string()],
            zone_areas: vec![100.0, 150.0, 200.0],
            heating_setpoints: vec![21.0, 21.0, 20.0],
            cooling_setpoints: vec![26.0, 26.0, 25.0],
            thermal_masses: vec![5e6, 7.5e6, 10e6],
        };

        let json = config.to_json().unwrap();
        let decoded = BuildingConfig::from_json(&json).unwrap();

        assert_eq!(decoded.building_id, "test_building");
        assert_eq!(decoded.num_zones, 3);
        assert_eq!(decoded.zone_names.len(), 3);
        assert_eq!(decoded.heating_setpoints, vec![21.0, 21.0, 20.0]);
    }

    #[test]
    fn test_building_config_create_zone_states() {
        let config = BuildingConfig {
            building_id: "test".to_string(),
            num_zones: 2,
            zone_names: vec!["Zone 1".to_string(), "Zone 2".to_string()],
            zone_areas: vec![100.0, 150.0],
            heating_setpoints: vec![21.0, 22.0],
            cooling_setpoints: vec![26.0, 27.0],
            thermal_masses: vec![5e6, 7.5e6],
        };

        let zone_states = config.create_zone_states();
        assert_eq!(zone_states.len(), 2);
        assert!((zone_states[0].heating_setpoint - 21.0).abs() < 1e-10);
        assert!((zone_states[1].heating_setpoint - 22.0).abs() < 1e-10);
    }

    #[test]
    fn test_live_twin_payload_json_serialization() {
        let payload = LiveTwinPayload::new(Uuid::new_v4()).with_zones(vec![
            ZoneState::new(0).with_temperatures(22.5, 21.8, 20.2),
            ZoneState::new(1).with_temperatures(23.0, 22.5, 21.0),
        ]);

        let json = payload.to_json().unwrap();
        let decoded = LiveTwinPayload::from_json(&json).unwrap();

        assert_eq!(decoded.zone_states.len(), 2);
        assert!((decoded.zone_states[0].t_air - 22.5).abs() < 1e-10);
        assert!((decoded.zone_states[1].t_air - 23.0).abs() < 1e-10);
    }
}
