"""
Real-time Monitoring and BAS Integration Module

This module provides real-time monitoring capabilities and Building Automation System
(BAS) integration for live operations.
Implements Issue #219: feat(api): Implement real-time monitoring and BAS integration
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Callable
from enum import Enum
import asyncio
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Router for monitoring endpoints
router = APIRouter(prefix="/monitoring", tags=["monitoring"])


class MetricType(str, Enum):
    """Types of metrics that can be monitored."""
    TEMPERATURE = "temperature"
    ENERGY = "energy"
    POWER = "power"
    HUMIDITY = "humidity"
    CO2 = "co2"
    OCCUPANCY = "occupancy"
    HVAC_STATUS = "hvac_status"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class KPI(BaseModel):
    """Key Performance Indicator."""
    name: str
    value: float
    unit: str
    target: Optional[float] = None
    status: str = "normal"  # normal, warning, critical


class Alert(BaseModel):
    """Alert notification."""
    id: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    zone_id: Optional[str] = None
    acknowledged: bool = False


class MetricReading(BaseModel):
    """A single metric reading."""
    metric_type: MetricType
    value: float
    unit: str
    zone_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class MonitoringData(BaseModel):
    """Complete monitoring snapshot."""
    zone_id: str
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    co2: Optional[float] = None
    energy_consumption: Optional[float] = None
    hvac_mode: Optional[str] = None
    heating_setpoint: Optional[float] = None
    cooling_setpoint: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[str, set] = {}  # zone_id -> set of websockets
    
    async def connect(self, websocket: WebSocket, zone_id: Optional[str] = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if zone_id:
            if zone_id not in self.subscriptions:
                self.subscriptions[zone_id] = set()
            self.subscriptions[zone_id].add(websocket)
        
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Remove from subscriptions
        for zone_id in list(self.subscriptions.keys()):
            if websocket in self.subscriptions[zone_id]:
                self.subscriptions[zone_id].remove(websocket)
        
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        if self.active_connections:
            await asyncio.gather(
                *[connection.send_json(message) for connection in self.active_connections],
                return_exceptions=True
            )
    
    async def send_to_zone(self, zone_id: str, message: dict):
        """Send a message to clients subscribed to a specific zone."""
        if zone_id in self.subscriptions:
            await asyncio.gather(
                *[conn.send_json(message) for conn in self.subscriptions[zone_id]],
                return_exceptions=True
            )


# Global connection manager
manager = ConnectionManager()


# In-memory storage for metrics and alerts
class MonitoringState:
    def __init__(self):
        self.metrics_history: Dict[str, List[MetricReading]] = {}
        self.alerts: List[Alert] = []
        self.kpis: Dict[str, KPI] = {}
        self.zones: Dict[str, MonitoringData] = {}
    
    def add_metric(self, zone_id: str, reading: MetricReading):
        """Add a metric reading to history."""
        key = f"{zone_id}_{reading.metric_type.value}"
        if key not in self.metrics_history:
            self.metrics_history[key] = []
        
        self.metrics_history[key].append(reading)
        
        # Keep only last 1000 readings per metric
        if len(self.metrics_history[key]) > 1000:
            self.metrics_history[key] = self.metrics_history[key][-1000:]
    
    def add_alert(self, alert: Alert):
        """Add a new alert."""
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def update_kpi(self, kpi: KPI):
        """Update a KPI value."""
        self.kpis[kpi.name] = kpi
    
    def get_zone_data(self, zone_id: str) -> Optional[MonitoringData]:
        """Get current data for a zone."""
        return self.zones.get(zone_id)
    
    def update_zone(self, data: MonitoringData):
        """Update zone data."""
        self.zones[data.zone_id] = data


state = MonitoringState()


# Initialize default KPIs
def init_kpis():
    """Initialize default KPIs."""
    default_kpis = [
        KPI(name="total_energy", value=0.0, unit="kWh", target=1000.0),
        KPI(name="peak_power", value=0.0, unit="kW", target=50.0),
        KPI(name="average_temperature", value=22.0, unit="°C", target=22.0, status="normal"),
        KPI(name="hvac_efficiency", value=3.5, unit="COP", target=3.0, status="normal"),
        KPI(name="occupancy_count", value=0.0, unit="persons", target=50.0),
        KPI(name="comfort_index", value=95.0, unit="%", target=90.0, status="normal"),
    ]
    for kpi in default_kpis:
        state.update_kpi(kpi)


init_kpis()


# WebSocket endpoint for real-time updates
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, zone_id: Optional[str] = None):
    """
    WebSocket endpoint for real-time monitoring updates.
    
    Connect with: ws://host:port/monitoring/ws?zone_id=zone1
    
    Messages received:
    - subscribe: Subscribe to specific zone updates
    - unsubscribe: Unsubscribe from zone updates
    
    Messages sent:
    - metric: New metric reading
    - alert: New alert notification
    - kpi: KPI update
    - zone_update: Complete zone data update
    """
    await manager.connect(websocket, zone_id)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                msg_type = message.get("type")
                
                if msg_type == "subscribe":
                    target_zone = message.get("zone_id")
                    if target_zone:
                        await manager.disconnect(websocket)
                        await manager.connect(websocket, target_zone)
                        await websocket.send_json({
                            "type": "subscribed",
                            "zone_id": target_zone
                        })
                
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
                    
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# REST endpoints
@router.get("/zones", response_model=List[str])
async def get_zones():
    """Get list of all monitored zones."""
    return list(state.zones.keys())


@router.get("/zones/{zone_id}", response_model=MonitoringData)
async def get_zone(zone_id: str):
    """Get current monitoring data for a specific zone."""
    data = state.get_zone_data(zone_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id} not found")
    return data


@router.post("/zones/{zone_id}")
async def update_zone(zone_id: str, data: MonitoringData):
    """Update monitoring data for a zone."""
    data.zone_id = zone_id
    state.update_zone(data)
    
    # Broadcast update to subscribers
    await manager.send_to_zone(zone_id, {
        "type": "zone_update",
        "data": data.model_dump()
    })
    
    return {"status": "success"}


@router.get("/metrics/{zone_id}/{metric_type}", response_model=List[MetricReading])
async def get_metrics(zone_id: str, metric_type: MetricType):
    """Get metric history for a zone."""
    key = f"{zone_id}_{metric_type.value}"
    return state.metrics_history.get(key, [])


@router.post("/metrics")
async def add_metric(reading: MetricReading):
    """Add a new metric reading."""
    if reading.zone_id:
        state.add_metric(reading.zone_id, reading)
        
        # Broadcast to subscribers
        await manager.send_to_zone(reading.zone_id, {
            "type": "metric",
            "data": reading.model_dump()
        })
    
    return {"status": "success"}


@router.get("/alerts", response_model=List[Alert])
async def get_alerts(acknowledged: Optional[bool] = None):
    """Get all alerts, optionally filtered by acknowledged status."""
    if acknowledged is not None:
        return [a for a in state.alerts if a.acknowledged == acknowledged]
    return state.alerts


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    for alert in state.alerts:
        if alert.id == alert_id:
            alert.acknowledged = True
            return {"status": "success", "alert_id": alert_id}
    
    raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")


@router.post("/alerts")
async def create_alert(alert: Alert):
    """Create a new alert."""
    state.add_alert(alert)
    
    # Broadcast alert
    await manager.broadcast({
        "type": "alert",
        "data": alert.model_dump()
    })
    
    return {"status": "success", "alert_id": alert.id}


@router.get("/kpis", response_model=List[KPI])
async def get_kpis():
    """Get all KPIs."""
    return list(state.kpis.values())


@router.get("/kpis/{kpi_name}", response_model=KPI)
async def get_kpi(kpi_name: str):
    """Get a specific KPI."""
    if kpi_name not in state.kpis:
        raise HTTPException(status_code=404, detail=f"KPI {kpi_name} not found")
    return state.kpis[kpi_name]


@router.post("/kpis")
async def update_kpi(kpi: KPI):
    """Update a KPI."""
    # Determine status based on target
    if kpi.target is not None:
        if abs(kpi.value - kpi.target) / kpi.target > 0.1:
            kpi.status = "critical"
        elif abs(kpi.value - kpi.target) / kpi.target > 0.05:
            kpi.status = "warning"
        else:
            kpi.status = "normal"
    
    state.update_kpi(kpi)
    
    # Broadcast update
    await manager.broadcast({
        "type": "kpi",
        "data": kpi.model_dump()
    })
    
    return {"status": "success"}


# BACnet/Modbus integration placeholder
class BASIntegration:
    """
    Building Automation System integration base class.
    Extend this to implement BACnet/IP or Modbus protocols.
    """
    
    def __init__(self):
        self.connected = False
        self.device_address = None
    
    async def connect(self, address: str, protocol: str = "bacnet"):
        """Connect to a BAS device."""
        # Placeholder - implement actual protocol connection
        self.device_address = address
        self.connected = True
        logger.info(f"Connected to BAS at {address} using {protocol}")
    
    async def disconnect(self):
        """Disconnect from BAS device."""
        self.connected = False
        self.device_address = None
    
    async def read_value(self, object_id: str):
        """Read a value from the BAS."""
        if not self.connected:
            raise RuntimeError("Not connected to BAS")
        # Placeholder - implement actual read
        return 0.0
    
    async def write_value(self, object_id: str, value: float):
        """Write a value to the BAS."""
        if not self.connected:
            raise RuntimeError("Not connected to BAS")
        # Placeholder - implement actual write
        pass
    
    async def read_multiple(self, object_ids: List[str]) -> Dict[str, float]:
        """Read multiple values from the BAS."""
        if not self.connected:
            raise RuntimeError("Not connected to BAS")
        # Placeholder - implement actual read
        return {oid: 0.0 for oid in object_ids}


# Global BAS instance
bas_integration = BASIntegration()


@router.post("/bas/connect")
async def connect_bas(request: dict):
    """
    Connect to a Building Automation System.
    
    Request body:
        address: str - IP address of the BAS device
        protocol: str - Protocol to use (bacnet or modbus)
    """
    address = request.get("address")
    protocol = request.get("protocol", "bacnet")
    
    if not address:
        raise HTTPException(status_code=400, detail="address is required")
    
    await bas_integration.connect(address, protocol)
    return {"status": "success", "message": f"Connected to BAS at {address}"}


@router.post("/bas/disconnect")
async def disconnect_bas():
    """Disconnect from the Building Automation System."""
    await bas_integration.disconnect()
    return {"status": "success", "message": "Disconnected from BAS"}


@router.get("/bas/status")
async def get_bas_status():
    """Get BAS connection status."""
    return {
        "connected": bas_integration.connected,
        "device_address": bas_integration.device_address
    }


@router.get("/bas/read/{object_id}")
async def bas_read_value(object_id: str):
    """Read a value from the BAS."""
    try:
        value = await bas_integration.read_value(object_id)
        return {"object_id": object_id, "value": value}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/bas/write/{object_id}")
async def bas_write_value(object_id: str, value: float):
    """Write a value to the BAS."""
    try:
        await bas_integration.write_value(object_id, value)
        return {"status": "success", "object_id": object_id, "value": value}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/bas/read-multiple")
async def bas_read_multiple(object_ids: List[str]):
    """Read multiple values from the BAS."""
    try:
        values = await bas_integration.read_multiple(object_ids)
        return {"values": values}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
