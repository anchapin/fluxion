#!/usr/bin/env python3
"""
xDT Inference Example - Real-time Weather and Sensor Data Input

This script demonstrates how to feed real-time weather and sensor data into an
exported xDT (executable Digital Twin) to predict thermal response for BMS
edge deployment.

The example shows:
1. Loading an exported xDT model (ONNX format)
2. Feeding real-time weather data (temperature, humidity, solar radiation)
3. Feeding sensor data (zone temperature, occupancy)
4. Running inference and interpreting results
5. Simulating a 24-hour BMS control cycle

Usage:
    # With dummy surrogate model (built-in test)
    python tools/xdt_inference_example.py

    # With custom exported model
    python tools/xdt_inference_example.py --model exports/xdt/model.onnx

    # Run a 24-hour simulation
    python tools/xdt_inference_example.py --simulate-24h --model exports/xdt/model.onnx

Requirements:
    pip install numpy onnxruntime

Issue: #977
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort

    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    print("Warning: onnxruntime not installed. Install with: pip install onnxruntime")
    print("Using mock inference for demonstration.")


@dataclass
class WeatherData:
    exterior_temp: float
    humidity: float
    solar_irradiance: float
    wind_speed: float = 0.0
    dew_point: float = 0.0

    def to_feature_vector(self) -> List[float]:
        return [
            self.exterior_temp,
            self.humidity,
            self.solar_irradiance,
            self.wind_speed,
        ]

    @classmethod
    def from_epw_hour(
        cls,
        dry_bulb: float,
        dew_point: float,
        dni: float,
        dhi: float,
        ghi: float,
        wind_speed: float = 0.0,
    ) -> "WeatherData":
        solar = (dni + dhi + ghi) / 3.0 if ghi > 0 else 0.0
        return cls(
            exterior_temp=dry_bulb,
            humidity=dew_point,
            solar_irradiance=solar,
            wind_speed=wind_speed,
            dew_point=dew_point,
        )


@dataclass
class SensorData:
    zone_temp: float
    heating_setpoint: float = 20.0
    cooling_setpoint: float = 24.0
    occupancy: float = 0.0
    internal_gains: float = 0.0

    def to_feature_vector(self) -> List[float]:
        return [
            self.zone_temp,
            self.heating_setpoint,
            self.cooling_setpoint,
            self.occupancy,
            self.internal_gains,
        ]


@dataclass
class XDTPrediction:
    heating_load: float
    cooling_load: float
    inference_time_ms: float
    confidence: float = 1.0

    def total_load(self) -> float:
        return self.heating_load + self.cooling_load

    def to_dict(self) -> Dict[str, float]:
        return {
            "heating_load_W": self.heating_load,
            "cooling_load_W": self.cooling_load,
            "total_load_W": self.total_load(),
            "inference_time_ms": self.inference_time_ms,
            "confidence": self.confidence,
        }


class SimulatedSensorStream:
    def __init__(self, base_weather: List[WeatherData], diurnal_pattern: bool = True):
        self.base_weather = base_weather
        self.diurnal_pattern = diurnal_pattern
        self.current_hour = 0

    def get_current_weather(self) -> WeatherData:
        if not self.base_weather:
            hour = self.current_hour % 24
            if self.diurnal_pattern:
                temp = 15 + 10 * np.sin(np.pi * (hour - 6) / 12)
            else:
                temp = 15.0

            solar = (
                max(0, 800 * np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
            )

            return WeatherData(
                exterior_temp=temp,
                humidity=50.0,
                solar_irradiance=solar,
                wind_speed=2.0,
                dew_point=10.0,
            )

        return self.base_weather[self.current_hour % len(self.base_weather)]

    def get_current_sensors(self, hour: int) -> SensorData:
        weather = self.get_current_weather()

        if 8 <= hour <= 18:
            occupancy = 0.2 + 0.6 * np.sin(np.pi * (hour - 8) / 8)
        else:
            occupancy = 0.05

        zone_temp = 20 + 0.3 * (weather.exterior_temp - 15) + np.random.normal(0, 0.2)

        return SensorData(
            zone_temp=zone_temp,
            heating_setpoint=20.0 if weather.exterior_temp < 18 else 22.0,
            cooling_setpoint=25.0 if weather.exterior_temp > 22 else 24.0,
            occupancy=occupancy,
            internal_gains=occupancy * 100,
        )

    def advance(self):
        self.current_hour = (self.current_hour + 1) % 24


class MockXDTInference:
    def __init__(self, input_dim: int = 9, output_dim: int = 2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.session = None
        self.ort = None

    def predict(self, features: np.ndarray) -> XDTPrediction:
        start = time.perf_counter()

        heating = max(0, (features[1] - features[0]) * 100 + np.random.normal(0, 5))
        cooling = max(0, (features[0] - features[2]) * 80 + np.random.normal(0, 5))

        if features[4] > 0.5:
            heating += features[4] * 50

        elapsed_ms = (time.perf_counter() - start) * 1000

        return XDTPrediction(
            heating_load=heating,
            cooling_load=cooling,
            inference_time_ms=elapsed_ms,
            confidence=0.95,
        )

    def predict_batch(self, features_batch: np.ndarray) -> List[XDTPrediction]:
        return [self.predict(f) for f in features_batch]


class RealXDTInference:
    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError("onnxruntime required for RealXDTInference")

        self.providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            model_path,
            providers=self.providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def predict(self, features: np.ndarray) -> XDTPrediction:
        start = time.perf_counter()

        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        outputs = self.session.run(
            [self.output_name], {self.input_name: features.astype(np.float32)}
        )
        result = outputs[0][0]

        elapsed_ms = (time.perf_counter() - start) * 1000

        heating = max(0, result[0]) if len(result) > 0 else 0.0
        cooling = max(0, result[1]) if len(result) > 1 else 0.0

        return XDTPrediction(
            heating_load=heating,
            cooling_load=cooling,
            inference_time_ms=elapsed_ms,
            confidence=0.98,
        )

    def predict_batch(self, features_batch: np.ndarray) -> List[XDTPrediction]:
        if len(features_batch.shape) == 1:
            features_batch = features_batch.reshape(1, -1)

        outputs = self.session.run(
            [self.output_name], {self.input_name: features_batch.astype(np.float32)}
        )
        results = outputs[0]

        return [
            XDTPrediction(
                heating_load=max(0, r[0]) if len(r) > 0 else 0.0,
                cooling_load=max(0, r[1]) if len(r) > 1 else 0.0,
                inference_time_ms=0.0,
                confidence=0.98,
            )
            for r in results
        ]


class XDTRuntime:
    inference: "MockXDTInference | RealXDTInference"

    def __init__(self, model_path: Optional[str] = None, use_mock: bool = False):
        if use_mock or not ONNXRUNTIME_AVAILABLE:
            print("Using MOCK inference (onnxruntime not available)")
            self.inference = MockXDTInference()
            self.is_mock = True
        else:
            print(f"Loading xDT model from: {model_path}")
            if model_path is None:
                raise ValueError("model_path is required for RealXDTInference")
            self.inference = RealXDTInference(model_path)
            self.is_mock = False

    def combine_features(
        self,
        weather: WeatherData,
        sensors: SensorData,
        time_features: Tuple[int, int, int],
    ) -> np.ndarray:
        hour_of_day, day_of_year, month = time_features

        features = np.array(
            weather.to_feature_vector()
            + sensors.to_feature_vector()
            + [
                float(hour_of_day) / 24.0,
                float(day_of_year) / 365.0,
                float(month) / 12.0,
                1.0 if hour_of_day in range(8, 18) else 0.0,
            ],
            dtype=np.float32,
        )
        return features

    def predict(
        self,
        weather: WeatherData,
        sensors: SensorData,
        time_features: Tuple[int, int, int],
    ) -> XDTPrediction:
        features = self.combine_features(weather, sensors, time_features)
        return self.inference.predict(features)

    def predict_24h_cycle(
        self,
        sensor_stream: SimulatedSensorStream,
        start_hour: int = 0,
        days: int = 1,
    ) -> List[Tuple[datetime, XDTPrediction, WeatherData, SensorData]]:
        results = []
        current_time = datetime.now().replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        )

        for day in range(days):
            for hour in range(24):
                weather = sensor_stream.get_current_weather()
                sensors = sensor_stream.get_current_sensors(hour)
                time_features = (
                    hour,
                    current_time.timetuple().tm_yday,
                    current_time.month,
                )

                prediction = self.predict(weather, sensors, time_features)

                results.append((current_time, prediction, weather, sensors))

                sensor_stream.advance()
                current_time += timedelta(hours=1)

        return results

    def run_bms_control_simulation(
        self,
        sensor_stream: SimulatedSensorStream,
        heating_setpoint: float = 20.0,
        cooling_setpoint: float = 24.0,
        hours: int = 24,
    ) -> Dict[str, Any]:
        results = self.predict_24h_cycle(sensor_stream, days=hours // 24)

        total_heating = sum(p.heating_load for _, p, _, _ in results)
        total_cooling = sum(p.cooling_load for _, p, _, _ in results)
        avg_inference_time = np.mean([p.inference_time_ms for _, p, _, _ in results])

        hourly_temps = [w.exterior_temp for _, _, w, _ in results]
        hourly_zones = [s.zone_temp for _, _, _, s in results]

        return {
            "simulation_hours": hours,
            "total_heating_wh": total_heating,
            "total_cooling_wh": total_cooling,
            "total_energy_wh": total_heating + total_cooling,
            "total_heating_kwh": total_heating / 1000,
            "total_cooling_kwh": total_cooling / 1000,
            "total_energy_kwh": (total_heating + total_cooling) / 1000,
            "avg_inference_time_ms": avg_inference_time,
            "peak_heating_w": max(p.heating_load for _, p, _, _ in results),
            "peak_cooling_w": max(p.cooling_load for _, p, _, _ in results),
            "exterior_temp_range_c": (min(hourly_temps), max(hourly_temps)),
            "zone_temp_range_c": (min(hourly_zones), max(hourly_zones)),
            "hourly_results": [
                {
                    "time": t.isoformat(),
                    "heating_load_w": p.heating_load,
                    "cooling_load_w": p.cooling_load,
                    "exterior_temp_c": w.exterior_temp,
                    "zone_temp_c": s.zone_temp,
                    "occupancy": s.occupancy,
                }
                for t, p, w, s in results
            ],
        }


def print_prediction(p: XDTPrediction, prefix: str = ""):
    print(f"{prefix}Heating load: {p.heating_load:.1f} W")
    print(f"{prefix}Cooling load: {p.cooling_load:.1f} W")
    print(f"{prefix}Total load: {p.total_load():.1f} W")
    print(f"{prefix}Inference time: {p.inference_time_ms:.2f} ms")
    print(f"{prefix}Confidence: {p.confidence:.2%}")


def print_simulation_summary(results: Dict[str, Any]):
    print()
    print("=" * 60)
    print("24-HOUR BMS CONTROL SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Simulation period: {results['simulation_hours']} hours")
    print()
    print("ENERGY CONSUMPTION")
    print(f"  Total Heating: {results['total_heating_kwh']:.2f} kWh")
    print(f"  Total Cooling: {results['total_cooling_kwh']:.2f} kWh")
    print(f"  Total Energy:  {results['total_energy_kwh']:.2f} kWh")
    print()
    print("PEAK LOADS")
    print(f"  Peak Heating: {results['peak_heating_w']:.1f} W")
    print(f"  Peak Cooling: {results['peak_cooling_w']:.1f} W")
    print()
    print("TEMPERATURE RANGES")
    print(
        f"  Exterior: {results['exterior_temp_range_c'][0]:.1f}°C to {results['exterior_temp_range_c'][1]:.1f}°C"
    )
    print(
        f"  Zone:     {results['zone_temp_range_c'][0]:.1f}°C to {results['zone_temp_range_c'][1]:.1f}°C"
    )
    print()
    print("INFERENCE PERFORMANCE")
    print(f"  Avg inference time: {results['avg_inference_time_ms']:.2f} ms")
    print(f"  Throughput: {1000 / results['avg_inference_time_ms']:.0f} inferences/sec")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="xDT Inference Example - Real-time Weather/Sensor Input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with dummy model (no ONNX file needed)
  python tools/xdt_inference_example.py --mock

  # Run with specific ONNX model
  python tools/xdt_inference_example.py --model exports/xdt/model.onnx

  # Run 24-hour simulation
  python tools/xdt_inference_example.py --simulate-24h --model exports/xdt/model.onnx

  # Show hourly predictions for a day
  python tools/xdt_inference_example.py --hourly --model exports/xdt/model.onnx
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Path to xDT ONNX model",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock inference (no ONNX model required)",
    )
    parser.add_argument(
        "--simulate-24h",
        action="store_true",
        help="Run 24-hour BMS control simulation",
    )
    parser.add_argument(
        "--hourly",
        action="store_true",
        help="Show hourly predictions for a day",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("xDT Inference Example - Real-time Weather/Sensor Input")
    print("Issue: #977")
    print("=" * 60)
    print()

    use_mock = args.mock or (args.model is None)
    runtime = XDTRuntime(model_path=args.model, use_mock=use_mock)

    if args.simulate_24h or args.hourly:
        weather_data = []
        base_temp = 20.0
        for hour in range(24):
            temp = base_temp + 8 * np.sin(np.pi * (hour - 6) / 12)
            solar = (
                max(0, 700 * np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
            )
            weather_data.append(
                WeatherData(
                    exterior_temp=temp,
                    humidity=50.0,
                    solar_irradiance=solar,
                    wind_speed=2.0,
                )
            )

        sensor_stream = SimulatedSensorStream(weather_data)

        if args.simulate_24h:
            results = runtime.run_bms_control_simulation(sensor_stream, hours=24)
            print_simulation_summary(results)

            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\nResults saved to: {output_path}")

        elif args.hourly:
            print("HOURLY PREDICTIONS")
            print("-" * 60)
            print(
                f"{'Hour':<6} {'Ext Temp':<10} {'Zone Temp':<10} {'Heat Load':<12} {'Cool Load':<12}"
            )
            print("-" * 60)

            for hour in range(24):
                weather = sensor_stream.get_current_weather()
                sensors = sensor_stream.get_current_sensors(hour)
                time_features = (hour, 180, 6)

                prediction = runtime.predict(weather, sensors, time_features)

                print(
                    f"{hour:02d}:00  {weather.exterior_temp:8.1f}°C  {sensors.zone_temp:8.1f}°C  "
                    f"{prediction.heating_load:8.1f} W  {prediction.cooling_load:8.1f} W"
                )

                sensor_stream.advance()

        return 0

    print("SINGLE PREDICTION EXAMPLE")
    print("-" * 60)

    weather = WeatherData(
        exterior_temp=25.0,
        humidity=60.0,
        solar_irradiance=500.0,
        wind_speed=3.0,
    )

    sensors = SensorData(
        zone_temp=22.0,
        heating_setpoint=20.0,
        cooling_setpoint=24.0,
        occupancy=0.5,
        internal_gains=200.0,
    )

    time_features = (14, 180, 6)

    print("Weather Input:")
    print(f"  Exterior temp: {weather.exterior_temp}°C")
    print(f"  Humidity: {weather.humidity}%")
    print(f"  Solar irradiance: {weather.solar_irradiance} W/m²")
    print(f"  Wind speed: {weather.wind_speed} m/s")
    print()
    print("Sensor Input:")
    print(f"  Zone temp: {sensors.zone_temp}°C")
    print(f"  Heating setpoint: {sensors.heating_setpoint}°C")
    print(f"  Cooling setpoint: {sensors.cooling_setpoint}°C")
    print(f"  Occupancy: {sensors.occupancy:.0%}")
    print(f"  Internal gains: {sensors.internal_gains} W")
    print()
    print(
        f"Time: {time_features[0]:02d}:00, Day {time_features[1]}, Month {time_features[2]}"
    )
    print()

    prediction = runtime.predict(weather, sensors, time_features)

    print("PREDICTION OUTPUT")
    print("-" * 60)
    print_prediction(prediction)

    print()
    print("FEATURE VECTOR")
    features = runtime.combine_features(weather, sensors, time_features)
    print(f"Input features ({len(features)} dims): {features.round(3).tolist()}")

    print()
    print("BMS CONTROL DECISION")
    if prediction.heating_load > 100:
        print("  -> Action: CALL FOR HEATING")
        print(
            f"  -> Recommended supply temp: {20 + prediction.heating_load / 100:.1f}°C"
        )
    elif prediction.cooling_load > 100:
        print("  -> Action: CALL FOR COOLING")
        print(
            f"  -> Recommended supply temp: {24 - prediction.cooling_load / 100:.1f}°C"
        )
    else:
        print("  -> Action: NO HEATING/COOLING REQUIRED")
        print("  -> Zone is within setpoint bounds")

    return 0


if __name__ == "__main__":
    sys.exit(main())
