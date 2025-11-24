#!/usr/bin/env python3
"""
CLI Entry point for the Data Generation Tool.
"""

import argparse
import logging
import os
import random
import sys

# Ensure we can import the package if run directly
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from tools.data_gen import geometry, simulation, weather

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fluxion.data_gen")


def main():
    parser = argparse.ArgumentParser(description="Fluxion Synthetic Data Generator")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: download-weather
    weather_parser = subparsers.add_parser(
        "download-weather", help="Download standard weather files"
    )
    weather_parser.add_argument(
        "--out", default="assets/weather", help="Output directory for weather files"
    )

    # Subcommand: generate
    gen_parser = subparsers.add_parser(
        "generate", help="Generate synthetic data (Geometry + Simulation)"
    )
    gen_parser.add_argument(
        "--out", default="assets/synthetic_data", help="Output directory"
    )
    gen_parser.add_argument(
        "--weather-dir",
        default="assets/weather",
        help="Directory containing weather files",
    )
    gen_parser.add_argument(
        "--weather-file", default=None, help="Specific weather file name (optional)"
    )
    gen_parser.add_argument(
        "--count", type=int, default=1, help="Number of simulations to run"
    )

    # Parametric ranges (or specific values)
    gen_parser.add_argument("--width-min", type=float, default=5.0)
    gen_parser.add_argument("--width-max", type=float, default=20.0)
    gen_parser.add_argument("--length-min", type=float, default=5.0)
    gen_parser.add_argument("--length-max", type=float, default=20.0)
    gen_parser.add_argument("--height", type=float, default=3.5)
    gen_parser.add_argument("--wwr-min", type=float, default=0.1)
    gen_parser.add_argument("--wwr-max", type=float, default=0.8)

    args = parser.parse_args()

    if args.command == "download-weather":
        weather.download_standard_files(args.out)

    elif args.command == "generate":
        # Ensure weather exists
        if not os.path.exists(args.weather_dir):
            logger.info(
                f"Weather directory {args.weather_dir} not found. "
                "Downloading defaults..."
            )
            weather.download_standard_files(args.weather_dir)

        weather_files = [f for f in os.listdir(args.weather_dir) if f.endswith(".epw")]
        if not weather_files:
            logger.error("No EPW files found!")
            sys.exit(1)

        # Select weather file
        if args.weather_file:
            epw_path = weather.get_weather_file_path(
                args.weather_dir, args.weather_file
            )
        else:
            # Pick one random or first? Let's use first for consistency unless requested
            epw_path = os.path.join(args.weather_dir, weather_files[0])

        logger.info(f"Using weather file: {epw_path}")

        # Generation Loop
        for i in range(args.count):
            run_id = f"sim_{i:04d}"
            logger.info(f"Starting simulation {i + 1}/{args.count}: {run_id}")

            # Sample parameters
            w = random.uniform(args.width_min, args.width_max)
            length_val = random.uniform(args.length_min, args.length_max)
            wwr = random.uniform(args.wwr_min, args.wwr_max)
            orient = random.uniform(0, 360)

            logger.info(
                f"  Params: {w=:.2f}, {length_val=:.2f}, {wwr=:.2f}, {orient=:.1f}"
            )

            # Create Model
            model = geometry.create_shoebox_model(
                width=w,
                length=length_val,
                height=args.height,
                wwr=wwr,
                orientation_degrees=orient,
            )

            # Run Simulation
            success = simulation.run_simulation(
                model=model,
                weather_file_path=epw_path,
                output_dir=args.out,
                run_name=run_id,
            )

            if success:
                logger.info(f"  Completed {run_id}")
            else:
                logger.warning(f"  Failed {run_id}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
