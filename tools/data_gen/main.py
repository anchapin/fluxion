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

from tools.data_gen import ashrae_140_generator, geometry, simulation, weather

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

    # Subcommand: generate-ashrae
    ashrae_parser = subparsers.add_parser(
        "generate-ashrae", help="Generate ASHRAE 140 test cases with variations"
    )
    ashrae_parser.add_argument(
        "--out", default="assets/ashrae_140_cases.json", help="Output JSON file"
    )
    ashrae_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    ashrae_parser.add_argument(
        "--u-value-variation",
        type=float,
        default=0.5,
        help="U-value variation fraction (default: 0.5 = ±50%%)",
    )
    ashrae_parser.add_argument(
        "--setpoint-variation",
        type=float,
        default=5.0,
        help="Setpoint variation in °C (default: 5.0)",
    )
    ashrae_parser.add_argument(
        "--variations-per-case",
        type=int,
        default=0,
        help="Number of variations per case (0 = no variations, just base cases)",
    )
    ashrae_parser.add_argument(
        "--base-only",
        action="store_true",
        help="Generate only base cases without variations (same as --variations-per-case 0)",
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
    gen_parser.add_argument("--u-value-min", type=float, default=0.5)
    gen_parser.add_argument("--u-value-max", type=float, default=3.0)
    gen_parser.add_argument("--setpoint-min", type=float, default=19.0)
    gen_parser.add_argument("--setpoint-max", type=float, default=24.0)

    args = parser.parse_args()

    if args.command == "generate-ashrae":
        generator = ashrae_140_generator.ASHRAE140CaseGenerator(seed=args.seed)

        if args.base_only:
            cases = generator.generate_all_cases()
            logger.info(f"Generated {len(cases)} base ASHRAE 140 cases")
        elif args.variations_per_case > 0:
            cases = generator.generate_all_variations(
                u_value_variation=args.u_value_variation,
                setpoint_variation=args.setpoint_variation,
                num_variations_per_case=args.variations_per_case,
            )
            logger.info(f"Generated {len(cases)} ASHRAE 140 cases with variations")
        else:
            # Default: generate base cases
            cases = generator.generate_all_cases()
            logger.info(f"Generated {len(cases)} base ASHRAE 140 cases")

        ashrae_140_generator.save_cases_to_json(cases, args.out)
        logger.info(f"ASHRAE 140 cases saved to {args.out}")

    elif args.command == "download-weather":
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
            u_value = random.uniform(args.u_value_min, args.u_value_max)
            setpoint = random.uniform(args.setpoint_min, args.setpoint_max)

            params = {
                "width": w,
                "length": length_val,
                "height": args.height,
                "wwr": wwr,
                "orientation": orient,
                "u_value": u_value,
                "hvac_setpoint": setpoint,
                "weather_file": os.path.basename(epw_path),
            }

            logger.info(
                f"  Params: {w=:.2f}, {length_val=:.2f}, {wwr=:.2f}, {orient=:.1f}, "
                f"{u_value=:.2f}, {setpoint=:.1f}"
            )

            # Create Model
            model = geometry.create_shoebox_model(
                width=w,
                length=length_val,
                height=args.height,
                wwr=wwr,
                orientation_degrees=orient,
                window_u_value=u_value,
                hvac_setpoint=setpoint,
            )

            # Run Simulation
            success = simulation.run_simulation(
                model=model,
                weather_file_path=epw_path,
                output_dir=args.out,
                run_name=run_id,
                params=params,
            )

            if success:
                logger.info(f"  Completed {run_id}")
            else:
                logger.warning(f"  Failed {run_id}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
