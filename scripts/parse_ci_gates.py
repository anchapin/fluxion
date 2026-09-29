#!/usr/bin/env python3
"""Parse .github/ci-gates.yaml into a GitHub Actions matrix.

Issue #4254: Gate Registry Pattern.

This script reads the gate registry config and outputs a JSON matrix
for the gate-runner.yml workflow. Each enabled gate becomes a matrix entry.

Usage:
    python3 scripts/parse_ci_gates.py --config .github/ci-gates.yaml --output $GITHUB_OUTPUT

The output is written as `matrix=<json>` to the GITHUB_OUTPUT file.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML not installed, trying to parse manually...", file=sys.stderr)
    yaml = None


def parse_gates(config_path: Path) -> list[dict]:
    """Parse the gate registry YAML and return list of enabled gates."""
    with open(config_path) as f:
        if yaml:
            data = yaml.safe_load(f)
        else:
            # Fallback: minimal YAML parser for this specific format
            # (only used if PyYAML not available)
            raise RuntimeError("PyYAML required to parse ci-gates.yaml")

    gates = []
    for gate in data.get("gates", []):
        # Skip disabled gates
        if not gate.get("enabled", True):
            continue

        # Build matrix entry with defaults
        gates.append({
            "name": gate["name"],
            "description": gate.get("description", ""),
            "script": gate["script"],
            "args": gate.get("args", []),
            "timeout_minutes": gate.get("timeout_minutes", 10),
            "required": gate.get("required", True),
        })

    return gates


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse CI gate registry")
    parser.add_argument("--config", required=True, help="Path to ci-gates.yaml")
    parser.add_argument("--output", required=True, help="Path to GITHUB_OUTPUT file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    gates = parse_gates(config_path)
    matrix_json = json.dumps(gates)

    # Write to GITHUB_OUTPUT
    with open(args.output, "a") as f:
        f.write(f"matrix={matrix_json}\n")

    print(f"Parsed {len(gates)} enabled gates from {config_path}")
    for gate in gates:
        print(f"  - {gate['name']}: {gate['script']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
