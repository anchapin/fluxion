#!/usr/bin/env python3
"""Display gate status from gate_status.json file."""

import json
import sys


def main():
    try:
        with open("gate_status.json") as f:
            data = json.load(f)

        overall = "PASSED" if data["overall_passed"] else "FAILED"
        summary = data["summary"]
        print(f"Overall: {overall}")
        print(f"Gates: {summary['passed']}/{summary['total']} passed")
        print()

        for gate in data["gates"]:
            if not gate["passed"]:
                print(f"  ❌ {gate['category']}/{gate['name']}: {gate['message']}")

    except FileNotFoundError:
        print("gate_status.json not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing gate_status.json: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
