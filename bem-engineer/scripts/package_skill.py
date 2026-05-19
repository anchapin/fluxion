#!/usr/bin/env python3
"""Package a skill directory into a .skill zip file."""

import sys
import zipfile
from pathlib import Path


def package_skill(skill_dir: str):
    skill_path = Path(skill_dir).resolve()
    if not skill_path.exists():
        print(f"Error: {skill_dir} does not exist")
        sys.exit(1)

    skill_file = skill_path.with_suffix(".skill")

    with zipfile.ZipFile(skill_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(skill_path.rglob("*")):
            if file_path.is_file():
                # Skip hidden files, __pycache__, node_modules
                rel = file_path.relative_to(skill_path)
                parts = rel.parts
                if any(
                    p.startswith(".") or p == "__pycache__" or p == "node_modules"
                    for p in parts
                ):
                    continue
                arcname = str(rel)
                zf.write(file_path, arcname)
                print(f"  {arcname}")

    size_kb = skill_file.stat().st_size / 1024
    print(f"\nPackaged: {skill_file} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 package_skill.py <path/to/skill-folder>")
        sys.exit(1)
    package_skill(sys.argv[1])
