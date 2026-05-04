#!/usr/bin/env python3
"""Remove empty files and directories (safe paths only)."""

import os
import shutil

os.chdir("/home/alex/Projects/fluxion")

safe_paths = ["./target/", "./.sdd/", "./.automaker/"]


def is_safe(path):
    for sp in safe_paths:
        if path.startswith(sp) or f"/{sp.lstrip('./')}" in path:
            return False
    return True


# Remove empty files
removed_files = 0
for dirpath, dirnames, filenames in os.walk("."):
    if not is_safe(dirpath):
        continue
    for f in filenames:
        fp = os.path.join(dirpath, f)
        if f == ".gitkeep":
            continue
        try:
            if os.path.isfile(fp) and os.path.getsize(fp) == 0:
                os.remove(fp)
                removed_files += 1
                print(f"Removed file: {fp}")
        except Exception as e:
            print(f"Error removing {fp}: {e}")

# Remove empty directories
removed_dirs = 0
for dirpath, dirnames, filenames in os.walk(".", topdown=False):
    if not is_safe(dirpath):
        continue
    try:
        if os.path.isdir(dirpath) and not os.listdir(dirpath):
            os.rmdir(dirpath)
            removed_dirs += 1
            print(f"Removed dir: {dirpath}")
    except Exception as e:
        print(f"Error removing dir {dirpath}: {e}")

# Remove .ruff_cache directories
removed_ruff = 0
for dirpath, dirnames, filenames in os.walk("."):
    if ".ruff_cache" in dirnames:
        dp = os.path.join(dirpath, ".ruff_cache")
        if is_safe(dp):
            try:
                shutil.rmtree(dp)
                removed_ruff += 1
                print(f"Removed .ruff_cache: {dp}")
            except Exception as e:
                print(f"Error removing .ruff_cache {dp}: {e}")

print(
    f"\nSummary: {removed_files} files, {removed_dirs} dirs, {removed_ruff} .ruff_cache removed"
)
