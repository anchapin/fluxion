#!/usr/bin/env python3
"""Fetch, install and verify the ASHRAE 140 accompanying files.

ASHRAE sells Standard 140 and distributes its accompanying files alongside it.
fluxion is a public repository, so it does not redistribute those bytes. Instead
it records a provenance chain and verifies a copy the user supplies themselves.

The conformance claim is about READING the normative files. It does not require
hosting them, so nothing is lost by this arrangement.

Source of truth for the download:
  https://data.ashrae.org/standard140/
  https://data.ashrae.org/standard140/accompany.html
  Download link -> ASHRAE bookstore, "140-2023 Supplemental Files"

Subcommands
-----------
record   Point at an extracted archive. Computes SHA-256 for every file named in
         the manifest and writes them into the provenance record as publisher
         hashes. Run once, by someone holding a licensed copy.

install  Copy the normative files out of an extracted archive into the working
         tree, verifying each against the recorded publisher hash first.

verify   Check the working tree against the provenance record. Exit non-zero on
         any missing file, hash mismatch, or file whose publisher hash has never
         been recorded. This is the CI gate.

status   Human-readable summary. Always exits 0.

Usage
-----
  python3 scripts/fetch_ashrae140.py record  --archive ~/Std140_2023_Supplemental
  python3 scripts/fetch_ashrae140.py install --archive ~/Std140_2023_Supplemental
  python3 scripts/fetch_ashrae140.py verify
  python3 scripts/fetch_ashrae140.py status
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROVENANCE = os.path.join(REPO, "data", "reference", "ashrae140", "provenance.json")

UNRECORDED = "publisher_hash_unrecorded"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load() -> dict:
    if not os.path.exists(PROVENANCE):
        sys.exit(f"provenance record not found: {PROVENANCE}")
    with open(PROVENANCE) as fh:
        return json.load(fh)


def save(doc: dict) -> None:
    with open(PROVENANCE, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")


def find_in_archive(archive: str, archive_path: str) -> str | None:
    """Locate a manifest entry inside an extracted archive.

    Tries the recorded path first, then falls back to a case-insensitive
    basename search, because the archive ships Windows-style paths and casing
    varies between extraction tools.
    """
    direct = os.path.join(archive, archive_path.replace("\\", os.sep))
    if os.path.exists(direct):
        return direct
    # os.path.basename does not split on backslashes on POSIX, and the archive
    # ships Windows-style paths, so normalise before taking the basename.
    want = os.path.basename(archive_path.replace("\\", os.sep)).lower()
    for root, _dirs, names in os.walk(archive):
        for name in names:
            if name.lower() == want:
                return os.path.join(root, name)
    return None


def iter_entries(doc: dict):
    for group in doc["file_groups"]:
        for entry in group["files"]:
            yield group, entry


def cmd_record(args: argparse.Namespace) -> int:
    doc = load()
    found = missing = 0
    for group, entry in iter_entries(doc):
        src = find_in_archive(args.archive, entry["archive_path"])
        if src is None:
            print(f"  MISSING  {entry['archive_path']}")
            missing += 1
            continue
        digest = sha256(src)
        prior = entry.get("publisher_sha256", UNRECORDED)
        if prior not in (UNRECORDED, digest):
            print(f"  CONFLICT {entry['name']}: recorded {prior[:16]}... "
                  f"but archive has {digest[:16]}...")
            if not args.force:
                print("           refusing to overwrite; re-run with --force")
                missing += 1
                continue
        entry["publisher_sha256"] = digest
        entry["publisher_bytes"] = os.path.getsize(src)
        found += 1
    doc["chain"]["publisher_hashes_recorded_utc"] = (
        _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
    )
    if args.archive_sha256:
        doc["chain"]["archive_sha256"] = args.archive_sha256
    if args.retrieved:
        doc["chain"]["retrieved_utc"] = args.retrieved
    save(doc)
    print(f"\nrecorded {found} file(s), {missing} unresolved -> {PROVENANCE}")
    return 1 if missing else 0


def cmd_install(args: argparse.Namespace) -> int:
    doc = load()
    installed = failed = 0
    for group, entry in iter_entries(doc):
        dest = os.path.join(REPO, group["install_dir"], entry["name"])
        expected = entry.get("publisher_sha256", UNRECORDED)
        if expected == UNRECORDED:
            print(f"  SKIP     {entry['name']}: no publisher hash recorded, run 'record' first")
            failed += 1
            continue
        src = find_in_archive(args.archive, entry["archive_path"])
        if src is None:
            print(f"  MISSING  {entry['archive_path']}")
            failed += 1
            continue
        digest = sha256(src)
        if digest != expected:
            print(f"  MISMATCH {entry['name']}: archive copy does not match the recorded hash")
            failed += 1
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)
        print(f"  ok       {entry['name']}")
        installed += 1
    print(f"\ninstalled {installed} file(s), {failed} failed")
    return 1 if failed else 0


def cmd_verify(args: argparse.Namespace) -> int:
    doc = load()
    ok = mismatch = absent = unrecorded = 0
    for group, entry in iter_entries(doc):
        path = os.path.join(REPO, group["install_dir"], entry["name"])
        expected = entry.get("publisher_sha256", UNRECORDED)
        if not os.path.exists(path):
            print(f"  ABSENT     {entry['name']}")
            absent += 1
            continue
        digest = sha256(path)
        if expected == UNRECORDED:
            note = entry.get("working_copy_sha256")
            tag = "matches working copy on file" if note == digest else "DIFFERS from working copy on file"
            print(f"  UNVERIFIED {entry['name']}: no publisher hash recorded ({tag})")
            unrecorded += 1
            continue
        if digest == expected:
            ok += 1
        else:
            print(f"  MISMATCH   {entry['name']}")
            print(f"             expected {expected}")
            print(f"             actual   {digest}")
            mismatch += 1
    total = ok + mismatch + absent + unrecorded
    print(f"\n{ok}/{total} verified against publisher hashes; "
          f"{mismatch} mismatched, {absent} absent, {unrecorded} unverified")
    if mismatch or absent:
        print("\nFAIL: the working tree does not match the recorded provenance chain.")
        return 1
    if unrecorded:
        print("\nFAIL: publisher hashes have never been recorded for the files above.")
        print("      No Section 6 conformance claim may be published until they are.")
        print("      Obtain the supplemental files from ASHRAE and run:")
        print("        python3 scripts/fetch_ashrae140.py record --archive <extracted-dir>")
        return 1
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    doc = load()
    chain = doc["chain"]
    print(f"source      {chain['publisher']}")
    print(f"page        {chain['page_url']}")
    print(f"edition     {chain['edition']}")
    print(f"archive     {chain.get('archive_sha256') or '(not recorded)'}")
    print(f"retrieved   {chain.get('retrieved_utc') or '(not recorded)'}")
    print(f"hashes rec. {chain.get('publisher_hashes_recorded_utc') or '(never)'}")
    print()
    for group in doc["file_groups"]:
        n = len(group["files"])
        rec = sum(1 for f in group["files"]
                  if f.get("publisher_sha256", UNRECORDED) != UNRECORDED)
        print(f"  {group['id']:<12} {rec}/{n} publisher hashes recorded "
              f"-> {group['install_dir']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("record", help="record publisher hashes from an extracted archive")
    p.add_argument("--archive", required=True, help="path to the extracted supplemental files")
    p.add_argument("--archive-sha256", help="SHA-256 of the downloaded archive itself")
    p.add_argument("--retrieved", help="ISO-8601 UTC timestamp of the download")
    p.add_argument("--force", action="store_true", help="overwrite conflicting recorded hashes")
    p.set_defaults(fn=cmd_record)

    p = sub.add_parser("install", help="copy verified files into the working tree")
    p.add_argument("--archive", required=True, help="path to the extracted supplemental files")
    p.set_defaults(fn=cmd_install)

    p = sub.add_parser("verify", help="check the working tree against the provenance chain")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("status", help="summarise the provenance chain")
    p.set_defaults(fn=cmd_status)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
