#!/usr/bin/env python3
"""fluxion.pyi drift gate — Issue #2509.

Asserts that every ``#[pyclass]`` registered in the primary ``fluxion`` Python
module (``#[pymodule] fn fluxion`` in ``src/lib.rs``) plus the ``multi_zone``
submodule (``src/python/bindings.rs``) has a corresponding ``class`` entry in
``fluxion.pyi``. Also checks the registered free functions (``#[pyfunction]``
wired via ``add_function`` / ``wrap_pyfunction!``) and the four exception types
added via ``m.add("...", ...)``.

Exit code 0 = stub is complete, non-zero = drift detected.

Run directly:
    python3 scripts/check_pyi_drift.py

Wired into CI via the self-contained Rust integration test
``tests/pyi_stub_completeness.rs`` (which re-implements the same logic so the
gate runs in default ``cargo test`` without a Python dependency).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LIB_RS = REPO_ROOT / "src" / "lib.rs"
MOD_RS = REPO_ROOT / "src" / "python" / "mod.rs"
BINDINGS_RS = REPO_ROOT / "src" / "python" / "bindings.rs"
PYI = REPO_ROOT / "fluxion.pyi"

# All files that may contain #[pyclass(name="...")] / #[pymethods] defs.
PYCLASS_SOURCES = [
    REPO_ROOT / "src" / "lib.rs",
    REPO_ROOT / "src" / "api" / "parameters.rs",
    REPO_ROOT / "src" / "python" / "bindings.rs",
    REPO_ROOT / "src" / "python" / "hvac_bindings.rs",
    REPO_ROOT / "src" / "python" / "model_bindings.rs",
    REPO_ROOT / "src" / "python" / "multi_node_bindings.rs",
    REPO_ROOT / "src" / "python" / "osm_bindings.rs",
    # Issue #3402: files extracted by #2493 (BatchOracle pymethods /
    # construction pyclasses) were missing from the scan list.
    REPO_ROOT / "src" / "python" / "batch_oracle_bindings.rs",
    REPO_ROOT / "src" / "python" / "construction_bindings.rs",
]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"error: required file not found: {path}", file=sys.stderr)
        sys.exit(2)


def _balanced_block(src: str, start: int) -> str | None:
    """Return the text inside the first ``{...}`` block at/after ``start``."""
    i = src.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i + 1 : j]
    return None


def _pymodule_body(src: str, fn_name: str) -> str:
    """Return the body text of ``#[pymodule] fn <fn_name>``."""
    for m in re.finditer(
        r"#\[pymodule\][^\n]*\n\s*(?:pub\s+)?fn\s+" + re.escape(fn_name) + r"\b", src
    ):
        body = _balanced_block(src, m.end())
        if body is not None:
            return body
    return ""


def _build_pyclass_name_map() -> dict[str, str]:
    """Map rust struct/enum name -> python class name from #[pyclass(name=...)] attrs.

    Falls back to stripping a leading ``Py`` prefix when no explicit ``name`` is set.
    """
    mapping: dict[str, str] = {}
    for path in PYCLASS_SOURCES:
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        for m in re.finditer(r"#\[pyclass([^\]]*)\]", src):
            opts = m.group(1)
            explicit = re.search(r'name\s*=\s*"([^"]+)"', opts)
            tail = src[m.end() : m.end() + 600]
            sm = re.search(r"(?:pub\s+)?(?:struct|enum)\s+(\w+)", tail)
            if not sm:
                continue
            rust = sm.group(1)
            if explicit:
                mapping[rust] = explicit.group(1)
            else:
                mapping.setdefault(rust, rust[2:] if rust.startswith("Py") else rust)
    return mapping


def registered_classes() -> list[str]:
    """Python class names registered in the primary ``fluxion`` module."""
    lib = _read(LIB_RS)
    name_map = _build_pyclass_name_map()
    body = _pymodule_body(lib, "fluxion")
    if not body:
        print("error: could not locate #[pymodule] fn fluxion in src/lib.rs", file=sys.stderr)
        sys.exit(2)
    out: list[str] = []
    for body_src in (body, _pymodule_body(_read(BINDINGS_RS), "multi_zone")):
        for m in re.finditer(r"add_class::<([^>]+)>", body_src):
            rust_full = m.group(1).strip()
            rust = rust_full.split("::")[-1]
            out.append(name_map.get(rust, rust[2:] if rust.startswith("Py") else rust))
    return out


def registered_functions() -> list[str]:
    """Python function names registered at the top level of the fluxion module.

    Includes functions added directly (hvac/osm) and via the multi_zone submodule
    call (create_multi_zone_model_* / export_gbxml).
    """
    lib = _read(LIB_RS)
    body = _pymodule_body(lib, "fluxion")
    funcs = set()
    for m in re.finditer(r"wrap_pyfunction!\(\s*([\w:]+)", body):
        funcs.add(m.group(1).split("::")[-1])
    # multi_zone submodule adds these (src/python/bindings.rs #[pymodule] fn multi_zone).
    mz_body = _pymodule_body(_read(BINDINGS_RS), "multi_zone")
    for m in re.finditer(r"wrap_pyfunction!\(\s*([\w:]+)", mz_body):
        funcs.add(m.group(1).split("::")[-1])
    return sorted(funcs)


def registered_exceptions() -> list[str]:
    lib = _read(LIB_RS)
    body = _pymodule_body(lib, "fluxion")
    return re.findall(r'm\.add\("(\w+)"', body)


def pyi_classes() -> set[str]:
    src = _read(PYI)
    return set(re.findall(r"(?m)^class\s+(\w+)\b", src))


def pyi_functions() -> set[str]:
    src = _read(PYI)
    return set(re.findall(r"(?m)^def\s+(\w+)\b", src))


def main() -> int:
    classes = registered_classes()
    funcs = registered_functions()
    excs = registered_exceptions()

    pyi_cls = pyi_classes()
    pyi_fns = pyi_functions()

    failures: list[str] = []

    missing_classes = sorted(set(classes) - pyi_cls)
    if missing_classes:
        failures.append(
            "Missing class stubs in fluxion.pyi (registered in #[pymodule] fn fluxion "
            "but no `class X` declaration):\n  "
            + "\n  ".join(missing_classes)
        )

    missing_funcs = sorted(set(funcs) - pyi_fns)
    if missing_funcs:
        failures.append(
            "Missing function stubs in fluxion.pyi:\n  " + "\n  ".join(missing_funcs)
        )

    missing_exc = sorted(set(excs) - pyi_cls)
    if missing_exc:
        failures.append(
            "Missing exception stubs in fluxion.pyi:\n  " + "\n  ".join(missing_exc)
        )

    print(f"fluxion.pyi drift gate: {len(classes)} classes, {len(funcs)} functions, "
          f"{len(excs)} exceptions registered.")
    print(f"fluxion.pyi declares {len(pyi_cls)} classes and {len(pyi_fns)} top-level functions.")

    if failures:
        print("\nDRIFT DETECTED — fluxion.pyi is out of sync with the PyO3 registrations:\n")
        for f in failures:
            print(f + "\n")
        print(
            "Fix: add the missing declarations to fluxion.pyi, or update the "
            "registrations in src/lib.rs (#[pymodule] fn fluxion) / "
            "src/python/bindings.rs (#[pymodule] fn multi_zone)."
        )
        return 1

    print("OK — fluxion.pyi is in sync with PyO3 registrations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
