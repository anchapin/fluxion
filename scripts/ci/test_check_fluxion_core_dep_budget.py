"""
Tests for ``scripts/check_fluxion_core_dep_budget.py`` (Issue #3467).

The dependency-budget gate enforces the documented "dependency-light leaf"
contract for ``fluxion-core``. These tests pin the static manifest scan,
the dynamic ``cargo tree`` parsing shape, and the overall ``main()``
exit-code contract so a silent regex/allow-list regression cannot
reintroduce a heavyweight dep into the default build.

Pattern (mirrors ``test_check_ashrae_cases_cycle.py`` /
``test_check_physics_sim_cycle.py``): load the script as a fresh
module, redirect its module-level path constants at a ``tmp_path``
fixture tree, drive each scanner + ``main()`` through clean and
offender scenarios.

We stub out the ``cargo_tree_packages`` shim so the tests do not need a
working Rust toolchain or a populated lockfile — the dynamic-tree
contract is checked by the existing CI matrix (``cargo test -p
fluxion-core`` + ``cargo tree -p fluxion-core``) on the real tree; the
unit-level correctness of the gating logic is what we pin here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT_NAME = "check_fluxion_core_dep_budget"


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the dep-budget gate script."""
    return load_script(SCRIPT_NAME)


def _redirect(checker, tmp_path, monkeypatch) -> None:
    """Point every module-level path constant at the ``tmp_path`` mock repo."""
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        checker, "FLUXION_CORE_MANIFEST", tmp_path / "fluxion-core" / "Cargo.toml"
    )


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _stub_cargo_tree(
    monkeypatch, checker, *, default: set[str], featured: set[str]
) -> None:
    """Replace ``cargo_tree_packages`` so the dynamic passes don't actually
    shell out. The unit-level contract we pin here is what the gate
    *does* with the parsed crate names; the real ``cargo tree`` parsing
    shape is exercised by the Scripts Test Suite against the real repo
    (see ``scripts-tests.yml``).
    """

    def fake(manifest_path, *, features=None):
        assert manifest_path == checker.FLUXION_CORE_MANIFEST
        if features is None:
            return set(default)
        return set(featured)

    monkeypatch.setattr(checker, "cargo_tree_packages", fake)


# ---------------------------------------------------------------------------
# Static-manifest scanner
# ---------------------------------------------------------------------------


def test_static_clean_manifest_passes(checker, tmp_path, monkeypatch):
    """A clean fluxion-core Cargo.toml produces zero static findings."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(
        checker.FLUXION_CORE_MANIFEST,
        """
[package]
name = "fluxion-core"
version = "1.0.0"
edition = "2021"

[features]
default = []
tmy3-download = ["dep:reqwest", "dep:directories", "dep:sha2"]

[dependencies]
num-traits = "0.2"
serde.workspace = true
serde_json = "1.0"
serde_yaml = "0.9"
thiserror.workspace = true
log = "0.4"
reqwest = { workspace = true, optional = true }
sha2 = { version = "0.11", optional = true }
directories = { version = "6.0", optional = true }

[dev-dependencies]
proptest = "1.5"
mockito = "1.7"
""",
    )
    sections = checker.parse_manifest(checker.FLUXION_CORE_MANIFEST)
    failures, warnings = checker.static_manifest_check(sections)
    assert failures == []
    # mockito is on the heavyweight allow-list; the static check reports
    # it as a dev-dep advisory warning.
    assert any("mockito" in w for w in warnings)


def test_static_flags_non_optional_reqwest(checker, tmp_path, monkeypatch):
    """A regression that adds reqwest as a non-optional dep is caught."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(
        checker.FLUXION_CORE_MANIFEST,
        """
[package]
name = "fluxion-core"
version = "1.0.0"
edition = "2021"

[dependencies]
num-traits = "0.2"
serde = "1.0"
reqwest = "0.12"
""",
    )
    sections = checker.parse_manifest(checker.FLUXION_CORE_MANIFEST)
    failures, _warnings = checker.static_manifest_check(sections)
    assert any("reqwest" in f for f in failures)


def test_static_allows_optional_reqwest(checker, tmp_path, monkeypatch):
    """reqwest with optional = true is allowed (feature-gated)."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(
        checker.FLUXION_CORE_MANIFEST,
        """
[package]
name = "fluxion-core"
version = "1.0.0"
edition = "2021"

[features]
tmy3-download = ["dep:reqwest"]

[dependencies]
num-traits = "0.2"
serde = "1.0"
reqwest = { version = "0.12", optional = true }
""",
    )
    sections = checker.parse_manifest(checker.FLUXION_CORE_MANIFEST)
    failures, _warnings = checker.static_manifest_check(sections)
    assert failures == []


def test_static_flags_hyper(checker, tmp_path, monkeypatch):
    """A regression that adds hyper as a non-optional dep is caught."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(
        checker.FLUXION_CORE_MANIFEST,
        """
[package]
name = "fluxion-core"
version = "1.0.0"
edition = "2021"

[dependencies]
num-traits = "0.2"
serde = "1.0"
hyper = "1.0"
""",
    )
    sections = checker.parse_manifest(checker.FLUXION_CORE_MANIFEST)
    failures, _warnings = checker.static_manifest_check(sections)
    assert any("hyper" in f for f in failures)


def test_static_flags_directories(checker, tmp_path, monkeypatch):
    """A regression that adds directories as a non-optional dep is caught."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(
        checker.FLUXION_CORE_MANIFEST,
        """
[package]
name = "fluxion-core"
version = "1.0.0"
edition = "2021"

[dependencies]
num-traits = "0.2"
serde = "1.0"
directories = "6.0"
""",
    )
    sections = checker.parse_manifest(checker.FLUXION_CORE_MANIFEST)
    failures, _warnings = checker.static_manifest_check(sections)
    assert any("directories" in f for f in failures)


# ---------------------------------------------------------------------------
# Dynamic-tree gate
# ---------------------------------------------------------------------------


def test_dynamic_tree_clean_passes(checker, tmp_path, monkeypatch):
    """Default-feature tree without reqwest / hyper / tokio passes."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(checker.FLUXION_CORE_MANIFEST, "[package]\nname = \"x\"\n")
    _stub_cargo_tree(
        monkeypatch,
        checker,
        default={"serde", "serde_json", "log", "num-traits", "thiserror"},
        featured={"serde", "reqwest", "directories", "sha2"},
    )
    failures, featured_failures = checker.dynamic_tree_check(
        checker.FLUXION_CORE_MANIFEST
    )
    assert failures == []
    assert featured_failures == []


def test_dynamic_tree_flags_default_reqwest(checker, tmp_path, monkeypatch):
    """If reqwest leaks into the default-feature tree, the gate fails."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(checker.FLUXION_CORE_MANIFEST, "[package]\nname = \"x\"\n")
    _stub_cargo_tree(
        monkeypatch,
        checker,
        default={"serde", "reqwest"},  # reqwest should NOT be in default
        featured={"reqwest", "directories", "sha2"},
    )
    failures, _featured = checker.dynamic_tree_check(checker.FLUXION_CORE_MANIFEST)
    assert any("reqwest" in f for f in failures)


def test_dynamic_tree_flags_transitive_hyper(checker, tmp_path, monkeypatch):
    """Transitive heavyweight crates (hyper, tokio) are flagged too."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(checker.FLUXION_CORE_MANIFEST, "[package]\nname = \"x\"\n")
    _stub_cargo_tree(
        monkeypatch,
        checker,
        default={"serde", "hyper", "tokio"},
        featured={"hyper", "tokio"},
    )
    failures, _featured = checker.dynamic_tree_check(checker.FLUXION_CORE_MANIFEST)
    assert any("hyper" in f for f in failures)
    assert any("tokio" in f for f in failures)


def test_dynamic_tree_flags_missing_featured(checker, tmp_path, monkeypatch):
    """If the tmy3-download feature tree is missing reqwest, gate fails.

    This is the positive-direction check: a regression that breaks the
    feature gating (e.g. accidentally removes `dep:reqwest` from the
    feature dep list) is caught here.
    """
    _redirect(checker, tmp_path, monkeypatch)
    _write(checker.FLUXION_CORE_MANIFEST, "[package]\nname = \"x\"\n")
    _stub_cargo_tree(
        monkeypatch,
        checker,
        default={"serde"},
        featured={"directories", "sha2"},  # reqwest missing!
    )
    _default_failures, featured_failures = checker.dynamic_tree_check(
        checker.FLUXION_CORE_MANIFEST
    )
    assert any("reqwest" in f for f in featured_failures)


# ---------------------------------------------------------------------------
# End-to-end main()
# ---------------------------------------------------------------------------


def test_main_clean_returns_0(checker, tmp_path, monkeypatch, capsys):
    """main() exits 0 with a clean manifest + tree."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(
        checker.FLUXION_CORE_MANIFEST,
        """
[package]
name = "fluxion-core"
version = "1.0.0"
edition = "2021"

[features]
tmy3-download = ["dep:reqwest", "dep:directories", "dep:sha2"]

[dependencies]
num-traits = "0.2"
serde = "1.0"
serde_json = "1.0"
thiserror = "2.0"
log = "0.4"
reqwest = { version = "0.12", optional = true }
sha2 = { version = "0.11", optional = true }
directories = { version = "6.0", optional = true }

[dev-dependencies]
mockito = "1.7"
""",
    )
    _stub_cargo_tree(
        monkeypatch,
        checker,
        default={"serde", "serde_json", "log", "num-traits", "thiserror"},
        featured={"serde", "reqwest", "directories", "sha2"},
    )
    rc = checker.main()
    assert rc == 0
    captured = capsys.readouterr()
    assert "dependency budget is intact" in captured.out


def test_main_offending_returns_1(checker, tmp_path, monkeypatch, capsys):
    """main() exits 1 when the manifest is dirty."""
    _redirect(checker, tmp_path, monkeypatch)
    _write(
        checker.FLUXION_CORE_MANIFEST,
        """
[package]
name = "fluxion-core"
version = "1.0.0"
edition = "2021"

[dependencies]
num-traits = "0.2"
reqwest = "0.12"
""",
    )
    _stub_cargo_tree(
        monkeypatch,
        checker,
        default={"reqwest"},
        featured={"reqwest"},
    )
    rc = checker.main()
    assert rc == 1
    captured = capsys.readouterr()
    assert "DEPENDENCY-BUDGET REGRESSION DETECTED" in captured.out


def test_main_returns_2_when_manifest_missing(checker, tmp_path, monkeypatch, capsys):
    """main() exits 2 when the manifest file is missing (script error)."""
    _redirect(checker, tmp_path, monkeypatch)
    # Deliberately do NOT write FLUXION_CORE_MANIFEST.
    rc = checker.main()
    assert rc == 2
