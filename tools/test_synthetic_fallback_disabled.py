#!/usr/bin/env python3
"""
Regression test for Issue #1338.

Asserts that ``tools/train_surrogate.py`` refuses to silently substitute
synthetic data when ``data/training/`` is empty or missing. Production
surrogate retraining MUST go through the physics-extraction pipeline;
the synthetic fallback is now gated behind
``--allow-synthetic-for-benchmark-only``.

Run::

    pytest tools/test_synthetic_fallback_disabled.py -v

Exit codes:
  * 0 — all assertions pass (Issue #1338 still satisfied).
  * non-zero — at least one assertion failed (Issue #1338 regression).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "tools" / "train_surrogate.py"


def _run_train(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke ``tools/train_surrogate.py`` in a subprocess.

    Using a subprocess (rather than importing the module) mirrors the way
    the script is invoked in CI and ensures that ``SystemExit`` actually
    produces a non-zero exit code (the acceptance criterion).
    """
    cmd = [sys.executable, str(TRAIN_SCRIPT), *args]
    return subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


class TestSyntheticFallbackDisabled(unittest.TestCase):
    """Issue #1338 — synthetic fallback must be hard-gated."""

    # ---------------------------------------------------------------
    # Acceptance criterion #1: empty data dir → non-zero exit, no leak
    # ---------------------------------------------------------------

    def test_empty_dir_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            result = _run_train(["--data-dir", empty])
        self.assertNotEqual(
            result.returncode, 0,
            f"train_surrogate.py must exit non-zero on empty data dir.\n"
            f"stdout={result.stdout}\nstderr={result.stderr}",
        )
        # Error message must point the operator at the documented fix.
        combined = (result.stdout + result.stderr).lower()
        self.assertIn("physics-extracted", combined,
                      "Error message must mention physics-extracted samples.")
        self.assertIn("--allow-synthetic-for-benchmark-only", combined,
                      "Error message must advertise the benchmark opt-in flag.")

    def test_missing_dir_fails_fast(self) -> None:
        """The default ``data/training/`` path does not exist in a clean clone."""
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "does", "not", "exist")
            result = _run_train(["--data-dir", missing])
        self.assertNotEqual(
            result.returncode, 0,
            "train_surrogate.py must exit non-zero when --data-dir is missing.",
        )

    def test_dir_without_csv_glob_fails_fast(self) -> None:
        """A non-empty directory that lacks ``samples_*.csv`` must also fail."""
        with tempfile.TemporaryDirectory() as tmp:
            # Drop a noise file that does NOT match the samples_*.csv glob.
            (Path(tmp) / "README.txt").write_text("noise")
            result = _run_train(["--data-dir", tmp])
        self.assertNotEqual(
            result.returncode, 0,
            "train_surrogate.py must fail when no samples_*.csv is present.",
        )

    # ---------------------------------------------------------------
    # Acceptance criterion #2: benchmark opt-in still works
    # ---------------------------------------------------------------

    def test_benchmark_flag_unlocks_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            result = _run_train(
                ["--data-dir", empty, "--allow-synthetic-for-benchmark-only"],
            )
        self.assertEqual(
            result.returncode, 0,
            f"--allow-synthetic-for-benchmark-only must succeed on empty dir.\n"
            f"stdout={result.stdout}\nstderr={result.stderr}",
        )
        combined = (result.stdout + result.stderr).lower()
        self.assertIn("synthetic data", combined,
                      "Benchmark branch must log the synthetic-data backfill.")

    # ---------------------------------------------------------------
    # Acceptance criterion #3: callsite annotations
    # ---------------------------------------------------------------

    def test_synthetic_callsites_annotated(self) -> None:
        """Every actual ``generate_synthetic_thermal_data(`` callsite must
        carry the ``synthetic-only benchmark path — NOT for production
        models`` tag (or be inside the function whose docstring carries it).

        Uses AST so that docstring cross-references like
        ``:func:`generate_synthetic_thermal_data``` are NOT counted as
        callsites — the acceptance criterion applies to actual call sites.
        """
        import ast
        import re

        src = TRAIN_SCRIPT.read_text()
        tag = "synthetic-only benchmark path — NOT for production models"

        tree = ast.parse(src, filename=str(TRAIN_SCRIPT))

        # 1. Collect real call sites (Call nodes whose func.name matches).
        call_sites: list[tuple[str, int]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name == "generate_synthetic_thermal_data":
                    call_sites.append((name, node.lineno))

        # We expect at least one call site (from load_training_data's
        # benchmark-only branch); before the fix there were two.
        self.assertGreaterEqual(
            len(call_sites), 1,
            "Expected ≥1 call site of generate_synthetic_thermal_data in "
            "the benchmark-only branch.",
        )

        # 2. For each call site, the tag must appear in the 12-line window
        # above OR below the call (covers both docstring-on-def patterns
        # and inline `#` comments).
        lines = src.splitlines()
        for _name, lineno in call_sites:
            window = "\n".join(lines[max(0, lineno - 12): lineno + 1])
            self.assertIn(
                tag, window,
                f"generate_synthetic_thermal_data call at line {lineno} "
                f"is missing the '{tag}' annotation within 12 lines above.",
            )

        # 3. The function definition itself must carry the tag (it is the
        # single canonical place where the policy lives).
        func_defs = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "generate_synthetic_thermal_data"
        ]
        self.assertEqual(
            len(func_defs), 1,
            "Expected exactly one definition of generate_synthetic_thermal_data.",
        )
        def_lineno = func_defs[0].lineno
        window = "\n".join(lines[max(0, def_lineno - 1): def_lineno + 30])
        self.assertIn(
            tag, window,
            f"generate_synthetic_thermal_data definition at line "
            f"{def_lineno} is missing the '{tag}' annotation.",
        )

        # 4. No raw ``generate_synthetic_thermal_data(`` call line should
        # be free of the tag in a ±6 line window — i.e. even inline calls
        # must surface a tag in their immediate context.
        for _name, lineno in call_sites:
            tight = "\n".join(lines[max(0, lineno - 6): lineno + 6])
            self.assertRegex(
                tight, re.compile(re.escape(tag)),
                f"Call at line {lineno} lacks the '{tag}' tag in its "
                f"immediate context.",
            )

    def test_main_no_silent_fallback(self) -> None:
        """``main()`` must NOT wrap ``load_training_data`` in a broad
        ``except Exception`` that re-introduces the synthetic fallback.
        """
        src = TRAIN_SCRIPT.read_text()
        # Find the main() function body and assert no broad-except fallback.
        main_start = src.find("def main():")
        self.assertGreater(main_start, -1, "main() not found.")
        main_body = src[main_start:]
        self.assertNotIn(
            "except Exception", main_body,
            "main() must not catch Exception around load_training_data; "
            "this is the silent synthetic fallback Issue #1338 removed.",
        )

    def test_load_training_data_signature(self) -> None:
        """``load_training_data`` must accept the new gate parameter."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_ts_under_test", TRAIN_SCRIPT,
        )
        self.assertIsNotNone(spec, "Could not build module spec for train_surrogate.py")
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        import inspect

        sig = inspect.signature(module.load_training_data)
        self.assertIn(
            "allow_synthetic_for_benchmark_only", sig.parameters,
            "load_training_data must accept allow_synthetic_for_benchmark_only.",
        )

    def test_help_advertises_flag(self) -> None:
        """The CLI must surface the new flag in --help so operators can find it."""
        result = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT), "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn(
            "--allow-synthetic-for-benchmark-only", result.stdout,
            "--help must advertise the benchmark-only opt-in flag.",
        )


if __name__ == "__main__":
    unittest.main()