#!/usr/bin/env python3
"""Post a deterministic-gate failure PR comment.

Invoked by the `Fluxion Determinism Gate (Issue #1351)` listener job in
`.github/workflows/ashrae_validation.yml`. The bash heredoc that used to
embed this script (`<<'PYEOF'`) failed to parse on every workflow run
because the YAML multi-line string preserves indentation, but bash's
quoted heredoc terminator (`<<'PYEOF'`, no dash) requires the closing
delimiter at column 0. See:
  https://github.com/anchapin/fluxion/issues/1397

Environment variables (set by the calling bash step):
    UPSTREAM_CONCLUSION   "success" | "failure" | "cancelled" | "timed_out"
    UPSTREAM_RUN_URL      URL to the upstream workflow run
    UPSTREAM_HEAD_SHA     commit SHA being checked
    COMMENT_FILE         output path the bash step will `gh pr comment`
"""
import os

CONCLUSION = os.environ["UPSTREAM_CONCLUSION"]
RUN_URL = os.environ["UPSTREAM_RUN_URL"]
HEAD_SHA = os.environ["UPSTREAM_HEAD_SHA"]
COMMENT_FILE = os.environ["COMMENT_FILE"]

LINES = [
    "## :rotating_light: Cross-Platform Determinism Gate FAILED",
    "",
    f"The `Cross-Platform Determinism CI` workflow concluded **{CONCLUSION}** "
    f"for commit `{HEAD_SHA}` on this PR. The PR cannot be merged until the "
    "determinism check passes on **all three OS matrix entries** "
    "(ubuntu-latest, windows-latest, macos-latest).",
    "",
    f"**Upstream run:** {RUN_URL}",
    "",
    "### Common causes",
    "- A new `HashMap` / `HashSet` was introduced where a deterministic "
    "`BTreeMap` is required (see #1297).",
    "- A non-deterministic `f32` reduction path (SIMD reordering, parallel "
    "reduction) was added.",
    "- A new dependency pulled in non-portable FP code (rebuild against "
    "`--release --features wiring-tracing` to reproduce).",
    "",
    "### Re-run",
    "Push a new commit to this branch to re-trigger the upstream "
    "`Cross-Platform Determinism CI` workflow.",
    "",
    "---",
    "*This comment is auto-posted by the `Fluxion Determinism Gate` listener "
    "job (issue #1351, closing #1297 acceptance gap).*",
]

with open(COMMENT_FILE, "w") as f:
    f.write("\n".join(LINES))