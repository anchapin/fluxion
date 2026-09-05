#!/usr/bin/env python3
"""Stub for the future OpenEvolve adapter.

Issue #3338's recommendation (per `tools/evolution/README.md` and
the issue's "Why OpenEvolve" rationale) is that the campaign driver
itself lives **out-of-tree** — the in-tree harness
(`crates/fluxion-evaluator/`) is the only thing that *scores*
candidates. This stub documents how the OpenEvolve adapter is
expected to be invoked once installed (via `pip install openevolve`
or cloning the upstream repo).

The in-tree `run_bounded_campaign.py` is the **bounded re-run**:
it's the trust artifact the issue says must land in this PR even
when the full ≥200-gen OpenEvolve campaign is out of session scope.

# How to invoke the real OpenEvolve adapter

```text
$ ollama serve &
$ ollama pull qwen3.5:9.7B    # or qwen3.5:4b (whichever is local)
$ pip install openevolve
$ python3 tools/evolution/scripts/run_openevolve_campaign.py \
    --config tools/evolution/configs/solar_simd.yaml \
    --generations 200 --population 32 --islands 8 \
    --checkpoint-every 10
```

The OpenEvolve adapter (a follow-up PR) reads
`tools/evolution/configs/solar_simd.yaml`, drives OpenEvolve against
the in-tree `fluxion-evaluator` binary, and writes per-generation
summaries under `tools/evolution/results/solar_simd/`. The bounded
short re-run (this file's sibling, `run_bounded_campaign.py`)
documents that the harness reaches all seed kernels and that the
invariant battery trips on any fitness-regressing candidate.

This stub exits non-zero with a clear message pointing at the
companion script; it is **not** the bounded re-run.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "openevolve adapter is out-of-tree (issue #3338, see\n"
        "  tools/evolution/README.md\n"
        "  tools/evolution/scripts/run_bounded_campaign.py\n"
        "for the bounded re-run that lands in this PR.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
