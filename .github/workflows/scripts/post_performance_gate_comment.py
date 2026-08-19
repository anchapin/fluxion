#!/usr/bin/env python3
"""Write the Fluxion Performance Gate failure PR comment to a file.

Invoked by the `fluxion-performance-gate` listener job in
`.github/workflows/ashrae_validation.yml`. The bash heredoc that used to
embed this body (`<<'COMMENTEOF'`) failed to parse on every workflow run
because the YAML multi-line string preserves indentation, but bash's
quoted heredoc terminator (`<<'COMMENTEOF'`, no dash) requires the closing
delimiter at column 0. The same root cause was previously fixed for the
determinism listener (`post_determinism_gate_comment.py`); this script
extends the established pattern to the performance-gate listener.

See: https://github.com/anchapin/fluxion/issues/3117
"""
import os

OUTPUT_FILE = os.environ["OUTPUT_FILE"]

BODY = """## Performance Regression Detected :warning:

The **Fluxion Performance Gate** has failed. Upstream benchmarks show a performance regression exceeding the 10% threshold.

### Action Required
Please investigate the performance regression and either:
- Fix the regression before merging, OR
- Establish a new baseline on `main` if the regression is intentional

### Details
See the upstream Performance Dashboard run for detailed benchmark results.
"""

with open(OUTPUT_FILE, "w") as f:
    f.write(BODY)
