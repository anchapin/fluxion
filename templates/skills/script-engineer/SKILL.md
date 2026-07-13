# Script Engineer Skill Template

> **TL;DR**: Guidance for writing effective scripts and automation within Fluxion.
> **Key decisions**: Scripts live in bin/ and tools/ | Shell for orchestration | Python for data processing | Always add tests.
> **Owned by**: DevOps team
> **Reviewed**: 2026-07-13

## When to Write a Script

Write a script when:
- A task is repeated 3+ times
- The steps are error-prone if done manually
- The task requires running multiple commands in sequence
- You want to automate a workflow for other team members

Do NOT write a script when:
- It's a one-off task with no repetition expected
- A Makefile target or existing tool already handles it
- The task is better handled by a CI/CD pipeline

## Script Conventions

### Location
- **bin/** — User-facing CLI tools and automation
- **tools/** — Developer utilities and data processing
- **scripts/** — Internal/build scripts (not user-facing)

### Shell Scripts
```bash
#!/usr/bin/env bash
set -euo pipefail  # Strict mode

# Always use absolute paths or resolve relative to script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
```

### Python Scripts
```python
#!/usr/bin/env python3
"""Script description."""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--input", required=True, help="Input file")
    args = parser.parse_args()

if __name__ == "__main__":
    main()
```

## Testing Scripts

Every script should have tests:
- Shell scripts: Test with `bats` (Bash Automated Testing)
- Python scripts: Use `pytest`

```bash
# Test example for shell script
@test "script returns 0 on success" {
  run ./bin/my_script.sh
  assert_success
}
```

## Documentation Requirements

Each script must have:
1. Shebang line
2. Brief description comment
3. `--help` or `-h` flag support
4. Usage examples in comments or README

## Security Considerations

- Never hardcode secrets — use environment variables
- Validate all inputs
- Use `set -euo pipefail` in shell scripts
- Prefer existing libraries over custom implementations
