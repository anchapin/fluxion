# Fluxion — Agent Instructions

## Required Reading (MANDATORY)

Before working on ANY issue, read `ARCHITECTURE.md` in the repository root. It contains:
- Module dependency diagram (Mermaid) with data flow
- Explicit input/output contracts for all 5 physics modules
- Trait hierarchy for ML surrogate swap points
- Module status table showing which modules are isolated and tested
- Single-timestep sequence diagram

**Rule**: Do NOT modify physics code without checking ARCHITECTURE.md first. If the code doesn't match the documented interfaces, update ARCHITECTURE.md to reflect reality OR fix the code to match the architecture.

## Validation Strategy (Current)

We are in **Phase 1: Module Isolation**. The rules:
1. **No ASHRAE 140 system-level testing** until individual modules pass E+ reference tests
2. **No parameter tuning** to make system tests pass — fix the underlying math
3. **Each module must match EnergyPlus within 1% tolerance** on isolated scenarios
4. Modules: Weather -> Solar -> Conduction -> Ventilation -> Zone Balance (test in this order)

## Module Boundaries

```
Weather (epw.rs)          -> Solar (solar.rs)           -> Zone Balance (thermal_model.rs)
                          -> Ventilation (ventilation.rs) -> Zone Balance
                          -> Conduction (solver_trait.rs) -> Zone Balance
```

Every module interaction goes through a Rust trait:
- `HeatConductionSolver` — conduction (5R1C, CTF, FD, future ML)
- `VentilationSchedule` — ventilation (constant, scheduled, weather-dependent)
- `ThermalModelTrait` — zone solver (physics, surrogate, hybrid)

## Key Files

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` | Module boundaries, I/O contracts, diagrams |
| `src/physics/solver_trait.rs` | HeatConductionSolver trait definition |
| `src/sim/thermal_model.rs` | ThermalModelTrait definition |
| `src/sim/solar.rs` | Solar position and irradiance calculations |
| `src/sim/ventilation.rs` | Ventilation schedule trait |
| `tests/reference_data/` | EnergyPlus CSV reference data for unit tests |

## Mathematical Reasoning — Use Python, NOT Mental Math

**Rule**: When performing calculations, verifying formulas, or reasoning about numerical results, **always write and execute Python code**. Never attempt mental arithmetic or "reason through" math in your head.

LLMs are probabilistic text predictors, not calculators. They predict the most likely next token, not the mathematically correct answer. This means:
- `8747.7704` can be confidently stated as `72` with no internal alarm
- Arithmetic errors compound silently through multi-step derivations
- Even simple operations like `1 + 1` are pattern completion, not computation

### What to code in Python (use `ctx_execute` with `language: "python"`)

| Task | Why code it |
|------|-------------|
| Unit conversions | Off-by-one errors in exponents are common |
| Formula verification | Symbolic vs numeric comparison catches typos |
| Reference data comparison | Exact floating-point comparison matters |
| Parameter calculations | Solar angles, thermal resistances, flow rates |
| Statistical analysis | Mean, std dev, tolerances, error metrics |
| Sanity checks | "Does this value make physical sense?" |

### Example pattern

**BAD** (mental math — unreliable):
> The solar altitude is arcsin(sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(hra)).
> For lat=40, dec=23.5, hra=30, that's about... 58 degrees.

**GOOD** (Python — verifiable):
```python
import math
lat, dec, hra = math.radians(40), math.radians(23.5), math.radians(30)
altitude = math.asin(math.sin(lat)*math.sin(dec) + math.cos(lat)*math.cos(dec)*math.cos(hra))
print(f"Solar altitude: {math.degrees(altitude):.4f}°")
```

### Reference
This rule is based on the finding that LLMs solve math reliably by coding, not by reasoning. See: [LLMs & Math: Problem Solved by Coding](https://gregrobison.medium.com/llms-math-problem-solved-by-coding-a5a5b5c4453a)

---

# context-mode — MANDATORY routing rules

You have context-mode MCP tools available. These rules are NOT optional — they protect your context window from flooding. A single unrouted command can dump 56 KB into context and waste the entire session.

## BLOCKED commands — do NOT attempt these

### curl / wget — BLOCKED
Any shell command containing `curl` or `wget` will be intercepted and blocked by the context-mode plugin. Do NOT retry.
Instead use:
- `context-mode_ctx_fetch_and_index(url, source)` to fetch and index web pages
- `context-mode_ctx_execute(language: "javascript", code: "const r = await fetch(...)")` to run HTTP calls in sandbox

### Inline HTTP — BLOCKED
Any shell command containing `fetch('http`, `requests.get(`, `requests.post(`, `http.get(`, or `http.request(` will be intercepted and blocked. Do NOT retry with shell.
Instead use:
- `context-mode_ctx_execute(language, code)` to run HTTP calls in sandbox — only stdout enters context

### Direct web fetching — BLOCKED
Do NOT use any direct URL fetching tool. Use the sandbox equivalent.
Instead use:
- `context-mode_ctx_fetch_and_index(url, source)` then `context-mode_ctx_search(queries)` to query the indexed content

## REDIRECTED tools — use sandbox equivalents

### Shell (>20 lines output)
Shell is ONLY for: `git`, `mkdir`, `rm`, `mv`, `cd`, `ls`, `npm install`, `pip install`, and other short-output commands.
For everything else, use:
- `context-mode_ctx_batch_execute(commands, queries)` — run multiple commands + search in ONE call
- `context-mode_ctx_execute(language: "shell", code: "...")` — run in sandbox, only stdout enters context

### File reading (for analysis)
If you are reading a file to **edit** it → reading is correct (edit needs content in context).
If you are reading to **analyze, explore, or summarize** → use `context-mode_ctx_execute_file(path, language, code)` instead. Only your printed summary enters context.

### grep / search (large results)
Search results can flood context. Use `context-mode_ctx_execute(language: "shell", code: "grep ...")` to run searches in sandbox. Only your printed summary enters context.

## Tool selection hierarchy

1. **GATHER**: `context-mode_ctx_batch_execute(commands, queries)` — Primary tool. Runs all commands, auto-indexes output, returns search results. ONE call replaces 30+ individual calls.
2. **FOLLOW-UP**: `context-mode_ctx_search(queries: ["q1", "q2", ...])` — Query indexed content. Pass ALL questions as array in ONE call.
3. **PROCESSING**: `context-mode_ctx_execute(language, code)` | `context-mode_ctx_execute_file(path, language, code)` — Sandbox execution. Only stdout enters context.
4. **WEB**: `context-mode_ctx_fetch_and_index(url, source)` then `context-mode_ctx_search(queries)` — Fetch, chunk, index, query. Raw HTML never enters context.
5. **INDEX**: `context-mode_ctx_index(content, source)` — Store content in FTS5 knowledge base for later search.

## Output constraints

- Keep responses under 500 words.
- Write artifacts (code, configs, PRDs) to FILES — never return them as inline text. Return only: file path + 1-line description.
- When indexing content, use descriptive source labels so others can `search(source: "label")` later.

## ctx commands

| Command | Action |
|---------|--------|
| `ctx stats` | Call the `stats` MCP tool and display the full output verbatim |
| `ctx doctor` | Call the `doctor` MCP tool, run the returned shell command, display as checklist |
| `ctx upgrade` | Call the `upgrade` MCP tool, run the returned shell command, display as checklist |
