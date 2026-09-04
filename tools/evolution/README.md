# tools/evolution/

Out-of-tree evolver adapters (Python shims) that drive the
`fluxion-evaluator` binary.

This directory is **intentionally empty** in this PR. Issue #3336
delivers the in-tree harness contract only; the OpenEvolve adapter
itself is tracked as a follow-up PR (the adapter is out-of-tree by
design — see the issue's "Why OpenEvolve" rationale).

## Planned contents

- `openevolve_adapter.py` — thin Python shim that invokes the
  evaluator binary and maps schema-v1 JSON onto OpenEvolve's
  evaluator output contract (`score`/`scores` + validity).
- `config.yaml` — pinned OpenEvolve config (islands, population,
  checkpoint interval, `llm.api_base: http://localhost:11434/v1`).
- `README.md` — local-LLM setup blueprint
  (`ollama run qwen2.5-coder:32b` or
  `vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --port 11434`).

## Why not in-tree

The evolver is fundamentally an *external* campaign driver (it spins
up LLM queries, manages a population database, drives a checkpoint
loop). Keeping it out-of-tree lets users swap evolvers — OpenEvolve,
FunSearch, AlphaEvolve — without touching the harness. The harness
contract is evolver-agnostic.

## Tracking

Follow-up issue: see the repository issue tracker for the next PR in
the #3337/#3338/#3339 sequence (state-space CTF heuristics, solar /
radiation SIMD, BDF DAE).