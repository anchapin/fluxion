#!/usr/bin/env python3
"""
TDQS Python cross-check implementation.

Mirrors the Rust implementation in tdqs.rs for double-checking results
and for use in `scripts/check_tdqs_regression.py`.

Formula:
  TDQS = Σᵢ [correct(dᵢ) × w(dᵢ) × cost_avoided(dᵢ)]
         ─────────────────────────────────────────────
         Σᵢ [w(dᵢ) × cost_available(dᵢ)]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Decision type catalogue
# ---------------------------------------------------------------------------

DECISION_WEIGHTS: dict[str, float] = {
    "solver_selection":    3.0,
    "adaptive_timestep":   1.5,
    "surrogate_routing":   2.0,
    "constraint_warning":  1.0,
    "hvac_horizon":        1.5,
}

MAX_COST_AVAILABLE_S: dict[str, float] = {
    "solver_selection":    300.0,
    "adaptive_timestep":    45.0,
    "surrogate_routing":     2.0,
    "constraint_warning":   30.0,
    "hvac_horizon":         10.0,
}

DECISION_TYPES = list(DECISION_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DecisionInstance:
    decision_type: str          # one of DECISION_TYPES
    correct: bool
    cost_avoided_s: float       # 0.0 when correct=False
    source_case: Optional[str] = None
    timestep_index: Optional[int] = None

    def numerator_contribution(self) -> float:
        if not self.correct:
            return 0.0
        w = DECISION_WEIGHTS[self.decision_type]
        return w * self.cost_avoided_s

    def denominator_contribution(self) -> float:
        w = DECISION_WEIGHTS[self.decision_type]
        c = MAX_COST_AVAILABLE_S[self.decision_type]
        return w * c

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionInstance":
        return cls(
            decision_type=d["decision_type"],
            correct=bool(d["correct"]),
            cost_avoided_s=float(d.get("cost_avoided_s", 0.0)),
            source_case=d.get("source_case"),
            timestep_index=d.get("timestep_index"),
        )


# ---------------------------------------------------------------------------
# TDQS computation
# ---------------------------------------------------------------------------

def compute_tdqs(decisions: list[DecisionInstance]) -> float:
    """Compute TDQS over a list of decisions. Returns 0.0 for empty list."""
    if not decisions:
        return 0.0
    num = sum(d.numerator_contribution() for d in decisions)
    den = sum(d.denominator_contribution() for d in decisions)
    if den == 0.0:
        return 0.0
    return max(0.0, min(1.0, num / den))


@dataclass
class TdqsBreakdown:
    overall: float
    per_type: dict[str, dict]  # type → {tdqs, correct, total}

    def tdqs_for(self, dt: str) -> Optional[float]:
        return self.per_type.get(dt, {}).get("tdqs")

    def accuracy_for(self, dt: str) -> Optional[float]:
        info = self.per_type.get(dt)
        if not info or info["total"] == 0:
            return None
        return info["correct"] / info["total"]


def compute_tdqs_breakdown(decisions: list[DecisionInstance]) -> TdqsBreakdown:
    overall = compute_tdqs(decisions)
    per_type: dict[str, dict] = {}
    for dt in DECISION_TYPES:
        subset = [d for d in decisions if d.decision_type == dt]
        n_correct = sum(1 for d in subset if d.correct)
        per_type[dt] = {
            "tdqs": compute_tdqs(subset),
            "correct": n_correct,
            "total": len(subset),
        }
    return TdqsBreakdown(overall=overall, per_type=per_type)


def regression_detected(new_tdqs: float, baseline_tdqs: float, threshold: float = 0.05) -> bool:
    """Return True if new_tdqs is more than `threshold` below baseline_tdqs."""
    return (baseline_tdqs - new_tdqs) > threshold


def regression_by_type(
    new: TdqsBreakdown,
    baseline: TdqsBreakdown,
    threshold: float = 0.05,
) -> list[str]:
    """Return decision type names that regressed."""
    regressions = []
    for dt in DECISION_TYPES:
        new_score = new.tdqs_for(dt) or 0.0
        base_score = baseline.tdqs_for(dt) or 0.0
        if regression_detected(new_score, base_score, threshold):
            regressions.append(dt)
    return regressions


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def load_decisions_from_json(path: str) -> list[DecisionInstance]:
    import json
    with open(path) as f:
        data = json.load(f)
    records = data if isinstance(data, list) else data.get("decisions", [])
    return [DecisionInstance.from_dict(r) for r in records]


def load_baseline_from_json(path: str) -> TdqsBreakdown:
    import json
    with open(path) as f:
        data = json.load(f)
    overall = float(data.get("overall", 0.0))
    per_type = {}
    for dt in DECISION_TYPES:
        info = data.get("per_type", {}).get(dt, {})
        per_type[dt] = {
            "tdqs": float(info.get("tdqs", 0.0)),
            "correct": int(info.get("correct", 0)),
            "total": int(info.get("total", 0)),
        }
    return TdqsBreakdown(overall=overall, per_type=per_type)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    # All correct → TDQS = 1.0
    all_correct = [
        DecisionInstance(dt, True, MAX_COST_AVAILABLE_S[dt])
        for dt in DECISION_TYPES
    ]
    score = compute_tdqs(all_correct)
    assert abs(score - 1.0) < 1e-10, f"Expected 1.0, got {score}"

    # All incorrect → TDQS = 0.0
    all_wrong = [DecisionInstance(dt, False, 0.0) for dt in DECISION_TYPES]
    assert compute_tdqs(all_wrong) == 0.0

    # Empty → 0.0
    assert compute_tdqs([]) == 0.0

    # Regression gate
    assert regression_detected(0.65, 0.72, 0.05)
    assert not regression_detected(0.68, 0.72, 0.05)
    assert not regression_detected(0.72, 0.72, 0.05)

    print("tdqs.py self-test passed ✓")


if __name__ == "__main__":
    _self_test()
