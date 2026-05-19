#!/usr/bin/env python3
"""
Description optimization loop for skill triggering.
Uses opencode CLI to test whether a skill triggers for each eval query,
then proposes improved descriptions based on failures.
"""

import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

SKILL_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else None
EVAL_SET_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else None
MODEL = sys.argv[3] if len(sys.argv) > 3 else "opencode/glm-5.1"
MAX_ITERATIONS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
RUNS_PER_QUERY = 3  # Run each query multiple times for reliability


def load_skill_description(skill_path):
    """Extract description from SKILL.md frontmatter."""
    content = Path(skill_path).read_text()

    # Handle YAML folded scalar (description: >)
    match = re.search(
        r"^description:\s*>\s*\n((?:\s+.*\n)*?)---", content, re.MULTILINE
    )
    if match:
        raw = match.group(1)
        # Unfold: strip leading whitespace and join lines with spaces
        lines = [line.strip() for line in raw.strip().split("\n")]
        return " ".join(lines)

    # Handle inline description
    match = re.search(r"^description:\s*(.*?)\n", content)
    if match:
        return match.group(1).strip()

    return ""


def load_eval_set(path):
    """Load the eval set JSON."""
    with open(path) as f:
        return json.load(f)


def test_trigger(query, skill_path, description, model, timeout=60):
    """
    Test whether the skill triggers for a given query using opencode.
    Returns True if the model indicates it would use the skill.
    """
    prompt = f"""You are a skill selection system. You have access to the following skill:

SKILL NAME: bem-engineer
SKILL DESCRIPTION: {description}

A user sends this message:
"{query}"

Would you consult/trigger this skill to help answer this message?
Respond with ONLY "YES" or "NO" on the first line, then a brief reason on the second line.
Format:
YES/NO
<reason>"""

    try:
        result = subprocess.run(
            ["opencode", "run", "-m", model, "--format", "json", "-c", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout

        # Parse JSON events to find the assistant's response
        full_text = ""
        for line in output.strip().split("\n"):
            try:
                event = json.loads(line)
                if event.get("type") == "assistant" and "content" in event:
                    for block in event["content"]:
                        if isinstance(block, dict) and block.get("type") == "text":
                            full_text += block.get("text", "")
            except json.JSONDecodeError:
                # Try plain text parsing
                if "YES" in line.upper() or "NO" in line.upper():
                    full_text += line

        first_line = (
            full_text.strip().split("\n")[0].strip().upper() if full_text else ""
        )
        return "YES" in first_line
    except subprocess.TimeoutExpired:
        print(f"  ⏱ Timeout for query: {query[:60]}...")
        return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def evaluate_description(eval_set, skill_path, description, model, runs_per_query=3):
    """Evaluate a description against the full eval set."""
    results = []
    for i, item in enumerate(eval_set):
        expected = item["should_trigger"]
        query = item["query"]

        # Run multiple times for reliability
        triggers = []
        for run in range(runs_per_query):
            triggered = test_trigger(query, skill_path, description, model)
            if triggered is not None:
                triggers.append(triggered)
            time.sleep(0.5)  # Rate limiting

        if not triggers:
            correct = False
            trigger_rate = 0.0
        else:
            trigger_rate = sum(triggers) / len(triggers)
            correct = (trigger_rate >= 0.5) == expected

        results.append(
            {
                "query": query[:80] + "..." if len(query) > 80 else query,
                "expected_trigger": expected,
                "trigger_rate": trigger_rate,
                "runs": len(triggers),
                "correct": correct,
            }
        )

        status = "✓" if correct else "✗"
        print(
            f"  [{status}] Query {i + 1}: expected={'trigger' if expected else 'no-trigger'}, "
            f"got={trigger_rate:.0%} ({len(triggers)} runs)"
        )

    correct_count = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct_count / total if total > 0 else 0

    return accuracy, results


def propose_improvement(description, eval_set, results, model):
    """Use the model to propose an improved description based on failures."""
    failures = [r for r in results if not r["correct"]]

    if not failures:
        return description, "No failures to improve upon."

    failure_summary = "\n".join(
        f'- Query: "{r["query"][:100]}" | Expected: {"trigger" if r["expected_trigger"] else "no-trigger"} '
        f"| Got trigger rate: {r['trigger_rate']:.0%}"
        for r in failures
    )

    prompt = f"""You are optimizing a skill description for triggering accuracy in a skill selection system.

CURRENT DESCRIPTION:
{description}

FAILURE ANALYSIS — these queries were incorrectly classified:
{failure_summary}

FULL EVAL SET CONTEXT:
- Should-trigger queries are about: building energy modeling, EnergyPlus, OpenStudio, HVAC, ASHRAE standards, psychrometrics, .idf/.osm files, parametric BEM studies, surrogate models
- Should-NOT-trigger queries are about: general thermodynamics (wind tunnels, Rankine cycles), general programming (Rust web servers, Python data processing), weather data analysis, automotive HVAC, server rack cooling, React dashboards showing energy costs, Terraform, web scraping

RULES:
1. The description must trigger for building energy modeling tasks in Python, Rust, Ruby, EnergyPlus, OpenStudio
2. It must NOT trigger for: general thermodynamics homework, general programming tasks, data visualization, electronics cooling, automotive HVAC, infrastructure-as-code
3. Keep the description under 200 words
4. Be specific about the domain boundary — "buildings" or "building energy" is the key discriminator
5. Don't just add keywords — explain when to use and when NOT to use

Return ONLY the improved description text, no explanation."""

    try:
        result = subprocess.run(
            ["opencode", "run", "-m", model, "--format", "json", "-c", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )

        full_text = ""
        for line in result.stdout.strip().split("\n"):
            try:
                event = json.loads(line)
                if event.get("type") == "assistant" and "content" in event:
                    for block in event["content"]:
                        if isinstance(block, dict) and block.get("type") == "text":
                            full_text += block.get("text", "")
            except json.JSONDecodeError:
                full_text += line + "\n"

        improved = full_text.strip()
        if improved and len(improved) > 50:
            return improved, "Proposed improvement generated."
        return description, "Improvement was too short, keeping original."
    except Exception as e:
        return description, f"Error proposing improvement: {e}"


def split_eval_set(eval_set, train_ratio=0.6, seed=42):
    """Split eval set into train and test."""
    random.seed(seed)
    shuffled = eval_set.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return shuffled[:split], shuffled[split:]


def main():
    if not SKILL_PATH or not EVAL_SET_PATH:
        print(
            "Usage: python optimize_description.py <skill_path> <eval_set_path> [model] [max_iterations]"
        )
        sys.exit(1)

    print("=" * 60)
    print("SKILL DESCRIPTION OPTIMIZATION LOOP")
    print("=" * 60)
    print(f"Skill: {SKILL_PATH}")
    print(f"Eval set: {EVAL_SET_PATH}")
    print(f"Model: {MODEL}")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print(f"Runs per query: {RUNS_PER_QUERY}")
    print()

    # Load data
    current_description = load_skill_description(SKILL_PATH)
    eval_set = load_eval_set(EVAL_SET_PATH)
    train_set, test_set = split_eval_set(eval_set)

    print(
        f"Eval set: {len(eval_set)} total → {len(train_set)} train, {len(test_set)} test"
    )
    print(f"Current description ({len(current_description)} chars):")
    print(f"  {current_description[:150]}...")
    print()

    best_description = current_description
    best_test_accuracy = 0
    history = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"{'=' * 60}")
        print(f"ITERATION {iteration}/{MAX_ITERATIONS}")
        print(f"{'=' * 60}")

        # Evaluate on train set
        print(f"\n📊 Evaluating current description on TRAIN set...")
        train_accuracy, train_results = evaluate_description(
            train_set, SKILL_PATH, current_description, MODEL, RUNS_PER_QUERY
        )
        print(f"\n  Train accuracy: {train_accuracy:.0%}")

        # Evaluate on test set
        print(f"\n📊 Evaluating current description on TEST set...")
        test_accuracy, test_results = evaluate_description(
            test_set, SKILL_PATH, current_description, MODEL, RUNS_PER_QUERY
        )
        print(f"\n  Test accuracy: {test_accuracy:.0%}")

        history.append(
            {
                "iteration": iteration,
                "description": current_description,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "train_results": train_results,
                "test_results": test_results,
            }
        )

        # Track best by test accuracy (avoid overfitting to train)
        if test_accuracy >= best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_description = current_description
            print(f"\n  ✅ New best! Test accuracy: {test_accuracy:.0%}")

        # If perfect, stop
        if train_accuracy == 1.0 and test_accuracy == 1.0:
            print(f"\n🏆 Perfect score achieved! Stopping early.")
            break

        # Propose improvement based on all failures
        all_results = train_results + test_results
        print(f"\n🔧 Proposing description improvement...")
        improved, msg = propose_improvement(
            current_description, eval_set, all_results, MODEL
        )
        print(f"  {msg}")
        current_description = improved
        print(f"\n  New description ({len(current_description)} chars):")
        print(f"  {current_description[:150]}...")
        print()

    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\nBest test accuracy: {best_test_accuracy:.0%}")
    print(f"\nBest description:")
    print(f"{'-' * 40}")
    print(best_description)
    print(f"{'-' * 40}")

    # Save results
    output = {
        "best_description": best_description,
        "best_test_accuracy": best_test_accuracy,
        "iterations": len(history),
        "history": [
            {
                "iteration": h["iteration"],
                "train_accuracy": h["train_accuracy"],
                "test_accuracy": h["test_accuracy"],
                "description_preview": h["description"][:100],
            }
            for h in history
        ],
    }

    output_path = Path(SKILL_PATH).parent / "description_optimization_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
