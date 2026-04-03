#!/usr/bin/env python3
"""
GSD-Mistral Vibe Adapter Layer

This adapter translates GSD commands to Mistral Vibe tool sequences,
allowing GSD workflows to run on Mistral Vibe CLI.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union


class GSDMistralAdapter:
    """Main adapter class for GSD-Mistral integration"""
    
    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir).absolute()
        self.planning_dir = self.project_dir / ".planning"
        self.config_file = self.planning_dir / "config.json"
        
        # Ensure planning directory exists
        self.planning_dir.mkdir(exist_ok=True)
        
        # Load or initialize config
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load GSD config or return defaults"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        
        # Default GSD config
        return {
            "mode": "interactive",
            "granularity": "standard",
            "model_profile": "balanced",
            "workflow": {
                "research": True,
                "plan_check": True,
                "verifier": True
            }
        }
    
    def _save_config(self):
        """Save current config to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _run_mistral_tool(self, tool_name: str, params: Dict) -> Dict:
        """Execute a Mistral Vibe tool and return result"""
        # This would be replaced with actual Mistral Vibe API calls
        # For now, we'll simulate the tool execution
        print(f"Executing {tool_name} with params: {params}")
        
        # Simulate different tool responses
        if tool_name == "ask_user_question":
            return {"response": input("User input: ")}
        elif tool_name == "task":
            return {"status": "completed", "result": "Task executed successfully"}
        elif tool_name == "read_file":
            file_path = params.get("path", "")
            try:
                with open(file_path, 'r') as f:
                    return {"content": f.read()}
            except FileNotFoundError:
                return {"error": f"File {file_path} not found"}
        
        return {"status": "success"}
    
    def _ensure_planning_structure(self):
        """Ensure basic GSD planning structure exists"""
        required_files = [
            "PROJECT.md",
            "REQUIREMENTS.md", 
            "ROADMAP.md",
            "STATE.md"
        ]
        
        for file_name in required_files:
            file_path = self.planning_dir / file_name
            if not file_path.exists():
                file_path.touch()
    
    # Core GSD Command Implementations
    
    def new_project(self, auto: bool = False, prd_file: Optional[str] = None):
        """Initialize a new GSD project"""
        print("Starting GSD project initialization...")
        
        # Step 1: Questions phase
        user_responses = []
        if not auto:
            project_questions = [
                {"question": "What is your project about?", "header": "Project"},
                {"question": "What are the main goals?", "header": "Goals"},
                {"question": "Any technical constraints?", "header": "Tech"}
            ]
            
            for q in project_questions:
                response = self._run_mistral_tool("ask_user_question", {
                    "questions": [q],
                    "content_preview": "Project initialization"
                })
                user_responses.append(response.get("response", ""))
        
        # Step 2: Create basic project files
        self._ensure_planning_structure()
        
        # Write PROJECT.md
        if user_responses and len(user_responses) >= 3:
            project_content = f"""# Project Vision

## Overview
{user_responses[0]}

## Goals
{user_responses[1]}

## Constraints
{user_responses[2]}
"""
        else:
            project_content = """# Project Vision

## Overview
New project

## Goals
To be defined

## Constraints
None
"""
        
        with open(self.planning_dir / "PROJECT.md", 'w') as f:
            f.write(project_content)
        
        # Create basic REQUIREMENTS.md and ROADMAP.md
        with open(self.planning_dir / "REQUIREMENTS.md", 'w') as f:
            f.write("# Requirements\n\n## v1 Requirements\n\n## v2 Requirements\n")
        
        with open(self.planning_dir / "ROADMAP.md", 'w') as f:
            f.write("# Roadmap\n\n## Phase 1: Foundation\n- [ ] Initial setup\n- [ ] Core functionality\n")
        
        # Save config file
        self._save_config()
        
        return {"status": "success", "message": "Project initialized"}
    
    def discuss_phase(self, phase_num: int):
        """Capture implementation decisions for a phase"""
        phase_dir = self.planning_dir / "phases" / f"{phase_num:02d}-phase-{phase_num}"
        phase_dir.mkdir(parents=True, exist_ok=True)
        
        context_file = phase_dir / "CONTEXT.md"
        
        # Get phase goal from roadmap
        roadmap_content = self._run_mistral_tool("read_file", {"path": str(self.planning_dir / "ROADMAP.md")})
        
        # Simple phase goal extraction (would be more sophisticated in real implementation)
        phase_goal = f"Phase {phase_num} implementation"
        
        # Ask discussion questions
        discussion_questions = [
            {"question": f"What are your preferences for {phase_goal}?", "header": "Pref"},
            {"question": "Any specific technologies to use?", "header": "Tech"},
            {"question": "What should be avoided?", "header": "Avoid"}
        ]
        
        responses = []
        for q in discussion_questions:
            response = self._run_mistral_tool("ask_user_question", {
                "questions": [q],
                "content_preview": f"Phase {phase_num} discussion"
            })
            responses.append(response.get("response", ""))
        
        # Write CONTEXT.md
        context_content = f"""# Phase {phase_num} Context

## Implementation Preferences
{responses[0]}

## Technology Choices
{responses[1]}

## Avoid
{responses[2]}
"""
        
        with open(context_file, 'w') as f:
            f.write(context_content)
        
        return {"status": "success", "context_file": str(context_file)}
    
    def plan_phase(self, phase_num: int):
        """Research and plan a phase"""
        # Search for or create phase directory - handle both standard format and descriptive names
        phases_dir = self.planning_dir / "phases"
        phase_dir = None
        
        # Try standard format first
        standard_dir = phases_dir / f"{phase_num:02d}-phase-{phase_num}"
        if standard_dir.exists():
            phase_dir = standard_dir
        else:
            # Search for directories starting with phase number
            for dir_entry in phases_dir.iterdir():
                if dir_entry.is_dir() and dir_entry.name.startswith(f"{phase_num:02d}-"):
                    phase_dir = dir_entry
                    break
        
        # If no existing directory found, create standard format
        if not phase_dir:
            phase_dir = phases_dir / f"{phase_num:02d}-phase-{phase_num}"
            phase_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Research (simulated)
        research_content = f"""# Phase {phase_num} Research

## Stack Analysis
- Language: Python
- Framework: FastAPI
- Database: SQLite

## Implementation Approaches
- Approach 1: Direct implementation
- Approach 2: Library-based

## Recommendations
- Use Approach 1 for simplicity
"""
        
        with open(phase_dir / "RESEARCH.md", 'w') as f:
            f.write(research_content)
        
        # Step 2: Create plan (simulated)
        plan_content = f"""# Phase {phase_num} Plan

## Task 1: Implement Core Feature
- Files: src/core.py
- Action: Create main functionality
- Verify: Test core feature works

## Task 2: Add Tests
- Files: tests/test_core.py
- Action: Write unit tests
- Verify: All tests pass
"""
        
        with open(phase_dir / f"{phase_num:02d}-01-PLAN.md", 'w') as f:
            f.write(plan_content)
        
        return {"status": "success", "plan_file": str(phase_dir / f"{phase_num:02d}-01-PLAN.md")}
    
    def execute_phase(self, phase_num: int):
        """Execute all plans in a phase"""
        # Search for phase directory - try both standard format and descriptive names
        phases_dir = self.planning_dir / "phases"
        phase_dir = None
        
        # Try standard format first
        standard_dir = phases_dir / f"{phase_num:02d}-phase-{phase_num}"
        if standard_dir.exists():
            phase_dir = standard_dir
        else:
            # Search for directories starting with phase number
            for dir_entry in phases_dir.iterdir():
                if dir_entry.is_dir() and dir_entry.name.startswith(f"{phase_num:02d}-"):
                    phase_dir = dir_entry
                    break
        
        if not phase_dir:
            return {"error": f"Phase {phase_num} directory not found"}
        
        # Find all PLAN files
        plan_files = list(phase_dir.glob("*PLAN.md"))
        
        if not plan_files:
            return {"error": f"No plans found in phase {phase_num} directory"}
        
        # Execute each plan (simulated)
        results = []
        for plan_file in plan_files:
            # In real implementation, this would spawn Mistral Vibe tasks
            result = self._run_mistral_tool("task", {
                "task": f"Execute plan: {plan_file.name}",
                "agent": "explore"
            })
            results.append(result)
        
        # Create summary
        summary_content = f"""# Phase {phase_num} Execution Summary

## Completed Tasks
- Task 1: Core feature implemented
- Task 2: Tests added and passing

## Files Changed
- src/core.py (new)
- tests/test_core.py (new)
"""
        
        with open(phase_dir / f"{phase_num:02d}-01-SUMMARY.md", 'w') as f:
            f.write(summary_content)
        
        return {"status": "success", "summary": summary_content}
    
    def progress(self):
        """Show current project progress"""
        # Check what files exist
        status = {
            "project_initialized": (self.planning_dir / "PROJECT.md").exists(),
            "requirements_defined": (self.planning_dir / "REQUIREMENTS.md").exists(),
            "roadmap_created": (self.planning_dir / "ROADMAP.md").exists(),
            "phases": []
        }
        
        # Check phases
        phases_dir = self.planning_dir / "phases"
        if phases_dir.exists():
            for phase_dir in phases_dir.iterdir():
                if phase_dir.is_dir():
                    phase_num = phase_dir.name.split('-')[0]
                    phase_status = {
                        "number": phase_num,
                        "discussed": (phase_dir / "CONTEXT.md").exists(),
                        "planned": len(list(phase_dir.glob("*PLAN.md"))) > 0,
                        "executed": len(list(phase_dir.glob("*SUMMARY.md"))) > 0
                    }
                    status["phases"].append(phase_status)
        
        return status


def main():
    """CLI entry point for GSD-Mistral adapter"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GSD-Mistral Vibe Adapter")
    parser.add_argument("command", help="GSD command to execute")
    parser.add_argument("--auto", action="store_true", help="Auto mode")
    parser.add_argument("--phase", type=int, help="Phase number")
    parser.add_argument("--project-dir", default=".", help="Project directory")
    
    args = parser.parse_args()
    
    adapter = GSDMistralAdapter(args.project_dir)
    
    if args.command == "new-project":
        result = adapter.new_project(auto=args.auto)
    elif args.command == "discuss-phase":
        if not args.phase:
            print("Error: --phase required for discuss-phase")
            return 1
        result = adapter.discuss_phase(args.phase)
    elif args.command == "plan-phase":
        if not args.phase:
            print("Error: --phase required for plan-phase")
            return 1
        result = adapter.plan_phase(args.phase)
    elif args.command == "execute-phase":
        if not args.phase:
            print("Error: --phase required for execute-phase")
            return 1
        result = adapter.execute_phase(args.phase)
    elif args.command == "progress":
        result = adapter.progress()
    else:
        print(f"Unknown command: {args.command}")
        return 1
    
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    main()