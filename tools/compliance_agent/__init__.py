"""
Code Compliance Agent for Building Energy Modeling

This module provides an automated compliance checking agent that uses LLMs
to verify building energy models against ASHRAE 90.1 and IECC standards.

Supported LLM Backends:
- Mock: For testing without external dependencies
- Ollama: Local LLM using llama.cpp
- OpenAI: Cloud-based LLM (GPT-4, GPT-3.5)

Usage:
    from tools.compliance_agent import CodeComplianceAgent
    
    agent = CodeComplianceAgent(backend="mock")
    result = agent.check_compliance(model_data, standard="ASHRAE90.1")
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Standard(str, Enum):
    """Supported compliance standards."""
    ASHRAE90_1_2019 = "ASHRAE90.1-2019"
    ASHRAE90_1_2022 = "ASHRAE90.1-2022"
    IECC_2021 = "IECC-2021"
    IECC_2024 = "IECC-2024"


class ComplianceStatus(str, Enum):
    """Compliance check result status."""
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    ERROR = "ERROR"


@dataclass
class ComplianceRule:
    """Represents a single compliance rule from a standard."""
    rule_id: str
    title: str
    description: str
    category: str
    applicable_parameters: List[str]
    threshold: Optional[Dict[str, Any]] = None
    reference: Optional[str] = None


@dataclass
class ComplianceCheckResult:
    """Result of a single compliance check."""
    rule_id: str
    rule_title: str
    status: ComplianceStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "high"  # high, medium, low
    recommendation: Optional[str] = None


@dataclass
class ComplianceReport:
    """Complete compliance report for a building energy model."""
    model_name: str
    standard: str
    timestamp: str
    overall_status: ComplianceStatus
    checks: List[ComplianceCheckResult]
    summary: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "model_name": self.model_name,
            "standard": self.standard,
            "timestamp": self.timestamp,
            "overall_status": self.overall_status.value,
            "summary": self.summary,
            "checks": [
                {
                    "rule_id": c.rule_id,
                    "rule_title": c.rule_title,
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details,
                    "severity": c.severity,
                    "recommendation": c.recommendation,
                }
                for c in self.checks
            ],
            "metadata": self.metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def print_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"\n{'='*60}",
            f"COMPLIANCE REPORT: {self.model_name}",
            f"{'='*60}",
            f"Standard: {self.standard}",
            f"Timestamp: {self.timestamp}",
            f"\nSUMMARY:",
            f"  Total Checks: {self.summary.get('total', 0)}",
            f"  Compliant: {self.summary.get('compliant', 0)}",
            f"  Non-Compliant: {self.summary.get('non_compliant', 0)}",
            f"  Needs Review: {self.summary.get('needs_review', 0)}",
            f"  Not Applicable: {self.summary.get('not_applicable', 0)}",
            f"\nOVERALL STATUS: {self.overall_status.value}",
            f"{'='*60}\n",
        ]
        
        # Add non-compliant items
        non_compliant = [c for c in self.checks if c.status == ComplianceStatus.NON_COMPLIANT]
        if non_compliant:
            lines.append("NON-COMPLIANT ITEMS:")
            for check in non_compliant:
                lines.append(f"  - [{check.rule_id}] {check.rule_title}")
                lines.append(f"    {check.message}")
                if check.recommendation:
                    lines.append(f"    Recommendation: {check.recommendation}")
                lines.append("")
        
        return "\n".join(lines)


# Default compliance rules for ASHRAE 90.1
ASHRAE_90_1_RULES = [
    ComplianceRule(
        rule_id="5.1.1",
        title="Envelope Insulation",
        description="Building envelope thermal resistance must meet minimum R-values",
        category="Envelope",
        applicable_parameters=["wall_r_value", "roof_r_value", "floor_r_value"],
        threshold={"wall_r_value": 13.0, "roof_r_value": 30.0, "floor_r_value": 10.0},
        reference="ASHRAE 90.1-2019 Table 5.5-1",
    ),
    ComplianceRule(
        rule_id="5.1.2",
        title="Fenestration U-Factor",
        description="Windows and doors must meet maximum U-factor requirements",
        category="Envelope",
        applicable_parameters=["window_u_factor", "door_u_factor"],
        threshold={"window_u_factor": 0.40, "door_u_factor": 0.70},
        reference="ASHRAE 90.1-2019 Table 5.5-4",
    ),
    ComplianceRule(
        rule_id="5.1.3",
        title="Fenestration SHGC",
        description="Windows must meet solar heat gain coefficient requirements",
        category="Envelope",
        applicable_parameters=["window_shgc"],
        threshold={"window_shgc": 0.25},
        reference="ASHRAE 90.1-2019 Table 5.5-4",
    ),
    ComplianceRule(
        rule_id="6.1.1",
        title="HVAC Equipment Efficiency",
        description="HVAC equipment must meet minimum efficiency requirements",
        category="HVAC",
        applicable_parameters=["hvac_efficiency", "cop", "ieer"],
        threshold={"cop": 3.0, "ieer": 10.0},
        reference="ASHRAE 90.1-2019 Table 6.8.1-1",
    ),
    ComplianceRule(
        rule_id="6.2.2",
        title="HVAC Controls - Temperature",
        description="HVAC systems must have proper temperature controls",
        category="HVAC",
        applicable_parameters=["heating_setpoint", "cooling_setpoint", "deadband"],
        threshold={"deadband": 5.0},
        reference="ASHRAE 90.1-2019 Section 6.3.2",
    ),
    ComplianceRule(
        rule_id="6.3.1",
        title="Ventilation Requirements",
        description="Mechanical ventilation must meet minimum outdoor air rates",
        category="HVAC",
        applicable_parameters=["ventilation_rate", "outdoor_air_per_person"],
        threshold={"outdoor_air_per_person": 5.0},  # cfm/person
        reference="ASHRAE 90.1-2019 Table 6.2.2.1",
    ),
    ComplianceRule(
        rule_id="7.1.1",
        title="Lighting Power Density",
        description="Interior lighting power must not exceed maximum LPD",
        category="Lighting",
        applicable_parameters=["lighting_power_density"],
        threshold={"lighting_power_density": 0.9},  # W/sq ft
        reference="ASHRAE 90.1-2019 Table 9.3.1-1",
    ),
    ComplianceRule(
        rule_id="7.4.2",
        title="Lighting Controls",
        description="Lighting must have automatic shutoff controls",
        category="Lighting",
        applicable_parameters=["lighting_control_type", "occupancy_sensors"],
        reference="ASHRAE 90.1-2019 Section 9.4.1",
    ),
    ComplianceRule(
        rule_id="8.1.1",
        title="Service Water Heating",
        description="Water heating systems must meet efficiency requirements",
        category="Service Water Heating",
        applicable_parameters=["water_heater_efficiency", "water_heater_type"],
        threshold={"water_heater_efficiency": 0.90},
        reference="ASHRAE 90.1-2019 Table 8.8.1-1",
    ),
    ComplianceRule(
        rule_id="9.1.1",
        title="Electric Power",
        description="Transformers and motors must meet efficiency requirements",
        category="Electric Power",
        applicable_parameters=["transformer_efficiency", "motor_efficiency"],
        threshold={"transformer_efficiency": 0.97, "motor_efficiency": 0.95},
        reference="ASHRAE 90.1-2019 Section 9.9",
    ),
]

# Default compliance rules for IECC
IECC_RULES = [
    ComplianceRule(
        rule_id="R103.2",
        title="Envelope Thermal Performance",
        description="Building envelope must meet minimum thermal performance requirements",
        category="Envelope",
        applicable_parameters=["wall_r_value", "roof_r_value", "floor_r_value"],
        threshold={"wall_r_value": 20.0, "roof_r_value": 38.0, "floor_r_value": 10.0},
        reference="IECC 2021 Table R402.1.2",
    ),
    ComplianceRule(
        rule_id="R402.1.3",
        title="Fenestration U-Factor",
        description="Windows must meet maximum U-factor requirements",
        category="Envelope",
        applicable_parameters=["window_u_factor"],
        threshold={"window_u_factor": 0.30},
        reference="IECC 2021 Table R402.1.2",
    ),
    ComplianceRule(
        rule_id="R402.1.4",
        title="Fenestration SHGC",
        description="Windows must meet solar heat gain coefficient requirements",
        category="Envelope",
        applicable_parameters=["window_shgc"],
        threshold={"window_shgc": 0.25},
        reference="IECC 2021 Table R402.1.2",
    ),
    ComplianceRule(
        rule_id="R403.5.1",
        title="Mechanical Ventilation",
        description="Mechanical ventilation must provide minimum outdoor air",
        category="HVAC",
        applicable_parameters=["ventilation_rate"],
        threshold={"ventilation_rate": 0.03},  # ACH
        reference="IECC 2021 Section R403.5.1",
    ),
    ComplianceRule(
        rule_id="R404.1",
        title="Lighting Efficiency",
        description="Lighting must meet efficiency requirements",
        category="Lighting",
        applicable_parameters=["lighting_power_density"],
        threshold={"lighting_power_density": 0.8},  # W/sq ft
        reference="IECC 2021 Table R405.5.2(1)",
    ),
]


def get_rules_for_standard(standard: Standard) -> List[ComplianceRule]:
    """Get the compliance rules for a specific standard."""
    if "ASHRAE" in standard.value:
        return ASHRAE_90_1_RULES
    elif "IECC" in standard.value:
        return IECC_RULES
    else:
        logger.warning(f"Unknown standard: {standard}, returning ASHRAE 90.1 rules")
        return ASHRAE_90_1_RULES


# Import main agent class for convenience
from tools.compliance_agent.agent import CodeComplianceAgent

__all__ = [
    "CodeComplianceAgent",
    "ComplianceCheckResult",
    "ComplianceReport",
    "ComplianceRule",
    "ComplianceStatus",
    "Standard",
    "get_rules_for_standard",
]
