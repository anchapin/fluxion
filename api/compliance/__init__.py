"""
Compliance Agent Module for ASHRAE 90.1 / IECC

This module provides an LLM-powered agent that checks building energy models
for compliance with ASHRAE 90.1 and IECC standards.

Usage:
    from api.compliance import ComplianceAgent, ComplianceDataAggregator
    
    # Process simulation results
    aggregator = ComplianceDataAggregator(building_area_m2=1000)
    metrics = aggregator.process_simulation_results(
        hourly_temperatures=temps,
        heating_loads=heating,
        cooling_loads=cooling,
    )
    
    # Create agent and check compliance
    agent = ComplianceAgent()
    result = agent.check_compliance(metrics, baseline_metrics)

For LLM-powered report generation:
    from api.compliance import create_compliance_agent
    
    agent = create_compliance_agent(standard="ASHRAE 90.1-2019")
    prompt = agent.generate_prompt(proposed_metrics, baseline_metrics)
    # Send to LLM for natural language report
"""

from api.compliance.agent import (
    ComplianceAgent,
    ComplianceAgentConfig,
    create_compliance_agent,
)

from api.compliance.data_aggregation import (
    ComplianceMetrics,
    ComplianceDataAggregator,
    create_sample_metrics,
)

from api.compliance.prompt_engine import (
    ComplianceStandard,
    ReportFormat,
    CompliancePromptEngine,
    create_prompt_for_llm,
)

from api.compliance.report_generator import (
    ComplianceReportGenerator,
    ReportMetadata,
    generate_compliance_report,
)

__all__ = [
    # Agent
    "ComplianceAgent",
    "ComplianceAgentConfig",
    "create_compliance_agent",
    # Data aggregation
    "ComplianceMetrics",
    "ComplianceDataAggregator",
    "create_sample_metrics",
    # Prompt engine
    "ComplianceStandard",
    "ReportFormat",
    "CompliancePromptEngine",
    "create_prompt_for_llm",
    # Report generator
    "ComplianceReportGenerator",
    "ReportMetadata",
    "generate_compliance_report",
]
