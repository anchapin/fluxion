"""
Code Compliance Agent for Building Energy Modeling

This module provides an automated compliance checking agent that uses LLMs
to verify building energy models against ASHRAE 90.1 and IECC standards.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from . import ComplianceCheckResult, ComplianceReport, ComplianceStatus, Standard, get_rules_for_standard
from .llm_backend import LLMBackend, LLMResponse, create_backend

logger = logging.getLogger(__name__)

# Default system prompt for the compliance agent
DEFAULT_SYSTEM_PROMPT = """You are an expert building energy modeling compliance agent.
Your task is to analyze building energy model data and check compliance against ASHRAE 90.1 and IECC standards.

When analyzing model data:
1. Extract relevant parameters (window U-factor, wall R-value, HVAC efficiency, lighting power density, etc.)
2. Compare each parameter against the compliance rules provided
3. Determine if the design meets, exceeds, or fails each requirement
4. Provide specific recommendations for any non-compliant items

Return your findings in JSON format with the following structure:
{
    "compliance_checks": [
        {
            "rule_id": "rule identifier",
            "status": "COMPLIANT" | "NON_COMPLIANT" | "NEEDS_REVIEW" | "NOT_APPLICABLE",
            "message": "description of the compliance status",
            "details": {"parameter": "value", "required": "threshold"},
            "recommendation": "optional recommendation for non-compliant items"
        }
    ],
    "overall_status": "COMPLIANT" | "NON_COMPLIANT" | "NEEDS_REVIEW",
    "summary": {
        "total_checks": number,
        "compliant": number,
        "non_compliant": number,
        "needs_review": number
    }
}

Be precise and reference specific standard sections when applicable."""


class CodeComplianceAgent:
    """
    Automated Code Compliance Agent for Building Energy Modeling.
    
    This agent uses LLMs to check building energy models against ASHRAE 90.1
    and IECC standards, generating detailed compliance reports.
    
    Example:
        >>> from tools.compliance_agent import CodeComplianceAgent
        >>> 
        >>> # Using mock backend for testing
        >>> agent = CodeComplianceAgent(backend="mock")
        >>> 
        >>> # Building model data
        >>> model_data = {
        ...     "model_name": "Office Building A",
        ...     "wall_r_value": 15.0,
        ...     "window_u_factor": 0.50,
        ...     "hvac_cop": 3.5,
        ...     "lighting_power_density": 0.8,
        ... }
        >>> 
        >>> # Check compliance
        >>> report = agent.check_compliance(model_data, standard="ASHRAE90.1-2019")
        >>> print(report.print_summary())
    """
    
    def __init__(
        self,
        backend: Union[str, LLMBackend] = "mock",
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        **backend_kwargs
    ):
        """
        Initialize the compliance agent.
        
        Args:
            backend: LLM backend to use ("mock", "ollama", "openai") or LLMBackend instance
            system_prompt: Custom system prompt (uses default if not provided)
            temperature: Sampling temperature for LLM
            max_tokens: Maximum tokens for LLM response
            **backend_kwargs: Additional arguments for the backend
        """
        self._backend = create_backend(backend, **backend_kwargs)
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._temperature = temperature
        self._max_tokens = max_tokens
        
        logger.info(f"Initialized CodeComplianceAgent with backend: {self._backend.name}")
    
    @property
    def backend(self) -> LLMBackend:
        """Get the LLM backend."""
        return self._backend
    
    def is_available(self) -> bool:
        """Check if the LLM backend is available."""
        return self._backend.is_available()
    
    def _build_prompt(
        self,
        model_data: Dict[str, Any],
        rules: List[Any]
    ) -> str:
        """Build the compliance checking prompt."""
        # Format the rules
        rules_text = []
        for rule in rules:
            threshold_str = ""
            if rule.threshold:
                threshold_str = f" (thresholds: {json.dumps(rule.threshold)})"
            rules_text.append(
                f"- {rule.rule_id}: {rule.title}\n"
                f"  Description: {rule.description}\n"
                f"  Parameters: {', '.join(rule.applicable_parameters)}{threshold_str}\n"
                f"  Reference: {rule.reference or 'N/A'}"
            )
        
        rules_str = "\n\n".join(rules_text)
        
        # Format the model data
        model_str = json.dumps(model_data, indent=2)
        
        prompt = f"""Analyze the following building energy model data for compliance with the specified standards.

MODEL DATA:
```json
{model_str}
```

COMPLIANCE RULES TO CHECK:
{rules_str}

Based on the model data and compliance rules, determine the compliance status for each rule and provide recommendations.

Return your findings in JSON format."""
        
        return prompt
    
    def _parse_llm_response(self, response: LLMResponse) -> Dict[str, Any]:
        """Parse the LLM response into structured data."""
        content = response.content
        
        # Try to extract JSON from the response
        # First try direct JSON parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in the response (might be wrapped in markdown)
        json_pattern = r'\{[\s\S]*\}'
        matches = re.findall(json_pattern, content)
        
        for match in matches:
            try:
                data = json.loads(match)
                if "compliance_checks" in data or "overall_status" in data:
                    return data
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, return a default structure
        logger.warning("Could not parse LLM response as JSON, using fallback")
        return {
            "compliance_checks": [],
            "overall_status": "ERROR",
            "summary": {"error": "Failed to parse LLM response"},
            "raw_response": content[:500]
        }
    
    def check_compliance(
        self,
        model_data: Dict[str, Any],
        standard: Union[str, Standard] = Standard.ASHRAE90_1_2019,
        rules: Optional[List[Any]] = None,
        use_rules_engine: bool = True,
    ) -> ComplianceReport:
        """
        Check compliance of building model data against a standard.
        
        Args:
            model_data: Dictionary containing building model parameters
            standard: The compliance standard to check against
            rules: Optional custom rules (uses default rules if not provided)
            use_rules_engine: If True, also run rule-based checks alongside LLM
            
        Returns:
            ComplianceReport with detailed findings
        """
        # Convert standard string to enum if needed
        if isinstance(standard, str):
            standard = Standard(standard)
        
        # Get rules for the standard
        if rules is None:
            rules = get_rules_for_standard(standard)
        
        model_name = model_data.get("model_name", "Unknown Model")
        
        # Run LLM-based compliance check
        llm_results = None
        llm_error = None
        
        try:
            if self._backend.is_available():
                prompt = self._build_prompt(model_data, rules)
                
                messages = [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                response = self._backend.generate(
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens
                )
                
                llm_results = self._parse_llm_response(response)
                logger.info(f"LLM compliance check completed in {response.latency_ms:.0f}ms")
            else:
                logger.warning(f"LLM backend ({self._backend.name}) is not available")
                llm_error = f"Backend {self._backend.name} is not available"
        except Exception as e:
            logger.error(f"LLM compliance check failed: {e}")
            llm_error = str(e)
        
        # Build compliance results
        checks: List[ComplianceCheckResult] = []
        
        # If LLM returned results, use them
        if llm_results and "compliance_checks" in llm_results:
            for check_data in llm_results.get("compliance_checks", []):
                try:
                    status = ComplianceStatus(check_data.get("status", "NEEDS_REVIEW"))
                except ValueError:
                    status = ComplianceStatus.NEEDS_REVIEW
                
                checks.append(ComplianceCheckResult(
                    rule_id=check_data.get("rule_id", "UNKNOWN"),
                    rule_title=check_data.get("rule_title", "Unknown Rule"),
                    status=status,
                    message=check_data.get("message", ""),
                    details=check_data.get("details", {}),
                    recommendation=check_data.get("recommendation")
                ))
        
        # If no LLM results or also running rules engine, use rule-based checks
        if use_rules_engine or not checks:
            rule_checks = self._run_rules_engine(model_data, rules)
            if rule_checks:
                # Merge with LLM results if available
                if checks:
                    existing_ids = {c.rule_id for c in checks}
                    for rc in rule_checks:
                        if rc.rule_id not in existing_ids:
                            checks.append(rc)
                else:
                    checks = rule_checks
        
        # Calculate summary
        summary = {
            "total": len(checks),
            "compliant": sum(1 for c in checks if c.status == ComplianceStatus.COMPLIANT),
            "non_compliant": sum(1 for c in checks if c.status == ComplianceStatus.NON_COMPLIANT),
            "needs_review": sum(1 for c in checks if c.status == ComplianceStatus.NEEDS_REVIEW),
            "not_applicable": sum(1 for c in checks if c.status == ComplianceStatus.NOT_APPLICABLE),
            "error": sum(1 for c in checks if c.status == ComplianceStatus.ERROR),
        }
        
        # Determine overall status
        if llm_results and "overall_status" in llm_results:
            try:
                overall_status = ComplianceStatus(llm_results.get("overall_status", "NEEDS_REVIEW"))
            except ValueError:
                overall_status = ComplianceStatus.NEEDS_REVIEW
        elif summary["non_compliant"] > 0:
            overall_status = ComplianceStatus.NON_COMPLIANT
        elif summary["compliant"] > 0 and summary["non_compliant"] == 0:
            overall_status = ComplianceStatus.COMPLIANT
        elif summary["needs_review"] > 0:
            overall_status = ComplianceStatus.NEEDS_REVIEW
        else:
            overall_status = ComplianceStatus.NEEDS_REVIEW
        
        # Build metadata
        metadata = {
            "llm_backend": self._backend.name,
            "llm_available": self._backend.is_available(),
            "llm_error": llm_error,
            "llm_latency_ms": llm_results.get("_latency_ms") if llm_results else None,
            "rules_engine_used": use_rules_engine,
            "standard_version": standard.value,
        }
        
        return ComplianceReport(
            model_name=model_name,
            standard=standard.value,
            timestamp=datetime.now().isoformat(),
            overall_status=overall_status,
            checks=checks,
            summary=summary,
            metadata=metadata
        )
    
    def _run_rules_engine(
        self,
        model_data: Dict[str, Any],
        rules: List[Any]
    ) -> List[ComplianceCheckResult]:
        """
        Run rule-based compliance checks.
        
        This is a deterministic check that compares model parameters
        against predefined thresholds from the standards.
        """
        checks = []
        
        for rule in rules:
            # Get relevant parameters from model data
            params = {}
            for param_name in rule.applicable_parameters:
                # Try exact match first, then case-insensitive
                if param_name in model_data:
                    params[param_name] = model_data[param_name]
                else:
                    # Try case-insensitive search
                    for key, value in model_data.items():
                        if key.lower() == param_name.lower():
                            params[param_name] = value
                            break
            
            # If no relevant parameters found, mark as needs review
            if not params:
                checks.append(ComplianceCheckResult(
                    rule_id=rule.rule_id,
                    rule_title=rule.title,
                    status=ComplianceStatus.NEEDS_REVIEW,
                    message=f"No relevant parameters found in model data for this rule.",
                    details={"applicable_parameters": rule.applicable_parameters},
                    severity="low"
                ))
                continue
            
            # Check each parameter against thresholds
            non_compliant_params = []
            compliant_params = []
            
            if rule.threshold:
                for param_name, threshold in rule.threshold.items():
                    if param_name in params:
                        value = params[param_name]
                        
                        # Determine if it's a minimum or maximum requirement
                        # Based on common conventions for building standards
                        if "u_factor" in param_name.lower() or "shgc" in param_name.lower():
                            # Lower is better (maximum allowed)
                            if value > threshold:
                                non_compliant_params.append((param_name, value, threshold, "max"))
                            else:
                                compliant_params.append((param_name, value, threshold, "max"))
                        elif "r_value" in param_name.lower() or "efficiency" in param_name.lower() or "cop" in param_name.lower():
                            # Higher is better (minimum required)
                            if value < threshold:
                                non_compliant_params.append((param_name, value, threshold, "min"))
                            else:
                                compliant_params.append((param_name, value, threshold, "min"))
                        elif "density" in param_name.lower() or "power" in param_name.lower():
                            # Lower is better (maximum allowed)
                            if value > threshold:
                                non_compliant_params.append((param_name, value, threshold, "max"))
                            else:
                                compliant_params.append((param_name, value, threshold, "max"))
            
            # Determine status
            if non_compliant_params:
                status = ComplianceStatus.NON_COMPLIANT
                messages = []
                for param, value, thresh, req_type in non_compliant_params:
                    if req_type == "max":
                        messages.append(
                            f"{param} ({value}) exceeds maximum allowed ({thresh})"
                        )
                    else:
                        messages.append(
                            f"{param} ({value}) is below minimum required ({thresh})"
                        )
                
                # Generate recommendation
                recommendation = None
                if non_compliant_params:
                    param_name = non_compliant_params[0][0]
                    if "u_factor" in param_name.lower():
                        recommendation = "Consider using higher performance insulation or glazing."
                    elif "r_value" in param_name.lower():
                        recommendation = "Consider adding more insulation to meet the R-value requirement."
                    elif "efficiency" in param_name.lower() or "cop" in param_name.lower():
                        recommendation = "Consider upgrading to more efficient HVAC equipment."
                    elif "density" in param_name.lower():
                        recommendation = "Consider using more efficient lighting systems."
                
                checks.append(ComplianceCheckResult(
                    rule_id=rule.rule_id,
                    rule_title=rule.title,
                    status=status,
                    message="; ".join(messages),
                    details={
                        "non_compliant": dict((p, v) for p, v, t, rt in non_compliant_params),
                        "compliant": dict((p, v) for p, v, t, rt in compliant_params),
                        "thresholds": rule.threshold
                    },
                    severity="high",
                    recommendation=recommendation
                ))
            elif compliant_params:
                status = ComplianceStatus.COMPLIANT
                checks.append(ComplianceCheckResult(
                    rule_id=rule.rule_id,
                    rule_title=rule.title,
                    status=status,
                    message=f"All checked parameters meet requirements.",
                    details={
                        "compliant": dict((p, v) for p, v, t, rt in compliant_params),
                        "thresholds": rule.threshold
                    },
                    severity="low"
                ))
            else:
                checks.append(ComplianceCheckResult(
                    rule_id=rule.rule_id,
                    rule_title=rule.title,
                    status=ComplianceStatus.NEEDS_REVIEW,
                    message="Parameters present but could not determine compliance.",
                    details={"parameters": params},
                    severity="medium"
                ))
        
        return checks
    
    def save_report(
        self,
        report: ComplianceReport,
        output_path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Save a compliance report to file.
        
        Args:
            report: The compliance report to save
            output_path: Path to save the report
            format: Output format ("json" or "txt")
        """
        output_path = Path(output_path)
        
        if format == "json":
            with open(output_path, "w") as f:
                f.write(report.to_json())
        elif format == "txt":
            with open(output_path, "w") as f:
                f.write(report.print_summary())
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Report saved to {output_path}")
    
    def load_model_data(
        self,
        file_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Load building model data from a file.
        
        Supports JSON and CSV formats.
        
        Args:
            file_path: Path to the model data file
            
        Returns:
            Dictionary containing the model data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix == ".json":
            with open(file_path) as f:
                return json.load(f)
        elif file_path.suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(file_path)
            # Convert DataFrame to dict, taking first row if multiple
            if len(df) > 1:
                logger.warning(f"CSV has {len(df)} rows, using first row")
            return df.iloc[0].to_dict()
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
