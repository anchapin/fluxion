"""
Tests for the Code Compliance Agent
"""

import pytest
import json
from pathlib import Path

from tools.compliance_agent import (
    CodeComplianceAgent,
    ComplianceCheckResult,
    ComplianceReport,
    ComplianceStatus,
    Standard,
    get_rules_for_standard,
)
from tools.compliance_agent.llm_backend import (
    MockLLMBackend,
    LLMBackendFactory,
)


class TestComplianceRules:
    """Test compliance rules definitions."""
    
    def test_get_rules_for_standard_ashrae(self):
        rules = get_rules_for_standard(Standard.ASHRAE90_1_2019)
        assert len(rules) > 0
        assert any(r.rule_id.startswith("5.") for r in rules)
    
    def test_get_rules_for_standard_iecc(self):
        rules = get_rules_for_standard(Standard.IECC_2021)
        assert len(rules) > 0
        assert any(r.rule_id.startswith("R") for r in rules)


class TestMockBackend:
    """Test mock LLM backend."""
    
    def test_mock_backend_creation(self):
        backend = MockLLMBackend()
        assert backend.name == "mock"
        assert backend.is_available()
    
    def test_mock_backend_generate(self):
        backend = MockLLMBackend(response_delay_ms=10.0)
        response = backend.generate(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100
        )
        
        assert response.content is not None
        assert response.model == "mock-llm"
        assert response.latency_ms >= 10.0
    
    def test_mock_backend_compliance_response(self):
        backend = MockLLMBackend()
        response = backend.generate(
            messages=[{"role": "user", "content": "Check ASHRAE compliance for my building"}],
            temperature=0.7,
            max_tokens=100
        )
        
        # Should contain JSON with compliance checks
        data = json.loads(response.content)
        assert "compliance_checks" in data
        assert "overall_status" in data


class TestLLMBackendFactory:
    """Test LLM backend factory."""
    
    def test_create_mock_backend(self):
        backend = LLMBackendFactory.create("mock")
        assert isinstance(backend, MockLLMBackend)
    
    def test_list_backends(self):
        backends = LLMBackendFactory.list_backends()
        assert "mock" in backends
        assert "ollama" in backends
        assert "openai" in backends
    
    def test_create_unknown_backend(self):
        with pytest.raises(ValueError):
            LLMBackendFactory.create("unknown_backend")


class TestComplianceAgent:
    """Test the main compliance agent."""
    
    @pytest.fixture
    def agent(self):
        return CodeComplianceAgent(backend="mock")
    
    @pytest.fixture
    def sample_model_data(self):
        return {
            "model_name": "Test Building",
            "wall_r_value": 15.0,
            "window_u_factor": 0.35,
            "hvac_cop": 3.5,
            "lighting_power_density": 0.8,
            "water_heater_efficiency": 0.92,
        }
    
    def test_agent_creation(self):
        agent = CodeComplianceAgent(backend="mock")
        assert agent.is_available()
    
    def test_check_compliance_with_mock(self, agent, sample_model_data):
        report = agent.check_compliance(
            model_data=sample_model_data,
            standard=Standard.ASHRAE90_1_2019,
            use_rules_engine=True
        )
        
        assert isinstance(report, ComplianceReport)
        assert report.model_name == "Test Building"
        assert len(report.checks) > 0
        assert report.overall_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.NON_COMPLIANT, ComplianceStatus.NEEDS_REVIEW]
    
    def test_check_compliance_with_rules_engine(self, agent, sample_model_data):
        """Test that rules engine works even when LLM is not available."""
        # Create agent with a non-available backend
        agent = CodeComplianceAgent(
            backend="ollama",
            model="nonexistent"
        )
        
        report = agent.check_compliance(
            model_data=sample_model_data,
            standard=Standard.ASHRAE90_1_2019,
            use_rules_engine=True
        )
        
        assert isinstance(report, ComplianceReport)
        assert len(report.checks) > 0
        assert report.summary["total"] > 0
    
    def test_compliant_building(self, agent):
        """Test a fully compliant building."""
        model_data = {
            "model_name": "Compliant Building",
            "wall_r_value": 20.0,
            "roof_r_value": 40.0,
            "window_u_factor": 0.30,
            "window_shgc": 0.20,
            "hvac_cop": 4.0,
            "lighting_power_density": 0.5,
        }
        
        report = agent.check_compliance(
            model_data=model_data,
            standard=Standard.ASHRAE90_1_2019,
            use_rules_engine=True
        )
        
        # Most checks should pass
        assert report.summary["compliant"] > 0
    
    def test_non_compliant_building(self, agent):
        """Test a non-compliant building."""
        model_data = {
            "model_name": "Non-Compliant Building",
            "wall_r_value": 5.0,  # Too low
            "window_u_factor": 1.0,  # Too high
            "hvac_cop": 2.0,  # Too low
        }
        
        report = agent.check_compliance(
            model_data=model_data,
            standard=Standard.ASHRAE90_1_2019,
            use_rules_engine=True
        )
        
        assert report.summary["non_compliant"] > 0
    
    def test_iecc_standard(self, agent, sample_model_data):
        report = agent.check_compliance(
            model_data=sample_model_data,
            standard=Standard.IECC_2021,
            use_rules_engine=True
        )
        
        assert report.standard == "IECC-2021"
    
    def test_save_report_json(self, agent, sample_model_data, tmp_path):
        report = agent.check_compliance(
            model_data=sample_model_data,
            standard=Standard.ASHRAE90_1_2019,
            use_rules_engine=True
        )
        
        output_file = tmp_path / "test_report.json"
        agent.save_report(report, output_file, format="json")
        
        assert output_file.exists()
        
        # Verify the saved content
        with open(output_file) as f:
            saved_data = json.load(f)
        
        assert saved_data["model_name"] == "Test Building"
        assert "checks" in saved_data
    
    def test_save_report_txt(self, agent, sample_model_data, tmp_path):
        report = agent.check_compliance(
            model_data=sample_model_data,
            standard=Standard.ASHRAE90_1_2019,
            use_rules_engine=True
        )
        
        output_file = tmp_path / "test_report.txt"
        agent.save_report(report, output_file, format="txt")
        
        assert output_file.exists()
        
        # Verify the saved content contains expected text
        with open(output_file) as f:
            content = f.read()
        
        assert "COMPLIANCE REPORT" in content
        assert "Test Building" in content
    
    def test_load_model_data_json(self, agent, tmp_path):
        # Create a test JSON file
        model_data = {
            "model_name": "File Loaded Building",
            "wall_r_value": 15.0,
            "window_u_factor": 0.40
        }
        
        json_file = tmp_path / "model.json"
        with open(json_file, "w") as f:
            json.dump(model_data, f)
        
        loaded = agent.load_model_data(json_file)
        
        assert loaded["model_name"] == "File Loaded Building"
        assert loaded["wall_r_value"] == 15.0


class TestComplianceReport:
    """Test the ComplianceReport class."""
    
    def test_report_to_dict(self):
        report = ComplianceReport(
            model_name="Test",
            standard="ASHRAE90.1-2019",
            timestamp="2024-01-01T00:00:00",
            overall_status=ComplianceStatus.COMPLIANT,
            checks=[],
            summary={"total": 0, "compliant": 0}
        )
        
        data = report.to_dict()
        
        assert data["model_name"] == "Test"
        assert data["overall_status"] == "COMPLIANT"
    
    def test_report_to_json(self):
        report = ComplianceReport(
            model_name="Test",
            standard="ASHRAE90.1-2019",
            timestamp="2024-01-01T00:00:00",
            overall_status=ComplianceStatus.COMPLIANT,
            checks=[],
            summary={"total": 0, "compliant": 0}
        )
        
        json_str = report.to_json()
        
        assert '"model_name": "Test"' in json_str
    
    def test_report_print_summary(self):
        checks = [
            ComplianceCheckResult(
                rule_id="5.1.1",
                rule_title="Envelope Insulation",
                status=ComplianceStatus.COMPLIANT,
                message="Meets requirements"
            ),
            ComplianceCheckResult(
                rule_id="5.1.2",
                rule_title="Fenestration U-Factor",
                status=ComplianceStatus.NON_COMPLIANT,
                message="Exceeds maximum",
                recommendation="Replace windows"
            )
        ]
        
        report = ComplianceReport(
            model_name="Test Building",
            standard="ASHRAE90.1-2019",
            timestamp="2024-01-01T00:00:00",
            overall_status=ComplianceStatus.NON_COMPLIANT,
            checks=checks,
            summary={
                "total": 2,
                "compliant": 1,
                "non_compliant": 1
            }
        )
        
        summary = report.print_summary()
        
        assert "Test Building" in summary
        assert "Compliant: 1" in summary
        assert "Non-Compliant: 1" in summary
        assert "Fenestration U-Factor" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
