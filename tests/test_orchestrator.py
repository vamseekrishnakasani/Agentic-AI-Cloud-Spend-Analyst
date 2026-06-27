"""Tests for the CostAnalystOrchestrator module."""

from unittest.mock import MagicMock, patch

import pytest

from src.azure_cost_analyst.anomaly_detector import AnomalyDetector, AnomalyReport
from src.azure_cost_analyst.config import AppConfig, AzureConfig
from src.azure_cost_analyst.orchestrator import CostAnalystOrchestrator, OrchestratorResult


def _make_config(with_openai: bool = False) -> AppConfig:
    azure_cfg = AzureConfig(
        tenant_id="t",
        client_id="c",
        client_secret="s",
        subscription_id="sub",
    )
    return AppConfig(
        azure=azure_cfg,
        openai_api_key="sk-test" if with_openai else None,
    )


def _make_mock_cost_client(
    daily_costs=None, service_costs=None, rg_costs=None
) -> MagicMock:
    client = MagicMock()
    client.get_daily_costs.return_value = daily_costs or []
    client.get_cost_by_service.return_value = service_costs or []
    client.get_cost_by_resource_group.return_value = rg_costs or []
    return client


class TestOrchestratorResult:
    def test_succeeded_when_no_error(self):
        result = OrchestratorResult()
        assert result.succeeded

    def test_failed_when_error(self):
        result = OrchestratorResult(error="something went wrong")
        assert not result.succeeded


class TestCostAnalystOrchestratorRun:
    def test_run_with_empty_data_returns_result(self):
        mock_client = _make_mock_cost_client()
        config = _make_config()
        orch = CostAnalystOrchestrator(config=config, cost_client=mock_client)
        result = orch.run()
        # Empty data → collection route returns "end" (no anomaly detection)
        assert isinstance(result, OrchestratorResult)
        assert result.succeeded

    def test_run_with_normal_data_no_anomalies(self):
        costs = [{"date": f"2024-01-{i+1:02d}", "cost": 100.0} for i in range(20)]
        services = [{"service": "Compute", "cost": 100.0}]
        mock_client = _make_mock_cost_client(
            daily_costs=costs, service_costs=services
        )
        config = _make_config()
        orch = CostAnalystOrchestrator(config=config, cost_client=mock_client)
        result = orch.run()
        assert result.succeeded
        assert result.anomaly_report is not None
        assert not result.anomaly_report.has_anomalies

    def test_run_with_spike_detects_anomaly(self):
        costs = [{"date": f"2024-01-{i+1:02d}", "cost": 100.0} for i in range(19)]
        costs.append({"date": "2024-01-20", "cost": 5000.0})
        services = [{"service": "Compute", "cost": 100.0}]
        mock_client = _make_mock_cost_client(
            daily_costs=costs, service_costs=services
        )
        config = _make_config()
        orch = CostAnalystOrchestrator(config=config, cost_client=mock_client)
        result = orch.run()
        assert result.succeeded
        assert result.anomaly_report is not None
        assert result.anomaly_report.has_anomalies

    def test_run_returns_fallback_recommendations_without_openai(self):
        costs = [{"date": f"2024-01-{i+1:02d}", "cost": 100.0} for i in range(19)]
        costs.append({"date": "2024-01-20", "cost": 5000.0})
        services = [{"service": "Compute", "cost": 100.0}]
        mock_client = _make_mock_cost_client(
            daily_costs=costs, service_costs=services
        )
        config = _make_config(with_openai=False)
        orch = CostAnalystOrchestrator(config=config, cost_client=mock_client)
        result = orch.run()
        assert result.succeeded
        assert len(result.recommendations) > 0

    def test_data_collection_error_propagates(self):
        mock_client = _make_mock_cost_client()
        mock_client.get_daily_costs.side_effect = RuntimeError("Azure API down")
        config = _make_config()
        orch = CostAnalystOrchestrator(config=config, cost_client=mock_client)
        result = orch.run()
        assert not result.succeeded
        assert "Azure API down" in result.error

    def test_custom_anomaly_detector_used(self):
        costs = [{"date": f"2024-01-{i+1:02d}", "cost": 100.0} for i in range(20)]
        services = [{"service": "Compute", "cost": 100.0}]
        mock_client = _make_mock_cost_client(
            daily_costs=costs, service_costs=services
        )
        custom_detector = AnomalyDetector(threshold=0.1)  # very sensitive
        config = _make_config()
        orch = CostAnalystOrchestrator(
            config=config,
            cost_client=mock_client,
            anomaly_detector=custom_detector,
        )
        result = orch.run()
        assert result.succeeded
        # Custom detector is used: threshold=0.1 on uniform data → still no anomalies
        # (all z-scores are 0 for uniform data)
        assert result.anomaly_report is not None
