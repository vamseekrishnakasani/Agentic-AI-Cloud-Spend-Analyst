"""Tests for the AzureCostClient module."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.azure_cost_analyst.config import AzureConfig
from src.azure_cost_analyst.cost_client import AzureCostClient


def _make_config(**kwargs) -> AzureConfig:
    defaults = {
        "tenant_id": "test-tenant",
        "client_id": "test-client",
        "client_secret": "test-secret",
        "subscription_id": "test-sub",
    }
    defaults.update(kwargs)
    return AzureConfig(**defaults)


def _make_mock_result(rows, columns):
    """Build a mock Azure API query result."""
    mock_col = lambda name: MagicMock(name=name)  # noqa: E731
    result = MagicMock()
    result.rows = rows
    result.columns = [mock_col(c) for c in columns]
    for col_mock, col_name in zip(result.columns, columns):
        col_mock.name = col_name
    return result


class TestAzureConfigValidate:
    def test_valid_config_passes(self):
        cfg = _make_config()
        cfg.validate()  # should not raise

    def test_missing_field_raises(self):
        cfg = _make_config(tenant_id="")
        with pytest.raises(ValueError, match="AZURE_TENANT_ID"):
            cfg.validate()

    def test_multiple_missing_fields_raises(self):
        cfg = _make_config(tenant_id="", client_id="")
        with pytest.raises(ValueError, match="AZURE_TENANT_ID"):
            cfg.validate()


class TestAzureCostClientInit:
    @patch("src.azure_cost_analyst.cost_client.CostManagementClient")
    @patch("src.azure_cost_analyst.cost_client.ClientSecretCredential")
    def test_init_success(self, mock_cred, mock_client):
        config = _make_config()
        client = AzureCostClient(config=config)
        mock_cred.assert_called_once_with(
            tenant_id="test-tenant",
            client_id="test-client",
            client_secret="test-secret",
        )
        assert client is not None

    def test_init_with_missing_config_raises(self):
        cfg = _make_config(subscription_id="")
        with pytest.raises(ValueError):
            AzureCostClient(config=cfg)


class TestGetDailyCosts:
    @patch("src.azure_cost_analyst.cost_client.CostManagementClient")
    @patch("src.azure_cost_analyst.cost_client.ClientSecretCredential")
    def test_returns_list_of_dicts(self, _mock_cred, mock_client_cls):
        rows = [[100.0, "2024-01-01", "USD"], [200.0, "2024-01-02", "USD"]]
        columns = ["PreTaxCost", "Date", "Currency"]
        mock_result = _make_mock_result(rows, columns)

        mock_cm = MagicMock()
        mock_cm.query.usage.return_value = mock_result
        mock_client_cls.return_value = mock_cm

        client = AzureCostClient(config=_make_config())
        result = client.get_daily_costs(lookback_days=30)

        assert len(result) == 2
        assert result[0]["PreTaxCost"] == 100.0
        assert result[0]["Date"] == "2024-01-01"

    @patch("src.azure_cost_analyst.cost_client.CostManagementClient")
    @patch("src.azure_cost_analyst.cost_client.ClientSecretCredential")
    def test_azure_error_raises_runtime_error(self, _mock_cred, mock_client_cls):
        from azure.core.exceptions import AzureError

        mock_cm = MagicMock()
        mock_cm.query.usage.side_effect = AzureError("API error")
        mock_client_cls.return_value = mock_cm

        client = AzureCostClient(config=_make_config())
        with pytest.raises(RuntimeError, match="Failed to fetch daily costs"):
            client.get_daily_costs()


class TestGetCostByService:
    @patch("src.azure_cost_analyst.cost_client.CostManagementClient")
    @patch("src.azure_cost_analyst.cost_client.ClientSecretCredential")
    def test_returns_service_records(self, _mock_cred, mock_client_cls):
        rows = [[500.0, "Compute", "USD"], [200.0, "Storage", "USD"]]
        columns = ["PreTaxCost", "ServiceName", "Currency"]
        mock_result = _make_mock_result(rows, columns)

        mock_cm = MagicMock()
        mock_cm.query.usage.return_value = mock_result
        mock_client_cls.return_value = mock_cm

        client = AzureCostClient(config=_make_config())
        result = client.get_cost_by_service(lookback_days=30)

        assert len(result) == 2
        assert result[0]["ServiceName"] == "Compute"


class TestRowsToDicts:
    def test_basic_conversion(self):
        rows = [[1, "a"], [2, "b"]]
        columns = ["id", "label"]
        result = AzureCostClient._rows_to_dicts(rows, columns)
        assert result == [{"id": 1, "label": "a"}, {"id": 2, "label": "b"}]

    def test_empty_rows(self):
        assert AzureCostClient._rows_to_dicts([], ["id"]) == []
