"""Azure Cost Management API client."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from azure.identity import ClientSecretCredential
from azure.mgmt.costmanagement import CostManagementClient
from azure.mgmt.costmanagement.models import (
    ExportType,
    GranularityType,
    QueryAggregation,
    QueryColumn,
    QueryDataset,
    QueryDefinition,
    QueryFilter,
    QueryGrouping,
    QueryTimePeriod,
    TimeframeType,
)
from azure.core.exceptions import AzureError

from src.azure_cost_analyst.config import AzureConfig, setup_logging

logger = setup_logging(__name__)


class AzureCostClient:
    """Client for the Azure Cost Management API.

    Wraps :class:`azure.mgmt.costmanagement.CostManagementClient` with
    convenience methods for querying daily costs, service breakdowns, and
    resource-group spending.

    Args:
        config: :class:`~src.azure_cost_analyst.config.AzureConfig` instance
            holding Azure credentials.  If *None* the config is built from
            environment variables.

    Raises:
        ValueError: When required credential fields are missing.
    """

    def __init__(self, config: Optional[AzureConfig] = None) -> None:
        self._config = config or AzureConfig()
        self._config.validate()
        self._client = self._build_client()
        logger.info(
            "AzureCostClient initialised for subscription %s",
            self._config.subscription_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self) -> CostManagementClient:
        credential = ClientSecretCredential(
            tenant_id=self._config.tenant_id,
            client_id=self._config.client_id,
            client_secret=self._config.client_secret,
        )
        return CostManagementClient(credential)

    @property
    def _scope(self) -> str:
        return f"/subscriptions/{self._config.subscription_id}"

    @staticmethod
    def _date_range(lookback_days: int) -> tuple[str, str]:
        """Return ISO-8601 start/end strings for the last *lookback_days* days."""
        end = datetime.now(tz=timezone.utc).date()
        start = end - timedelta(days=lookback_days)
        return start.isoformat(), end.isoformat()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_daily_costs(self, lookback_days: int = 30) -> List[Dict[str, Any]]:
        """Fetch daily aggregated costs for the subscription.

        Args:
            lookback_days: Number of days to look back from today.

        Returns:
            List of dicts with keys ``date``, ``cost``, and ``currency``.

        Raises:
            RuntimeError: On Azure API errors.
        """
        start, end = self._date_range(lookback_days)
        logger.debug("Querying daily costs from %s to %s", start, end)

        query = QueryDefinition(
            type=ExportType.ACTUAL_COST,
            timeframe=TimeframeType.CUSTOM,
            time_period=QueryTimePeriod(
                from_property=datetime.fromisoformat(start),
                to=datetime.fromisoformat(end),
            ),
            dataset=QueryDataset(
                granularity=GranularityType.DAILY,
                aggregation={
                    "totalCost": QueryAggregation(
                        name="PreTaxCost", function="Sum"
                    )
                },
            ),
        )

        try:
            result = self._client.query.usage(scope=self._scope, parameters=query)
        except AzureError as exc:
            logger.error("Azure API error while fetching daily costs: %s", exc)
            raise RuntimeError(f"Failed to fetch daily costs: {exc}") from exc

        rows = result.rows or []
        columns = [col.name for col in (result.columns or [])]
        logger.info("Retrieved %d daily cost records", len(rows))
        return self._rows_to_dicts(rows, columns)

    def get_cost_by_service(self, lookback_days: int = 30) -> List[Dict[str, Any]]:
        """Fetch costs grouped by Azure service name.

        Args:
            lookback_days: Number of days to look back from today.

        Returns:
            List of dicts with keys ``service``, ``cost``, and ``currency``.

        Raises:
            RuntimeError: On Azure API errors.
        """
        start, end = self._date_range(lookback_days)
        logger.debug("Querying costs by service from %s to %s", start, end)

        query = QueryDefinition(
            type=ExportType.ACTUAL_COST,
            timeframe=TimeframeType.CUSTOM,
            time_period=QueryTimePeriod(
                from_property=datetime.fromisoformat(start),
                to=datetime.fromisoformat(end),
            ),
            dataset=QueryDataset(
                granularity=None,
                aggregation={
                    "totalCost": QueryAggregation(
                        name="PreTaxCost", function="Sum"
                    )
                },
                grouping=[
                    QueryGrouping(type="Dimension", name="ServiceName")
                ],
            ),
        )

        try:
            result = self._client.query.usage(scope=self._scope, parameters=query)
        except AzureError as exc:
            logger.error("Azure API error while fetching costs by service: %s", exc)
            raise RuntimeError(f"Failed to fetch costs by service: {exc}") from exc

        rows = result.rows or []
        columns = [col.name for col in (result.columns or [])]
        logger.info("Retrieved costs for %d services", len(rows))
        return self._rows_to_dicts(rows, columns)

    def get_cost_by_resource_group(
        self, lookback_days: int = 30
    ) -> List[Dict[str, Any]]:
        """Fetch costs grouped by Azure resource group.

        Args:
            lookback_days: Number of days to look back from today.

        Returns:
            List of dicts with keys ``resource_group``, ``cost``, and
            ``currency``.

        Raises:
            RuntimeError: On Azure API errors.
        """
        start, end = self._date_range(lookback_days)
        logger.debug("Querying costs by resource group from %s to %s", start, end)

        query = QueryDefinition(
            type=ExportType.ACTUAL_COST,
            timeframe=TimeframeType.CUSTOM,
            time_period=QueryTimePeriod(
                from_property=datetime.fromisoformat(start),
                to=datetime.fromisoformat(end),
            ),
            dataset=QueryDataset(
                granularity=None,
                aggregation={
                    "totalCost": QueryAggregation(
                        name="PreTaxCost", function="Sum"
                    )
                },
                grouping=[
                    QueryGrouping(type="Dimension", name="ResourceGroupName")
                ],
            ),
        )

        try:
            result = self._client.query.usage(scope=self._scope, parameters=query)
        except AzureError as exc:
            logger.error(
                "Azure API error while fetching costs by resource group: %s", exc
            )
            raise RuntimeError(
                f"Failed to fetch costs by resource group: {exc}"
            ) from exc

        rows = result.rows or []
        columns = [col.name for col in (result.columns or [])]
        logger.info("Retrieved costs for %d resource groups", len(rows))
        return self._rows_to_dicts(rows, columns)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _rows_to_dicts(
        rows: List[Any], columns: List[str]
    ) -> List[Dict[str, Any]]:
        """Convert raw API row data to a list of dictionaries.

        Args:
            rows: Row data from the Azure Cost Management API response.
            columns: Column name list matching the order of values in each row.

        Returns:
            List of dicts keyed by column name.
        """
        return [dict(zip(columns, row)) for row in rows]
