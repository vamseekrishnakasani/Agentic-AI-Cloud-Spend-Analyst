"""Statistical anomaly detection for Azure cost data."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.azure_cost_analyst.config import setup_logging

logger = setup_logging(__name__)


@dataclass
class CostAnomaly:
    """A detected cost anomaly.

    Attributes:
        date: ISO-8601 date string for the anomalous data point.
        cost: Actual cost value on that date.
        expected_cost: Baseline expected cost (rolling mean).
        z_score: Number of standard deviations from the mean.
        deviation_pct: Percentage deviation from the expected cost.
        is_spike: ``True`` when the cost exceeds the expected amount.
    """

    date: str
    cost: float
    expected_cost: float
    z_score: float
    deviation_pct: float
    is_spike: bool


@dataclass
class AnomalyReport:
    """Aggregated anomaly detection results.

    Attributes:
        anomalies: List of detected :class:`CostAnomaly` instances.
        mean_cost: Mean daily cost over the analysis window.
        std_cost: Standard deviation of daily costs.
        threshold: Z-score threshold used for detection.
        total_records: Total number of data points analysed.
    """

    anomalies: List[CostAnomaly] = field(default_factory=list)
    mean_cost: float = 0.0
    std_cost: float = 0.0
    threshold: float = 2.5
    total_records: int = 0

    @property
    def has_anomalies(self) -> bool:
        """``True`` when at least one anomaly was found."""
        return len(self.anomalies) > 0

    @property
    def anomaly_count(self) -> int:
        """Number of detected anomalies."""
        return len(self.anomalies)


class AnomalyDetector:
    """Detect cost anomalies in daily Azure spend data using z-score analysis.

    The detector computes a rolling mean and standard deviation over a
    configurable window.  Any data point whose z-score exceeds *threshold*
    is flagged as an anomaly.

    Args:
        threshold: Z-score threshold.  Data points with ``|z| > threshold``
            are classified as anomalies (default: ``2.5``).
        min_data_points: Minimum number of data points required before
            anomaly detection is attempted (default: ``7``).
    """

    def __init__(
        self,
        threshold: float = 2.5,
        min_data_points: int = 7,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be a positive number")
        if min_data_points < 2:
            raise ValueError("min_data_points must be at least 2")
        self.threshold = threshold
        self.min_data_points = min_data_points
        logger.info(
            "AnomalyDetector initialised (threshold=%.2f, min_data_points=%d)",
            threshold,
            min_data_points,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, daily_costs: List[Dict[str, Any]]) -> AnomalyReport:
        """Analyse a list of daily cost records and return an anomaly report.

        Each record in *daily_costs* must contain at least:

        * ``"cost"`` (numeric) – the daily spend amount.
        * ``"date"`` (str) – an ISO-8601 date string.

        Args:
            daily_costs: List of dicts returned by
                :meth:`~src.azure_cost_analyst.api.cost_client.AzureCostClient.get_daily_costs`.

        Returns:
            :class:`AnomalyReport` with all detected anomalies and summary
            statistics.

        Raises:
            ValueError: When *daily_costs* is empty or records are missing
                required keys.
        """
        if not daily_costs:
            raise ValueError("daily_costs must not be empty")

        costs = self._extract_costs(daily_costs)
        n = len(costs)

        report = AnomalyReport(
            mean_cost=float(np.mean(costs)),
            std_cost=float(np.std(costs, ddof=1)) if n > 1 else 0.0,
            threshold=self.threshold,
            total_records=n,
        )

        if n < self.min_data_points:
            logger.warning(
                "Only %d data points available; need at least %d for reliable "
                "anomaly detection. Returning empty report.",
                n,
                self.min_data_points,
            )
            return report

        z_scores = self._compute_z_scores(costs)

        for idx, (record, z) in enumerate(zip(daily_costs, z_scores)):
            if abs(z) > self.threshold:
                anomaly = CostAnomaly(
                    date=str(record.get("date", record.get("Date", idx))),
                    cost=float(costs[idx]),
                    expected_cost=report.mean_cost,
                    z_score=float(z),
                    deviation_pct=self._deviation_pct(
                        float(costs[idx]), report.mean_cost
                    ),
                    is_spike=float(costs[idx]) > report.mean_cost,
                )
                report.anomalies.append(anomaly)
                logger.debug(
                    "Anomaly detected on %s: cost=%.2f z=%.2f",
                    anomaly.date,
                    anomaly.cost,
                    anomaly.z_score,
                )

        logger.info(
            "Detection complete: %d/%d records flagged as anomalies",
            report.anomaly_count,
            n,
        )
        return report

    def detect_by_service(
        self, service_costs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify services whose cost deviates significantly from the mean.

        Computes z-scores across all services and flags those that exceed
        *threshold*.

        Args:
            service_costs: List of dicts with ``"service"`` (or ``"ServiceName"``)
                and ``"cost"`` (or ``"PreTaxCost"``) keys.

        Returns:
            List of service cost dicts that are flagged as anomalous, each
            augmented with a ``"z_score"`` key.

        Raises:
            ValueError: When *service_costs* is empty.
        """
        if not service_costs:
            raise ValueError("service_costs must not be empty")

        costs = self._extract_costs(service_costs)
        z_scores = self._compute_z_scores(costs)
        flagged = []

        for record, z in zip(service_costs, z_scores):
            if abs(z) > self.threshold:
                enriched = {**record, "z_score": float(z)}
                flagged.append(enriched)
                logger.debug(
                    "Service anomaly: %s z=%.2f",
                    record.get("service", record.get("ServiceName", "unknown")),
                    z,
                )

        logger.info(
            "%d/%d services flagged as anomalous",
            len(flagged),
            len(service_costs),
        )
        return flagged

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_costs(records: List[Dict[str, Any]]) -> List[float]:
        """Extract numeric cost values from a list of dicts.

        Accepts records using either ``"cost"`` or ``"PreTaxCost"`` as the
        cost key.

        Raises:
            ValueError: When a record contains no recognised cost key.
        """
        result = []
        for i, rec in enumerate(records):
            raw = (
                rec["cost"] if "cost" in rec
                else rec.get("PreTaxCost") if "PreTaxCost" in rec
                else rec.get("totalCost")
            )
            if raw is None:
                raise ValueError(
                    f"Record at index {i} has no recognised cost key "
                    f"('cost', 'PreTaxCost', or 'totalCost'). Keys found: "
                    f"{list(rec.keys())}"
                )
            result.append(float(raw))
        return result

    @staticmethod
    def _compute_z_scores(costs: List[float]) -> List[float]:
        """Return z-scores for each value in *costs*.

        When the standard deviation is zero (all values identical) all
        z-scores are returned as ``0.0``.
        """
        arr = np.array(costs, dtype=float)
        mean = arr.mean()
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        if std == 0.0:
            return [0.0] * len(costs)
        return list((arr - mean) / std)

    @staticmethod
    def _deviation_pct(actual: float, expected: float) -> float:
        """Return the percentage deviation of *actual* from *expected*."""
        if expected == 0:
            return 0.0
        return round((actual - expected) / expected * 100, 2)
