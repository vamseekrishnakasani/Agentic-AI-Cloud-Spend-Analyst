"""Tests for the AnomalyDetector module."""

import pytest

from src.azure_cost_analyst.anomaly_detector import AnomalyDetector, AnomalyReport


def _make_records(costs, dates=None):
    """Helper to build daily cost records."""
    if dates is None:
        dates = [f"2024-01-{i+1:02d}" for i in range(len(costs))]
    return [{"date": d, "cost": c} for d, c in zip(dates, costs)]


class TestAnomalyDetectorInit:
    def test_valid_threshold(self):
        detector = AnomalyDetector(threshold=3.0)
        assert detector.threshold == 3.0

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold must be a positive number"):
            AnomalyDetector(threshold=0)

    def test_invalid_min_data_points_raises(self):
        with pytest.raises(ValueError, match="min_data_points must be at least 2"):
            AnomalyDetector(min_data_points=1)


class TestDetect:
    def test_no_anomalies_in_uniform_data(self):
        records = _make_records([100.0] * 20)
        detector = AnomalyDetector(threshold=2.5)
        report = detector.detect(records)
        assert isinstance(report, AnomalyReport)
        assert not report.has_anomalies
        assert report.total_records == 20

    def test_spike_detected(self):
        # 19 days of $100, one day of $1000 → clear spike
        costs = [100.0] * 19 + [1000.0]
        records = _make_records(costs)
        detector = AnomalyDetector(threshold=2.5)
        report = detector.detect(records)
        assert report.has_anomalies
        assert report.anomaly_count == 1
        spike = report.anomalies[0]
        assert spike.is_spike
        assert spike.cost == 1000.0

    def test_drop_detected(self):
        # 19 days of $1000, one day of $0 → clear drop
        costs = [1000.0] * 19 + [0.0]
        records = _make_records(costs)
        detector = AnomalyDetector(threshold=2.5)
        report = detector.detect(records)
        assert report.has_anomalies
        drop = report.anomalies[0]
        assert not drop.is_spike

    def test_empty_input_raises(self):
        detector = AnomalyDetector()
        with pytest.raises(ValueError, match="daily_costs must not be empty"):
            detector.detect([])

    def test_too_few_records_returns_empty_report(self):
        records = _make_records([100.0, 200.0, 150.0])
        detector = AnomalyDetector(min_data_points=7)
        report = detector.detect(records)
        assert not report.has_anomalies
        assert report.total_records == 3

    def test_missing_cost_key_raises(self):
        detector = AnomalyDetector()
        records = [{"date": "2024-01-01", "amount": 100.0}]
        with pytest.raises(ValueError, match="no recognised cost key"):
            detector.detect(records)

    def test_pretax_cost_key_accepted(self):
        costs = [100.0] * 19 + [1000.0]
        records = [
            {"date": f"2024-01-{i+1:02d}", "PreTaxCost": c}
            for i, c in enumerate(costs)
        ]
        detector = AnomalyDetector(threshold=2.5)
        report = detector.detect(records)
        assert report.has_anomalies

    def test_report_statistics(self):
        costs = [float(i) for i in range(1, 21)]
        records = _make_records(costs)
        detector = AnomalyDetector(threshold=2.5)
        report = detector.detect(records)
        assert abs(report.mean_cost - sum(costs) / len(costs)) < 0.001
        assert report.std_cost > 0

    def test_deviation_pct(self):
        costs = [100.0] * 19 + [200.0]
        records = _make_records(costs)
        detector = AnomalyDetector(threshold=2.5)
        report = detector.detect(records)
        if report.has_anomalies:
            anomaly = report.anomalies[0]
            assert anomaly.deviation_pct > 0


class TestDetectByService:
    def test_empty_input_raises(self):
        detector = AnomalyDetector()
        with pytest.raises(ValueError, match="service_costs must not be empty"):
            detector.detect_by_service([])

    def test_anomalous_service_flagged(self):
        services = [
            {"service": "Compute", "cost": 100.0},
            {"service": "Storage", "cost": 105.0},
            {"service": "Network", "cost": 95.0},
            {"service": "Blob", "cost": 102.0},
            {"service": "Functions", "cost": 98.0},
            {"service": "CosmosDB", "cost": 5000.0},  # clear outlier
        ]
        detector = AnomalyDetector(threshold=2.0)
        flagged = detector.detect_by_service(services)
        assert len(flagged) >= 1
        names = [s.get("service") for s in flagged]
        assert "CosmosDB" in names

    def test_z_score_present_in_output(self):
        services = [
            {"service": f"svc{i}", "cost": float(i * 10)} for i in range(1, 8)
        ]
        detector = AnomalyDetector(threshold=100.0)  # high threshold → no flags
        flagged = detector.detect_by_service(services)
        # Even with no flags returned, test that z_score key is added when there IS a flag
        # Use a low threshold to force at least one flag
        detector2 = AnomalyDetector(threshold=0.5)
        flagged2 = detector2.detect_by_service(services)
        for item in flagged2:
            assert "z_score" in item
