"""Entry point for the Agentic Azure Cloud Spend Analyst."""

import sys

from src.azure_cost_analyst.config import AppConfig, setup_logging
from src.azure_cost_analyst.agents.orchestrator import CostAnalystOrchestrator

logger = setup_logging(__name__)


def main() -> int:
    """Run the full cost analysis pipeline and print results.

    Returns:
        Exit code: ``0`` on success, ``1`` on error.
    """
    logger.info("Agentic Azure Cloud Spend Analyst starting")
    config = AppConfig()

    orchestrator = CostAnalystOrchestrator(config=config)
    result = orchestrator.run()

    if not result.succeeded:
        logger.error("Analysis failed: %s", result.error)
        return 1

    report = result.anomaly_report
    if report:
        print(f"\n=== Daily Cost Anomaly Report ===")
        print(f"  Records analysed : {report.total_records}")
        print(f"  Mean daily cost  : ${report.mean_cost:,.2f}")
        print(f"  Std deviation    : ${report.std_cost:,.2f}")
        print(f"  Anomalies found  : {report.anomaly_count}")

        if report.has_anomalies:
            print("\n  Anomalous dates:")
            for anomaly in report.anomalies:
                direction = "spike" if anomaly.is_spike else "drop"
                print(
                    f"    {anomaly.date}: ${anomaly.cost:,.2f} "
                    f"({direction}, {anomaly.deviation_pct:+.1f}%, "
                    f"z={anomaly.z_score:.2f})"
                )

    if result.service_anomalies:
        print("\n=== Anomalous Services ===")
        for svc in result.service_anomalies:
            name = svc.get("service") or svc.get("ServiceName", "unknown")
            cost = svc.get("cost") or svc.get("PreTaxCost", 0)
            print(f"  {name}: ${cost:,.2f} (z={svc.get('z_score', 0):.2f})")

    if result.recommendations:
        print("\n=== Recommendations ===")
        for rec in result.recommendations:
            print(f"  • {rec}")

    logger.info("Analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
