"""LangGraph-based agent orchestrator for Azure cost analysis."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.azure_cost_analyst.agents.state import AgentState
from src.azure_cost_analyst.processing.anomaly_detector import AnomalyDetector, AnomalyReport
from src.azure_cost_analyst.api.cost_client import AzureCostClient
from src.azure_cost_analyst.config import AppConfig, setup_logging

logger = setup_logging(__name__)


@dataclass
class OrchestratorResult:
    """Final output produced by :class:`CostAnalystOrchestrator`.

    Attributes:
        anomaly_report: Anomaly detection summary for daily costs.
        service_anomalies: Anomalous services with z-scores.
        recommendations: Natural-language optimisation recommendations.
        error: Non-empty when the run encountered a fatal error.
    """

    anomaly_report: Optional[AnomalyReport] = None
    service_anomalies: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def succeeded(self) -> bool:
        """``True`` when the orchestration run completed without errors."""
        return not self.error


class CostAnalystOrchestrator:
    """LangGraph agent orchestrator for Azure cost analysis.

    The orchestrator wires together three agents via a directed state graph:

    1. **data_collection** – fetches cost data from Azure.
    2. **anomaly_detection** – runs statistical anomaly detection.
    3. **recommendation** – generates optimisation recommendations via an LLM.

    Args:
        config: Application configuration.  When *None* an :class:`AppConfig`
            is built from environment variables.
        cost_client: Pre-built :class:`AzureCostClient`.  Useful for testing.
        anomaly_detector: Pre-built :class:`AnomalyDetector`.  Useful for
            testing.
    """

    _SYSTEM_PROMPT = (
        "You are an expert Azure cloud cost optimisation analyst. "
        "Analyse the provided cost anomalies and produce a concise, "
        "actionable list of cost-reduction recommendations.  "
        "Be specific – reference service names, percentages, and concrete "
        "next steps where possible."
    )

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        cost_client: Optional[AzureCostClient] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
    ) -> None:
        self._config = config or AppConfig()
        self._cost_client = cost_client
        self._anomaly_detector = anomaly_detector or AnomalyDetector(
            threshold=self._config.anomaly_threshold
        )
        self._llm = self._build_llm()
        self._graph = self._build_graph()
        logger.info("CostAnalystOrchestrator initialised")

    # ------------------------------------------------------------------
    # LLM construction
    # ------------------------------------------------------------------

    def _build_llm(self) -> Optional[ChatOpenAI]:
        if not self._config.openai_api_key:
            logger.warning(
                "OPENAI_API_KEY not set; LLM-based recommendations will be skipped"
            )
            return None
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=self._config.openai_api_key,
            temperature=0,
        )

    # ------------------------------------------------------------------
    # LangGraph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("data_collection", self._data_collection_node)
        graph.add_node("anomaly_detection", self._anomaly_detection_node)
        graph.add_node("recommendation", self._recommendation_node)

        graph.set_entry_point("data_collection")
        graph.add_conditional_edges(
            "data_collection",
            self._route_after_collection,
            {"anomaly_detection": "anomaly_detection", "end": END},
        )
        graph.add_edge("anomaly_detection", "recommendation")
        graph.add_edge("recommendation", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _data_collection_node(self, state: AgentState) -> AgentState:
        """Fetch cost data from Azure Cost Management API."""
        logger.info("Node: data_collection – fetching Azure cost data")

        if self._cost_client is None:
            try:
                self._cost_client = AzureCostClient(self._config.azure)
            except (ValueError, Exception) as exc:
                logger.error("Failed to build AzureCostClient: %s", exc)
                state["error"] = str(exc)
                return state

        lookback = self._config.lookback_days
        try:
            state["daily_costs"] = self._cost_client.get_daily_costs(lookback)
            state["service_costs"] = self._cost_client.get_cost_by_service(lookback)
            state["resource_group_costs"] = (
                self._cost_client.get_cost_by_resource_group(lookback)
            )
            logger.info(
                "Collected %d daily, %d service, %d resource-group records",
                len(state["daily_costs"]),
                len(state["service_costs"]),
                len(state["resource_group_costs"]),
            )
        except RuntimeError as exc:
            logger.error("Data collection failed: %s", exc)
            state["error"] = str(exc)

        return state

    def _anomaly_detection_node(self, state: AgentState) -> AgentState:
        """Run anomaly detection on collected cost data."""
        logger.info("Node: anomaly_detection")

        try:
            state["anomaly_report"] = self._anomaly_detector.detect(
                state["daily_costs"]
            )
            state["service_anomalies"] = self._anomaly_detector.detect_by_service(
                state["service_costs"]
            )
            logger.info(
                "Anomaly detection complete: %d daily anomalies, %d service anomalies",
                state["anomaly_report"].anomaly_count,
                len(state["service_anomalies"]),
            )
        except (ValueError, Exception) as exc:
            logger.error("Anomaly detection failed: %s", exc)
            state["error"] = str(exc)

        return state

    def _recommendation_node(self, state: AgentState) -> AgentState:
        """Generate cost optimisation recommendations using the LLM."""
        logger.info("Node: recommendation")

        if state.get("error"):
            return state

        report = state.get("anomaly_report")
        service_anomalies = state.get("service_anomalies", [])

        if self._llm is None:
            state["recommendations"] = self._fallback_recommendations(
                report, service_anomalies
            )
            return state

        summary = self._build_anomaly_summary(report, service_anomalies)
        messages = [
            SystemMessage(content=self._SYSTEM_PROMPT),
            HumanMessage(content=summary),
        ]

        try:
            response: AIMessage = self._llm.invoke(messages)
            raw = response.content or ""
            state["recommendations"] = [
                line.strip()
                for line in raw.splitlines()
                if line.strip()
            ]
            state["messages"] = list(messages) + [response]
            logger.info(
                "LLM returned %d recommendation lines",
                len(state["recommendations"]),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM call failed: %s", exc)
            state["recommendations"] = self._fallback_recommendations(
                report, service_anomalies
            )

        return state

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    @staticmethod
    def _route_after_collection(state: AgentState) -> str:
        if state.get("error") or not state.get("daily_costs"):
            logger.warning("Skipping anomaly detection due to missing data or error")
            return "end"
        return "anomaly_detection"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_anomaly_summary(
        report: Optional[AnomalyReport],
        service_anomalies: List[Dict[str, Any]],
    ) -> str:
        lines = ["Azure Cost Anomaly Summary\n"]
        if report:
            lines.append(
                f"Daily costs – mean: ${report.mean_cost:.2f}, "
                f"std: ${report.std_cost:.2f}, "
                f"anomalies detected: {report.anomaly_count}"
            )
            for a in report.anomalies:
                lines.append(
                    f"  • {a.date}: ${a.cost:.2f} "
                    f"(z={a.z_score:.2f}, {'+' if a.is_spike else ''}"
                    f"{a.deviation_pct:.1f}% vs expected ${a.expected_cost:.2f})"
                )
        if service_anomalies:
            lines.append("\nAnomalous services:")
            for svc in service_anomalies:
                name = svc.get("service") or svc.get("ServiceName", "unknown")
                cost = svc.get("cost") or svc.get("PreTaxCost", 0)
                z = svc.get("z_score", 0)
                lines.append(f"  • {name}: ${cost:.2f} (z={z:.2f})")
        return "\n".join(lines)

    @staticmethod
    def _fallback_recommendations(
        report: Optional[AnomalyReport],
        service_anomalies: List[Dict[str, Any]],
    ) -> List[str]:
        recs = []
        if report and report.has_anomalies:
            recs.append(
                f"Review {report.anomaly_count} daily cost anomaly(ies) detected "
                f"in the last analysis window."
            )
        for svc in service_anomalies:
            name = svc.get("service") or svc.get("ServiceName", "unknown")
            recs.append(
                f"Investigate unusual spend for service '{name}' "
                f"(z-score: {svc.get('z_score', 0):.2f})."
            )
        if not recs:
            recs.append("No significant cost anomalies detected in the analysis window.")
        return recs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> OrchestratorResult:
        """Execute the full cost analysis pipeline.

        Returns:
            :class:`OrchestratorResult` with anomaly data and recommendations.
        """
        logger.info("Starting cost analysis orchestration")
        initial_state: AgentState = {
            "daily_costs": [],
            "service_costs": [],
            "resource_group_costs": [],
            "anomaly_report": None,
            "service_anomalies": [],
            "recommendations": [],
            "messages": [],
            "error": "",
        }

        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as exc:  # noqa: BLE001
            logger.error("Orchestration graph error: %s", exc)
            return OrchestratorResult(error=str(exc))

        return OrchestratorResult(
            anomaly_report=final_state.get("anomaly_report"),
            service_anomalies=final_state.get("service_anomalies", []),
            recommendations=final_state.get("recommendations", []),
            error=final_state.get("error", ""),
        )
