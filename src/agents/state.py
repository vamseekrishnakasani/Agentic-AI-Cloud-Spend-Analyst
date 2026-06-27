"""LangGraph agent shared state definition."""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from src.processing.anomaly_detector import AnomalyReport


class AgentState(TypedDict):
    """Shared state passed between LangGraph nodes.

    Attributes:
        daily_costs: Raw daily cost records from Azure.
        service_costs: Raw service-level cost records from Azure.
        resource_group_costs: Raw resource-group cost records from Azure.
        anomaly_report: Result of anomaly detection on daily costs.
        service_anomalies: Service-level anomaly records.
        recommendations: List of optimisation recommendations.
        messages: Conversation history used by the LLM node.
        error: Non-empty string when a node encounters a fatal error.
    """

    daily_costs: List[Dict[str, Any]]
    service_costs: List[Dict[str, Any]]
    resource_group_costs: List[Dict[str, Any]]
    anomaly_report: Optional[AnomalyReport]
    service_anomalies: List[Dict[str, Any]]
    recommendations: List[str]
    messages: List[Any]
    error: str
