"""Configuration and logging setup for Azure Cost Analyst."""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger with a standard format.

    Args:
        name: Logger name, typically the module ``__name__``.
        level: Logging level (default: ``logging.INFO``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


@dataclass
class AzureConfig:
    """Azure service principal credentials and scope configuration.

    Attributes:
        tenant_id: Azure Active Directory tenant ID.
        client_id: Service principal application (client) ID.
        client_secret: Service principal client secret.
        subscription_id: Azure subscription to analyse.
    """

    tenant_id: str = field(default_factory=lambda: os.environ.get("AZURE_TENANT_ID", ""))
    client_id: str = field(default_factory=lambda: os.environ.get("AZURE_CLIENT_ID", ""))
    client_secret: str = field(
        default_factory=lambda: os.environ.get("AZURE_CLIENT_SECRET", "")
    )
    subscription_id: str = field(
        default_factory=lambda: os.environ.get("AZURE_SUBSCRIPTION_ID", "")
    )

    def validate(self) -> None:
        """Raise :class:`ValueError` if any required field is missing."""
        missing = [
            field_name
            for field_name, value in {
                "AZURE_TENANT_ID": self.tenant_id,
                "AZURE_CLIENT_ID": self.client_id,
                "AZURE_CLIENT_SECRET": self.client_secret,
                "AZURE_SUBSCRIPTION_ID": self.subscription_id,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                f"Missing required Azure configuration fields: {', '.join(missing)}"
            )


@dataclass
class AppConfig:
    """Top-level application configuration.

    Attributes:
        azure: Azure credentials configuration.
        anomaly_threshold: Z-score threshold for anomaly classification.
        lookback_days: Number of past days to include in cost queries.
        openai_api_key: Optional OpenAI API key used by LangGraph agents.
    """

    azure: AzureConfig = field(default_factory=AzureConfig)
    anomaly_threshold: float = field(
        default_factory=lambda: float(os.environ.get("ANOMALY_THRESHOLD", "2.5"))
    )
    lookback_days: int = field(
        default_factory=lambda: int(os.environ.get("LOOKBACK_DAYS", "30"))
    )
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
    )
