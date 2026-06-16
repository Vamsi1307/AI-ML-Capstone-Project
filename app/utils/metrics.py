"""Logging and monitoring utilities."""

from typing import Dict, Any
from datetime import datetime

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Collects application metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, Any] = {
            "total_queries": 0,
            "total_documents_uploaded": 0,
            "total_chunks_created": 0,
            "errors": 0,
        }

    def record_query(self, success: bool = True):
        """Record a query."""
        self.metrics["total_queries"] += 1
        if not success:
            self.metrics["errors"] += 1

    def record_document_upload(self, chunk_count: int):
        """Record document upload."""
        self.metrics["total_documents_uploaded"] += 1
        self.metrics["total_chunks_created"] += chunk_count

    def record_error(self):
        """Record an error."""
        self.metrics["errors"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()

    def log_metrics(self):
        """Log current metrics."""
        logger.info("System metrics", metrics=self.metrics)


# Global metrics instance
metrics = MetricsCollector()
