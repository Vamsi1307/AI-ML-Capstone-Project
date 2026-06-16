"""Basic tests for the application."""

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


class TestHealthCheck:
    """Test health check endpoint."""

    def test_health_check_status(self):
        """Test health check returns 200."""
        response = client.get("/api/v1/health-check")
        assert response.status_code == 200

    def test_health_check_content(self):
        """Test health check response content."""
        response = client.get("/api/v1/health-check")
        data = response.json()
        assert data["status"] == "healthy"
        assert "message" in data
        assert "vector_count" in data


class TestRoot:
    """Test root endpoint."""

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "app" in data
        assert "version" in data


@pytest.mark.asyncio
async def test_invalid_query():
    """Test invalid query handling."""
    response = client.post(
        "/api/v1/ask-question",
        json={
            "query": "",  # Empty query
        },
    )
    # Should either reject or handle gracefully
    assert response.status_code in [400, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
