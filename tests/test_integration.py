"""FastAPI integration tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app
from tests.conftest import SAMPLE_NDA_TEXT


def test_upload_real_nda_pdf_returns_risk_json(pdf_factory) -> None:
    """Upload a generated NDA PDF and verify the live API risk response."""
    from main import sessions, users
    import secrets
    test_token = secrets.token_urlsafe(16)
    sessions[test_token] = "test@example.com"
    users["test@example.com"] = {"name": "Test", "email": "test@example.com", "password": "pass", "verified": True, "otp": "", "otp_expires_at": ""}
    client = TestClient(app)
    pdf_path = pdf_factory(SAMPLE_NDA_TEXT)
    with pdf_path.open("rb") as pdf_file:
        response = client.post(
            "/review",
            files={"file": ("sample-nda.pdf", pdf_file, "application/pdf")},
            headers={"Authorization": f"Bearer {test_token}"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["risk_score"] > 0
    assert payload["flags"]
    assert {"clause_name", "risk_level", "plain_english_explanation"}.issubset(
        payload["flags"][0]
    )


def test_metrics_endpoint_returns_aggregate_data() -> None:
    """Metrics endpoint returns dashboard-friendly counters."""
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert "total_contracts_reviewed" in payload
    assert "average_risk_score" in payload
