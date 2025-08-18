from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_insight_summarize_basic():
    """Test basic insight summary generation"""
    response = client.post(
        "/insight/summarize",
        json={
            "statements": [
                {"text": "I absolutely love this product!"},
                {"text": "This is the worst experience ever."}
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "total_statements" in data
    assert "positive_count" in data
    assert "negative_count" in data
    assert "top_sentiments" in data
    assert data["total_statements"] == 2

def test_insight_summarize_all_positive():
    """Test insight summary with all positive statements"""
    response = client.post(
        "/insight/summarize",
        json={
            "statements": [
                {"text": "This is amazing!"},
                {"text": "I love this so much!"},
                {"text": "Fantastic experience!"}
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_statements"] == 3
    assert data["positive_count"] >= 0
    assert "POSITIVE" in data["top_sentiments"]

def test_insight_summarize_mixed_sentiments():
    """Test insight summary with mixed sentiments"""
    response = client.post(
        "/insight/summarize",
        json={
            "statements": [
                {"text": "Great product quality!"},
                {"text": "Poor customer service."},
                {"text": "Amazing design!"},
                {"text": "Too expensive for the value."}
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_statements"] == 4
    assert isinstance(data["average_confidence"], float)
    assert 0 <= data["average_confidence"] <= 1

def test_summary_health_check():
    """Test summary service health check"""
    response = client.get("/insight/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_insight_single_statement():
    """Test insight generation with single statement"""
    response = client.post(
        "/insight/summarize",
        json={"statements": [{"text": "This is a test statement."}]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_statements"] == 1
    assert len(data["summary"]) > 0
