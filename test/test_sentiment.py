from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_sentiment_analyze_single_statement():
    """Test sentiment analysis with single statement"""
    response = client.post(
        "/sentiment/analyze",
        json={"statements": [{"text": "I love this amazing product!"}]}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["sentiment"] in ["POSITIVE", "NEGATIVE"]
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 1

def test_sentiment_analyze_multiple_statements():
    """Test sentiment analysis with multiple statements"""
    response = client.post(
        "/sentiment/analyze",
        json={
            "statements": [
                {"text": "This is fantastic!"},
                {"text": "This is terrible."},
                {"text": "Neutral statement here."}
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    for result in data:
        assert "sentiment" in result
        assert "confidence" in result
        assert "text" in result

def test_sentiment_health_check():
    """Test sentiment service health check"""
    response = client.get("/sentiment/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_sentiment_empty_statements():
    """Test sentiment analysis with empty statements list"""
    response = client.post(
        "/sentiment/analyze",
        json={"statements": []}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 0
