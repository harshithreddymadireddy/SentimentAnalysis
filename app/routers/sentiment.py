from fastapi import APIRouter, HTTPException
from typing import List
from app.models import BatchStatements, SentimentResult, HealthCheck
from app.services.sentiment_service import analyzer_instance
from app.utils import log_api_call

router = APIRouter()

@router.post("/analyze", response_model=List[SentimentResult])
async def analyze_sentiment(batch: BatchStatements):
    """
    Analyze sentiment for a batch of statements
    
    - **statements**: List of text statements to analyze
    - Returns sentiment analysis results with confidence scores
    """
    try:
        log_api_call("/sentiment/analyze", len(batch.statements))
        
        statements = [s.text for s in batch.statements]
        results = analyzer_instance.analyze(statements)
        
        # Format results to include original text
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append(SentimentResult(
                text=statements[i],
                sentiment=result["sentiment"],
                confidence=result["confidence"],
                label_scores=result.get("label_scores", {})
            ))
        
        return formatted_results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@router.get("/health", response_model=HealthCheck)
async def sentiment_health_check():
    """Check if sentiment analysis service is working"""
    try:
        # Test with a simple statement
        test_result = analyzer_instance.analyze(["This is a test"])
        return HealthCheck(
            status="healthy",
            message="Sentiment analysis service is operational"
        )
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Sentiment service unavailable: {str(e)}"
        )
