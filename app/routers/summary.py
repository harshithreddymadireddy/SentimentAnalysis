from fastapi import APIRouter, HTTPException
from app.models import BatchStatements, SentimentSummary, HealthCheck
from app.services.sentiment_service import analyzer_instance
from app.services.summary_service import summarizer_instance
from app.utils import log_api_call, calculate_average_confidence, count_sentiments

router = APIRouter()

@router.post("/summarize", response_model=SentimentSummary)
async def generate_insight_summary(batch: BatchStatements):
    """
    Generate insight summary from sentiment analysis of statements
    
    - **statements**: List of text statements to analyze and summarize
    - Returns comprehensive summary with statistics and insights
    """
    try:
        log_api_call("/insight/summarize", len(batch.statements))
        
        statements = [s.text for s in batch.statements]
        
        # Perform sentiment analysis
        sentiment_results = analyzer_instance.analyze(statements)
        
        # Count sentiments
        sentiment_counts = count_sentiments(sentiment_results)
        
        # Calculate average confidence
        avg_confidence = calculate_average_confidence(sentiment_results)
        
        # Generate natural language summary
        summary_text, top_sentiments = summarizer_instance.generate_summary(
            statements, sentiment_results
        )
        
        return SentimentSummary(
            total_statements=len(statements),
            positive_count=sentiment_counts["POSITIVE"],
            negative_count=sentiment_counts["NEGATIVE"],
            neutral_count=sentiment_counts["NEUTRAL"],
            summary=summary_text,
            top_sentiments=top_sentiments,
            average_confidence=avg_confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

@router.get("/health", response_model=HealthCheck)
async def summary_health_check():
    """Check if summary generation service is working"""
    try:
        # Test with simple statements
        test_statements = ["This is good", "This is bad"]
        test_sentiments = [
            {"sentiment": "POSITIVE", "confidence": 0.9},
            {"sentiment": "NEGATIVE", "confidence": 0.8}
        ]
        test_summary = summarizer_instance.generate_summary(test_statements, test_sentiments)
        
        return HealthCheck(
            status="healthy",
            message="Summary generation service is operational"
        )
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Summary service unavailable: {str(e)}"
        )
