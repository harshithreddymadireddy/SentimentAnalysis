import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_average_confidence(results: List[Dict[str, Any]]) -> float:
    """Calculate average confidence score from sentiment results"""
    if not results:
        return 0.0
    
    total_confidence = sum(result.get('confidence', 0.0) for result in results)
    return round(total_confidence / len(results), 3)

def count_sentiments(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count occurrences of each sentiment"""
    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    
    for result in results:
        sentiment = result.get('sentiment', '').upper()
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1
    
    return sentiment_counts

def format_confidence_percentage(confidence: float) -> str:
    """Format confidence as percentage string"""
    return f"{confidence * 100:.1f}%"

def log_api_call(endpoint: str, statement_count: int):
    """Log API call information"""
    logger.info(f"API call to {endpoint} with {statement_count} statements")
