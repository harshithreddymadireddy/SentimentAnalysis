from transformers import pipeline
import torch
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        try:
            logger.info("Loading sentiment analysis model...")
            self.analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Sentiment analysis model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise
    
    def analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a list of texts
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of dictionaries with sentiment results
        """
        try:
            if not texts:
                return []
            
            # Run sentiment analysis
            raw_results = self.analyzer(texts)
            
            # Process results
            processed_results = []
            for i, result in enumerate(raw_results):
                processed_result = {
                    "sentiment": result["label"],
                    "confidence": round(result["score"], 4),
                    "label_scores": {
                        result["label"]: round(result["score"], 4)
                    }
                }
                processed_results.append(processed_result)
            
            logger.info(f"Successfully analyzed {len(texts)} statements")
            return processed_results
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            raise

# Create global instance
analyzer_instance = SentimentAnalyzer()
