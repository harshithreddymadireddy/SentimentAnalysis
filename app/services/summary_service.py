from transformers import pipeline
import torch
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class InsightSummarizer:
    def __init__(self):
        try:
            # Load GPT-2 text generation model for NLG
            logger.info("Loading text generation model for insights...")
            self.generator = pipeline(
                "text-generation",
                model="gpt2",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Text generation model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load generation model: {e}")
            raise
    
    def generate_statistics(self, sentiments: List[Dict[str, Any]]) -> Tuple[int, int, int, float]:
        """Calculate sentiment counts and average confidence"""
        positive_count = sum(1 for s in sentiments if s["sentiment"] == "POSITIVE")
        negative_count = sum(1 for s in sentiments if s["sentiment"] == "NEGATIVE")
        total_count = len(sentiments)
        avg_confidence = round(sum(s["confidence"] for s in sentiments) / len(sentiments), 3) if sentiments else 0.0
        return total_count, positive_count, negative_count, avg_confidence
    
    def generate_nlg_summary(self, total: int, positive: int, negative: int, avg_confidence: float) -> str:
        if total == 0:
            return "No reviews available."
        
        positive_pct = round((positive / total) * 100, 1)
        negative_pct = round((negative / total) * 100, 1)
        
        prompt = (
            f"Analyze {total} influencer reviews: {positive} positive and {negative} negative.\n"
            f"Based on these percentages, provide a concise summary indicating if the product or service quality should be improved. "
            f"Use no additional information beyond these statistics.\n"
            "Summary:"
        )
        
        try:
            generated = self.generator(
                prompt,
                max_length=len(prompt.split()) + 30,
                num_return_sequences=1,
                do_sample=False,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=2.0
            )
            text = generated[0]['generated_text']
            summary = text.split("Summary:")[-1].strip()
            # Clean repeated phrases if any
            while "Based on these percentages" in summary:
                summary = summary.replace("Based on these percentages", "").strip()
            return summary or "The product quality appears satisfactory based on the current reviews."
        except Exception as e:
            logger.error(f"NLG summary generation error: {e}")
            if positive_pct >= 50:
                return "Product quality is generally positive; no urgent improvements needed."
            else:
                return "Product quality shows room for improvement based on current reviews."

    
    def generate_summary(self, statements: List[str], sentiments: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        try:
            total, positive, negative, avg_confidence = self.generate_statistics(sentiments)
            nlg_summary = self.generate_nlg_summary(total, positive, negative, avg_confidence)
            
            # Prepare clean statistics text
            stats_text = (
                f"Analyzed {total} influencer statements: "
                f"{positive} positive ({round((positive/total)*100 if total>0 else 0,1)}%), "
                f"{negative} negative ({round((negative/total)*100 if total>0 else 0,1)}%). "
                f"Average confidence {avg_confidence *100:.1f}%."
            )
            
            # Combine stats and NLG insight
            combined_summary = f"📊 STATISTICS:\n{stats_text}\n\n💡 INSIGHTS:\n{nlg_summary}"
            
            top_sentiments = []
            if positive > 0:
                top_sentiments.append("POSITIVE")
            if negative > 0:
                top_sentiments.append("NEGATIVE")
            
            return combined_summary, top_sentiments
        
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return "Unable to generate summary due to error.", ["ERROR"]

# Create global instance
summarizer_instance = InsightSummarizer()
