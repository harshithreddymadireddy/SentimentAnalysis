from pydantic import BaseModel
from typing import List, Optional

class Statement(BaseModel):
    text: str
    author: Optional[str] = None
    platform: Optional[str] = None

class BatchStatements(BaseModel):
    statements: List[Statement]

class SentimentResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    label_scores: Optional[dict] = None

class SentimentSummary(BaseModel):
    total_statements: int
    positive_count: int
    negative_count: int
    neutral_count: int
    summary: str
    top_sentiments: List[str]
    average_confidence: float

class HealthCheck(BaseModel):
    status: str
    message: str
