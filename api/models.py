from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class NewsQuery(BaseModel):
    query: str = Field(..., description="Search query for news articles (keywords or phrases).")
    from_date: Optional[str] = Field(None, description="Start date for articles (YYYY-MM-DD).", pattern=r"^\d{4}-\d{2}-\d{2}$")
    to_date: Optional[str] = Field(None, description="End date for articles (YYYY-MM-DD).", pattern=r"^\d{4}-\d{2}-\d{2}$")
    language: str = Field("en", description="The 2-letter ISO-639-1 code of the language.")
    sort_by: str = Field("publishedAt", description="Order to sort articles (relevancy, popularity, publishedAt).")
    page_size: int = Field(20, description="Number of results to return (max 100).", ge=1, le=100)

class SentimentResult(BaseModel):
    label: str
    score: float

class ArticleAnalysis(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    publishedAt: Optional[str] = None
    content: Optional[str] = None # Or description if content is null
    sentiment: Optional[SentimentResult] = None
    summary: Optional[str] = None
    impact_assessment: Optional[str] = None
    error: Optional[str] = None # To capture errors during analysis of a specific article

class AnalysisResponse(BaseModel):
    query: str
    status: str
    total_articles_processed: int
    results: List[ArticleAnalysis]
    overall_error: Optional[str] = None # For errors like API key missing

