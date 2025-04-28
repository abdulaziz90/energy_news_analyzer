from fastapi import FastAPI, HTTPException, Depends
import logging
from typing import List

from .models import NewsQuery, AnalysisResponse, ArticleAnalysis, SentimentResult
from .services import news_fetcher, analysis
from .utils.config import NEWS_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Energy Market News Analyzer API",
    description="Fetches energy market news, performs sentiment analysis, summarization, and impact assessment using LLMs.",
    version="0.1.0"
)

# Dependency to check for API key
def check_api_key():
    if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_NEWS_API_KEY_HERE":
        logger.error("NewsAPI key not configured.")
        raise HTTPException(
            status_code=503,
            detail="Server configuration error: NewsAPI key not set. Please configure the .env file."
        )
    # Check if models loaded correctly
    if not analysis.sentiment_pipeline or not analysis.summarizer_pipeline or not analysis.impact_llm:
         logger.error("One or more analysis models failed to load.")
         raise HTTPException(
            status_code=503,
            detail="Server configuration error: Analysis models could not be loaded. Check server logs."
        )

@app.post("/analyze", response_model=AnalysisResponse, dependencies=[Depends(check_api_key)])
async def analyze_news(query: NewsQuery):
    """Analyzes news articles based on the provided query.

    Fetches news articles using NewsAPI.org based on the query parameters.
    For each article, it performs:
    - Sentiment Analysis
    - Summarization
    - Potential Impact Assessment (experimental)
    """
    logger.info(f"Received analysis request for query: 
{query.model_dump_json(indent=2)}")

    articles = news_fetcher.fetch_news(
        query=query.query,
        from_date=query.from_date,
        to_date=query.to_date,
        language=query.language,
        sort_by=query.sort_by,
        page_size=query.page_size
    )

    if articles is None:
        # Error logged in fetch_news, return appropriate response
        return AnalysisResponse(
            query=query.query,
            status="error",
            total_articles_processed=0,
            results=[],
            overall_error="Failed to fetch news. Check NewsAPI key or service status."
        )

    analysis_results: List[ArticleAnalysis] = []
    processed_count = 0

    for article in articles:
        processed_count += 1
        article_data = ArticleAnalysis(
            title=article.get("title"),
            url=article.get("url"),
            publishedAt=article.get("publishedAt"),
            # Use content if available, otherwise fallback to description
            content=article.get("content") or article.get("description")
        )

        if not article_data.content:
            article_data.error = "Article content or description is missing."
            analysis_results.append(article_data)
            continue # Skip analysis if no text

        try:
            # 1. Sentiment Analysis
            sentiment_result = analysis.analyze_sentiment(article_data.content)
            if sentiment_result:
                article_data.sentiment = SentimentResult(**sentiment_result)

            # 2. Summarization
            summary = analysis.summarize_text(article_data.content)
            if summary:
                article_data.summary = summary

            # 3. Impact Assessment
            impact = analysis.assess_impact(article_data.summary or article_data.content, context=query.query)
            if impact:
                article_data.impact_assessment = impact

        except Exception as e:
            logger.error(f"Error analyzing article 
{article.get("title")}: {e}", exc_info=True)
            article_data.error = f"Analysis failed: {e}"

        analysis_results.append(article_data)

    logger.info(f"Finished analysis for query: 
{query.query}. Processed {processed_count} articles.")

    return AnalysisResponse(
        query=query.query,
        status="success",
        total_articles_processed=processed_count,
        results=analysis_results
    )

@app.get("/health")
async def health_check():
    """Basic health check endpoint."
    # Check if models are loaded as a basic health indicator
    model_status = {
        "sentiment_model_loaded": analysis.sentiment_pipeline is not None,
        "summarizer_model_loaded": analysis.summarizer_pipeline is not None,
        "impact_llm_loaded": analysis.impact_llm is not None,
        "news_api_key_configured": NEWS_API_KEY is not None and NEWS_API_KEY != "YOUR_NEWS_API_KEY_HERE"
    }
    if all(model_status.values()):
        return {"status": "ok", "details": model_status}
    else:
        # Return 503 if any critical component is missing
        raise HTTPException(
            status_code=503,
            detail={"status": "error", "details": model_status}
        )

# To run the API locally: uvicorn api.main:app --reload --port 8000
# Ensure you have a .env file in the project root with your NEWS_API_KEY

