import requests
import logging
from ..utils.config import NEWS_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

def fetch_news(query: str, from_date: str | None = None, to_date: str | None = None, language: str = "en", sort_by: str = "publishedAt", page_size: int = 20) -> list | None:
    """Fetches news articles from NewsAPI.org based on a query.

    Args:
        query: The search query (keywords or phrases).
        from_date: Optional start date for articles (YYYY-MM-DD).
        to_date: Optional end date for articles (YYYY-MM-DD).
        language: The 2-letter ISO-639-1 code of the language (default: en).
        sort_by: The order to sort the articles in (relevancy, popularity, publishedAt).
        page_size: The number of results to return per page (default: 20, max: 100).

    Returns:
        A list of articles, or None if an error occurs or no key is found.
    """
    if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_NEWS_API_KEY_HERE":
        logger.error("NewsAPI key not found or not configured in .env file. Please obtain a key from https://newsapi.org/ and add it to your .env file.")
        return None

    params = {
        "q": query,
        "language": language,
        "sortBy": sort_by,
        "pageSize": min(page_size, 100), # Ensure page_size doesn't exceed max
        "apiKey": NEWS_API_KEY
    }
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date

    try:
        response = requests.get(NEWS_API_ENDPOINT, params=params)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()

        if data.get("status") == "ok":
            articles = data.get("articles", [])
            logger.info(f"Successfully fetched {len(articles)} articles for query: '{query}'")
            return articles
        else:
            logger.error(f"NewsAPI error: {data.get('message')}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching news from NewsAPI: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during news fetching: {e}")
        return None

# Example usage (for testing)
if __name__ == "__main__":
    # Make sure to have a .env file with NEWS_API_KEY="your_key" in the root directory
    # when running this directly.
    import os
    from dotenv import load_dotenv
    # Adjust path to load .env from the project root if running from api/services
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)
    NEWS_API_KEY = os.getenv("NEWS_API_KEY") # Reload key after loading .env

    if NEWS_API_KEY and NEWS_API_KEY != "YOUR_NEWS_API_KEY_HERE":
        test_query = "renewable energy investment"
        articles = fetch_news(test_query, page_size=5)
        if articles:
            print(f"Fetched {len(articles)} articles for '{test_query}':")
            for i, article in enumerate(articles):
                print(f"  {i+1}. {article.get('title')}")
        else:
            print(f"Failed to fetch articles for '{test_query}'. Check logs and API key.")
    else:
        print("Skipping example usage: NEWS_API_KEY not configured in .env file.")

