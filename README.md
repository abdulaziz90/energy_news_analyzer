# Energy Market News Analyzer API

## Overview

This project provides a Python-based API and accompanying Jupyter notebooks for fetching, analyzing, and understanding news related to the energy market. It leverages natural language processing (NLP) techniques using open-source models via Hugging Face and Langchain to perform:

*   **News Aggregation:** Fetches relevant news articles from NewsAPI.org.
*   **Sentiment Analysis:** Determines the sentiment (positive/negative) of news content.
*   **Summarization:** Generates concise summaries of articles.
*   **Impact Assessment (Experimental):** Provides a qualitative assessment of the potential impact of news on the energy market context.

This project is designed as a portfolio piece to showcase skills in API development, NLP, LLM integration (using Langchain), data processing, and project structuring. It specifically targets interests relevant to companies in the energy analytics and services sectors.

## Features

*   **RESTful API:** Built with FastAPI, providing endpoints for news analysis.
*   **News Fetching:** Integrates with NewsAPI.org to retrieve news based on keywords, date ranges, language, etc.
*   **NLP Analysis:**
    *   Sentiment Analysis using `distilbert-base-uncased-finetuned-sst-2-english`.
    *   Summarization using `facebook/bart-large-cnn`.
    *   Experimental Impact Assessment using a simple Langchain LLMChain with `gpt2`.
*   **Asynchronous Processing:** FastAPI allows for efficient handling of requests.
*   **Configuration Management:** Uses `.env` files for managing API keys.
*   **Educational Notebooks:** Includes Jupyter notebooks that walk through each step of the process (fetching, sentiment, summarization, impact assessment).
*   **Cost-Effective:** Prioritizes free tiers (NewsAPI) and open-source models to minimize operational costs.

## Tech Stack & Architecture

*   **Backend Framework:** FastAPI
*   **NLP/LLM Libraries:** Langchain, Hugging Face `transformers`, `torch`, `sentencepiece`
*   **Data Validation:** Pydantic
*   **API Client (News):** `requests`
*   **Environment Management:** `python-dotenv`
*   **Notebook Environment:** Jupyter Notebook / Lab, `ipykernel`
*   **Language:** Python 3.10+

**Core Components:**

1.  **`main.py`:** FastAPI application definition, API endpoints (`/analyze`, `/health`).
2.  **`services/news_fetcher.py`:** Module responsible for interacting with the NewsAPI.org.
3.  **`services/analysis.py`:** Module containing functions for sentiment analysis, summarization, and impact assessment using Hugging Face pipelines and Langchain.
4.  **`models.py`:** Pydantic models for API request/response validation.
5.  **`utils/config.py`:** Utility for loading configuration (e.g., API keys) from environment variables.
6.  **`notebooks/`:** Directory containing step-by-step Jupyter notebooks demonstrating the functionality.

## Project Structure

```
energy_news_analyzer/
├── api/
│   ├── services/
│   │   ├── __init__.py
│   │   ├── analysis.py       # Sentiment, Summarization, Impact Assessment logic
│   │   └── news_fetcher.py   # NewsAPI interaction logic
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py         # Environment variable loading
│   ├── __init__.py
│   ├── main.py             # FastAPI app definition and endpoints
│   ├── models.py           # Pydantic models for API
│   └── requirements.txt    # Python package dependencies
├── notebooks/
│   ├── data/               # Directory for storing intermediate data (created by notebooks)
│   │   ├── fetched_articles.json
│   │   ├── sentiment_results.json
│   │   ├── summary_results.json
│   │   └── impact_assessment_results.json
│   ├── 01_news_fetching.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   ├── 03_summarization.ipynb
│   └── 04_impact_assessment.ipynb
├── .env.example          # Example environment variables file
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd energy_news_analyzer
    ```

2.  **Create a Virtual Environment:** (Recommended)
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r api/requirements.txt
    ```
    *Note: Installing PyTorch (`torch`) and `transformers` might take some time and download significant data (including model weights).* 

4.  **Configure Environment Variables:**
    *   Copy the example `.env.example` file to `.env`:
        ```bash
        cp .env.example .env
        ```
    *   **Edit the `.env` file:**
        *   Obtain a free API key from [NewsAPI.org](https://newsapi.org/).
        *   Replace `"YOUR_NEWS_API_KEY_HERE"` with your actual NewsAPI key:
          ```.env
          NEWS_API_KEY="<your-actual-newsapi-key>"
          ```

## Usage

There are two main ways to use this project:

### 1. Running the Jupyter Notebooks

This is the recommended way to understand the step-by-step process.

1.  **Start Jupyter Lab or Notebook:** Make sure your virtual environment is activated.
    ```bash
    jupyter lab
    # OR
    # jupyter notebook
    ```
2.  **Navigate:** Open your browser to the provided URL (usually `http://localhost:8888/`).
3.  **Open Notebooks:** Navigate into the `notebooks/` directory.
4.  **Run Sequentially:** Execute the notebooks in order (`01` to `04`). Each notebook builds upon the data generated by the previous one.
    *   `01_news_fetching.ipynb`: Fetches news and saves it to `notebooks/data/fetched_articles.json`.
    *   `02_sentiment_analysis.ipynb`: Loads fetched articles, performs sentiment analysis, and saves results.
    *   `03_summarization.ipynb`: Loads fetched articles, generates summaries, and saves results.
    *   `04_impact_assessment.ipynb`: Loads summaries/articles, performs impact assessment, and saves results.

    *Note: The first time you run notebooks 02, 03, and 04, the Hugging Face models will be downloaded, which can take a significant amount of time and disk space.*

### 2. Running the FastAPI API

This runs the backend service.

1.  **Start the API Server:** Make sure your virtual environment is activated and you are in the project root directory (`energy_news_analyzer`).
    ```bash
    uvicorn api.main:app --reload --port 8000
    ```
    *   `--reload`: Automatically restarts the server when code changes (useful for development).
    *   `--port 8000`: Specifies the port to run on.

2.  **Access API Documentation:** Open your browser to `http://localhost:8000/docs`. This will show the interactive Swagger UI documentation.

3.  **Send Requests:** You can use the Swagger UI or tools like `curl` or Postman to send POST requests to the `/analyze` endpoint.

    **Example using `curl`:**
    ```bash
    curl -X POST "http://localhost:8000/analyze" \
    -H "Content-Type: application/json" \
    -d 
{
      "query": "offshore wind energy policy",
      "page_size": 5,
      "sort_by": "relevancy"
    }

    ```

4.  **Health Check:** Check the API and model status at `http://localhost:8000/health`.

## Potential Improvements / Future Work

*   **More Sophisticated LLM:** Replace the basic `gpt2` model for impact assessment with a more capable open-source model (e.g., Llama, Mixtral) or integrate with a paid LLM API if budget allows.
*   **Langgraph Implementation:** Explore using Langgraph for more complex agentic workflows, potentially involving multiple analysis steps or conditional logic.
*   **Database Integration:** Store fetched news and analysis results in a database (e.g., SQLite, PostgreSQL, Azure Cosmos DB) for persistence and historical analysis.
*   **Improved Impact Assessment:** Develop a more robust prompt or fine-tune a model specifically for assessing energy market news impact.
*   **Entity Recognition:** Add Named Entity Recognition (NER) to identify companies, locations, and people mentioned in the news.
*   **Topic Modeling:** Implement topic modeling to categorize news articles automatically.
*   **Frontend:** Build a simple web interface (e.g., using React or Streamlit) to interact with the API.
*   **Deployment:** Containerize the application using Docker and deploy it to a cloud platform (e.g., Azure App Service, AWS ECS, Google Cloud Run).
*   **Error Handling:** Enhance error handling and reporting.

## License

This project is open-source and available under the [MIT License](LICENSE). (You would need to add a LICENSE file with the MIT license text).

