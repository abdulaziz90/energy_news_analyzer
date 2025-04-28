import logging
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Initialization (Load models once) ---

sentiment_pipeline = None
summarizer_pipeline = None
qa_pipeline = None # For impact assessment

DEFAULT_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_SUMMARIZER_MODEL = "facebook/bart-large-cnn"
# Using a QA model for impact assessment as a simpler alternative to a full reasoning model
DEFAULT_QA_MODEL = "distilbert-base-cased-distilled-squad"

try:
    logger.info(f"Loading sentiment analysis model: {DEFAULT_SENTIMENT_MODEL}")
    # Explicitly load model and tokenizer for sentiment
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_SENTIMENT_MODEL)
    sentiment_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_SENTIMENT_MODEL)
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=sentiment_model,
        tokenizer=sentiment_tokenizer,
        # Use CPU if no GPU or to avoid potential CUDA issues in sandbox
        device=-1 # -1 for CPU, 0 for first GPU
    )
    logger.info("Sentiment analysis model loaded successfully.")

    logger.info(f"Loading summarization model: {DEFAULT_SUMMARIZER_MODEL}")
    # Explicitly load model and tokenizer for summarization
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_SUMMARIZER_MODEL)
    summarizer_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_SUMMARIZER_MODEL)
    summarizer_pipeline = pipeline(
        "summarization",
        model=summarizer_model,
        tokenizer=summarizer_tokenizer,
        device=-1 # Use CPU
    )
    logger.info("Summarization model loaded successfully.")

    # logger.info(f"Loading QA model for impact assessment: {DEFAULT_QA_MODEL}")
    # qa_pipeline = pipeline(
    #     "question-answering",
    #     model=DEFAULT_QA_MODEL,
    #     tokenizer=DEFAULT_QA_MODEL,
    #     device=-1 # Use CPU
    # )
    # logger.info("QA model loaded successfully.")
    # Note: QA model might not be ideal for 'impact assessment'. 
    # Using a simple LLM chain approach instead for now.
    # If a more capable local LLM is needed, setup would be more complex.

    # Using a simple HuggingFacePipeline for basic text generation (impact assessment)
    # This uses a default model, might need adjustment for better results
    logger.info("Initializing basic LLM pipeline for impact assessment.")
    impact_llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2", # Using gpt2 as a basic, small model for demonstration
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 100},
        device=-1 # Use CPU
    )
    logger.info("Basic LLM pipeline initialized.")

except Exception as e:
    logger.error(f"Error loading Hugging Face models: {e}", exc_info=True)
    # Set pipelines to None if loading fails
    sentiment_pipeline = None
    summarizer_pipeline = None
    impact_llm = None

# --- Analysis Functions ---

def analyze_sentiment(text: str) -> dict | None:
    """Analyzes the sentiment of a given text using a pre-loaded pipeline.

    Args:
        text: The input text.

    Returns:
        A dictionary containing the sentiment label ("POSITIVE" or "NEGATIVE")
        and score, or None if the model is not available or an error occurs.
    """
    if not sentiment_pipeline:
        logger.error("Sentiment analysis model not available.")
        return None
    try:
        # Limit text length to avoid excessive processing time/memory
        max_length = sentiment_pipeline.tokenizer.model_max_length
        truncated_text = text[:max_length]
        results = sentiment_pipeline(truncated_text)
        # The pipeline returns a list, we take the first result
        if results:
            return results[0]
        else:
            return None
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}", exc_info=True)
        return None

def summarize_text(text: str, min_length_ratio: float = 0.1, max_length_ratio: float = 0.3) -> str | None:
    """Summarizes the given text using a pre-loaded pipeline.

    Args:
        text: The input text.
        min_length_ratio: Minimum length of the summary as a ratio of the original text.
        max_length_ratio: Maximum length of the summary as a ratio of the original text.

    Returns:
        The summarized text, or None if the model is not available or an error occurs.
    """
    if not summarizer_pipeline:
        logger.error("Summarization model not available.")
        return None
    try:
        # Calculate min/max length based on input text length
        text_length = len(text.split()) # Approx word count
        min_len = max(30, int(text_length * min_length_ratio)) # Ensure min length is reasonable
        max_len = min(150, int(text_length * max_length_ratio)) # Ensure max length is reasonable
        if min_len >= max_len:
             min_len = max(30, max_len // 2)

        # Limit input text length for the summarizer model
        max_input_length = summarizer_pipeline.tokenizer.model_max_length
        truncated_text = " ".join(text.split()[:max_input_length]) # Truncate by words

        results = summarizer_pipeline(truncated_text, max_length=max_len, min_length=min_len, do_sample=False)
        if results:
            return results[0]["summary_text"]
        else:
            return None
    except Exception as e:
        logger.error(f"Error during summarization: {e}", exc_info=True)
        return None

def assess_impact(text: str, context: str = "energy market") -> str | None:
    """Provides a qualitative assessment of the potential impact of the news text.
       Uses a simple LLM chain for demonstration.

    Args:
        text: The news article content or summary.
        context: The specific market context (e.g., "oil prices", "renewable energy stocks").

    Returns:
        A brief textual assessment of the potential impact, or None if the model fails.
    """
    if not impact_llm:
        logger.error("Impact assessment LLM not available.")
        return None

    template = """
    News article: {news_text}

    Based on the news article above, briefly assess its potential impact on the {context}. Consider factors like market sentiment, supply/demand, policy changes, or company performance.

    Potential Impact Assessment:"""

    prompt = PromptTemplate(template=template, input_variables=["news_text", "context"])
    chain = LLMChain(llm=impact_llm, prompt=prompt)

    try:
        # Limit input text length
        max_input_length = 500 # Arbitrary limit for the prompt
        truncated_text = text[:max_input_length]

        result = chain.run(news_text=truncated_text, context=context)
        # The result might include the prompt, try to extract just the assessment
        assessment_marker = "Potential Impact Assessment:"
        if assessment_marker in result:
            assessment = result.split(assessment_marker, 1)[1].strip()
            # Further cleanup if the model repeats the prompt etc.
            if template.split("\n\n")[0] in assessment:
                 assessment = assessment.split(template.split("\n\n")[0])[0].strip()
            return assessment if assessment else "Assessment could not be generated."
        else:
            # Fallback if marker not found, return raw output (might be messy)
            return result.strip()

    except Exception as e:
        logger.error(f"Error during impact assessment: {e}", exc_info=True)
        return None

# Example usage (for testing)
if __name__ == "__main__":
    test_text = ("Significant new government subsidies for solar panel installation were announced today. "
                 "This policy aims to boost renewable energy adoption and reduce reliance on fossil fuels. "
                 "Major solar companies saw their stock prices jump in response.")

    print("--- Testing Sentiment Analysis ---")
    sentiment = analyze_sentiment(test_text)
    if sentiment:
        print(f"Sentiment: {sentiment}")
    else:
        print("Sentiment analysis failed or model not loaded.")

    print("\n--- Testing Summarization ---")
    summary = summarize_text(test_text)
    if summary:
        print(f"Summary: {summary}")
    else:
        print("Summarization failed or model not loaded.")

    print("\n--- Testing Impact Assessment ---")
    impact = assess_impact(test_text, context="renewable energy stocks")
    if impact:
        print(f"Impact Assessment: {impact}")
    else:
        print("Impact assessment failed or model not loaded.")

