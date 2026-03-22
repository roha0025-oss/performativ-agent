import anthropic
import json
import os
from datetime import datetime, timezone

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """
You are a news classification agent for Performativ — a wealth management software platform 
serving private banks, family offices, asset managers, and financial institutions.

Your job is to read a news article and classify it as one of:
- GOOD_NEWS: Relevant to Performativ's business AND net positive (e.g. growth in wealth tech, 
  new regulations creating demand, competitor struggles, AI adoption in finance)
- BAD_NEWS: Relevant to Performativ's business BUT net negative (e.g. regulations that hurt 
  the sector, major competitor launches, cybersecurity incidents in fintech)
- UNRELATED: Not materially relevant (e.g. sports, entertainment, unrelated tech, local news)

Relevant topics include: wealth management software, portfolio management, private banks, 
family offices, RIAs, asset managers, regulation (DORA, MiFID II, FiDA), compliance, 
reporting, portfolio analytics, AI in regulated financial workflows, enterprise data 
integration, legacy modernization, custodian connectivity.

You MUST respond with ONLY valid JSON in this exact format, nothing else:
{
  "label": "GOOD_NEWS" or "BAD_NEWS" or "UNRELATED",
  "confidence": a number between 0.0 and 1.0,
  "reasoning": "2-3 sentence explanation of your decision",
  "relevance_topics": ["topic1", "topic2"]
}
"""

def validate_url(url: str) -> tuple[bool, str]:
    """Basic URL validation before calling the API."""
    if not url.startswith("http://") and not url.startswith("https://"):
        return False, "URL must start with http:// or https://"
    if len(url) < 10:
        return False, "URL is too short to be valid"
    if " " in url:
        return False, "URL contains spaces and is invalid"
    return True, ""

def classify_article(url: str) -> dict:
    """
    Takes a news article URL, uses Claude with web search to fetch and classify it.
    Returns a structured result dict.
    """

    # Validate URL before calling the API
    is_valid, validation_error = validate_url(url)
    if not is_valid:
        return {
            "url": url,
            "label": "FETCH_FAILED",
            "confidence": 0.0,
            "reasoning": f"Invalid URL: {validation_error}",
            "relevance_topics": [],
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "error": validation_error
        }

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[
                {
                    "role": "user",
                    "content": f"Please fetch and classify this news article: {url}"
                }
            ]
        )

        # Extract the text response from Claude's reply
        result_text = ""
        for block in response.content:
            if hasattr(block, "type") and block.type == "text":
                result_text += block.text

        result_text = result_text.strip()

        # If no text found, Claude may still be in tool-use mode — handle it
        if not result_text:
            return {
                "url": url,
                "label": "FETCH_FAILED",
                "confidence": 0.0,
                "reasoning": "Model returned no text response (tool use only).",
                "relevance_topics": [],
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "error": "Empty text response"
            }

        # Strip markdown code fences if present
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            result_text = "\n".join(lines[1:-1])

        classification = json.loads(result_text)
        confidence = classification["confidence"]
        label = classification["label"]

        # Flag low confidence results
        if confidence < 0.5:
            label = "LOW_CONFIDENCE"

        return {
            "url": url,
            "label": label,
            "confidence": confidence,
            "reasoning": classification["reasoning"],
            "relevance_topics": classification.get("relevance_topics", []),
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "error": None
        }

    except json.JSONDecodeError:
        return {
            "url": url,
            "label": "FETCH_FAILED",
            "confidence": 0.0,
            "reasoning": "Failed to parse classification response from the model.",
            "relevance_topics": [],
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "error": "JSON parse error"
        }

    except anthropic.APIConnectionError:
        return {
            "url": url,
            "label": "FETCH_FAILED",
            "confidence": 0.0,
            "reasoning": "Could not connect to the Anthropic API.",
            "relevance_topics": [],
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "error": "API connection error"
        }

    except anthropic.RateLimitError:
        return {
            "url": url,
            "label": "FETCH_FAILED",
            "confidence": 0.0,
            "reasoning": "Anthropic API rate limit reached. Please try again shortly.",
            "relevance_topics": [],
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "error": "Rate limit error"
        }

    except anthropic.AuthenticationError:
        return {
            "url": url,
            "label": "FETCH_FAILED",
            "confidence": 0.0,
            "reasoning": "Invalid Anthropic API key.",
            "relevance_topics": [],
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "error": "Authentication error - check your API key"
        }

    except Exception as e:
        return {
            "url": url,
            "label": "FETCH_FAILED",
            "confidence": 0.0,
            "reasoning": "An unexpected error occurred during classification.",
            "relevance_topics": [],
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }
