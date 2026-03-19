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

def classify_article(url: str) -> dict:
    try:
        response = client.messages.create(
            model="claude-opus-4-5",
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

        result_text = ""
        for block in response.content:
            if block.type == "text":
                result_text += block.text

        result_text = result_text.strip()

        if result_text.startswith("```"):
            lines = result_text.split("\n")
            result_text = "\n".join(lines[1:-1])

        classification = json.loads(result_text)

        return {
            "url": url,
            "label": classification["label"],
            "confidence": classification["confidence"],
            "reasoning": classification["reasoning"],
            "relevance_topics": classification.get("relevance_topics", []),
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "error": None
        }

    except json.JSONDecodeError:
        return {
            "url": url,
            "label": "UNRELATED",
            "confidence": 0.0,
            "reasoning": "Failed to parse classification response.",
            "relevance_topics": [],
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "error": "JSON parse error"
        }

    except Exception as e:
        return {
            "url": url,
            "label": "UNRELATED",
            "confidence": 0.0,
            "reasoning": "An error occurred during classification.",
            "relevance_topics": [],
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }