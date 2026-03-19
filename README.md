# Performativ News Classifier

A lightweight AI-powered backend service that classifies news articles by their relevance to Performativ's business.

## What it does

Given a news article URL, the agent automatically fetches and reads the article, then classifies it as:

- `GOOD_NEWS` — Relevant and net positive for Performativ
- `BAD_NEWS` — Relevant but net negative for Performativ  
- `UNRELATED` — Not materially relevant

## Architecture

Built using Option B (LLM with web search) — Claude reads and classifies the article in a single agent call, handling scraping, extraction, and reasoning together. No manual HTML parsing required.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/classify` | Classify a news article by URL |
| GET | `/latest` | Return the 20 most recent classifications |

## Example Request
```bash
curl -X POST https://performativ-agent-production.up.railway.app/classify \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.ft.com/content/example-article"}'
```

## Example Response
```json
{
  "url": "https://www.ft.com/content/example-article",
  "label": "GOOD_NEWS",
  "confidence": 0.91,
  "reasoning": "The article covers AI adoption in wealth management platforms, directly relevant to Performativ's core market.",
  "relevance_topics": ["wealth management software", "AI in finance"],
  "processed_at": "2026-03-19T10:00:00Z",
  "error": null
}
```

## Running Locally

1. Clone the repository
2. Create a `.env` file based on `.env.example`
3. Add your Anthropic API key to `.env`
4. Install dependencies:
```bash
pip install -r requirements.txt
```
5. Start the server:
```bash
uvicorn main:app --reload
```
6. Visit `http://localhost:8000/docs` to test the API

## Deployment

Deployed on Railway with continuous deployment from GitHub. Any push to `main` automatically redeploys the service.

## Requirements

- Python 3.10+
- Anthropic API key with credits
- Railway account (for deployment)
