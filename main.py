from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from classifier import classify_article
from datetime import datetime, timezone

app = FastAPI(
    title="Performativ News Classifier",
    description="Classifies news articles as GOOD_NEWS, BAD_NEWS, or UNRELATED for Performativ.",
    version="1.0.0"
)

# In-memory store for the last 20 classifications
recent_results = []

# Stats counter
stats = {
    "total_classified": 0,
    "GOOD_NEWS": 0,
    "BAD_NEWS": 0,
    "UNRELATED": 0,
    "LOW_CONFIDENCE": 0,
    "FETCH_FAILED": 0,
    "service_started_at": datetime.now(timezone.utc).isoformat()
}

class ClassifyRequest(BaseModel):
    url: str

@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok"}

@app.post("/classify")
def classify(request: ClassifyRequest):
    """
    Accepts a news article URL and returns a classification.
    """
    # Basic input validation
    if not request.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    
    if len(request.url) > 2000:
        raise HTTPException(status_code=400, detail="URL is too long")

    result = classify_article(request.url)

    # Update stats
    stats["total_classified"] += 1
    label = result["label"]
    if label in stats:
        stats[label] += 1

    # Save to recent results (keep last 20)
    recent_results.append(result)
    if len(recent_results) > 20:
        recent_results.pop(0)

    return result

@app.get("/latest")
def latest():
    """Returns the most recent classifications."""
    return {"results": list(reversed(recent_results))}

@app.get("/stats")
def get_stats():
    """Returns classification statistics since the service started."""
    return {
        "total_classified": stats["total_classified"],
        "breakdown": {
            "GOOD_NEWS": stats["GOOD_NEWS"],
            "BAD_NEWS": stats["BAD_NEWS"],
            "UNRELATED": stats["UNRELATED"],
            "LOW_CONFIDENCE": stats["LOW_CONFIDENCE"],
            "FETCH_FAILED": stats["FETCH_FAILED"],
        },
        "service_started_at": stats["service_started_at"]
    }
