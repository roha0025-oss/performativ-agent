from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from classifier import classify_article

app = FastAPI(
    title="Performativ News Classifier",
    description="Classifies news articles as GOOD_NEWS, BAD_NEWS, or UNRELATED for Performativ.",
    version="1.0.0"
)

recent_results = []

class ClassifyRequest(BaseModel):
    url: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify")
def classify(request: ClassifyRequest):
    if not request.url.startswith("http"):
        raise HTTPException(status_code=400, detail="URL must start with http or https")

    result = classify_article(request.url)

    recent_results.append(result)
    if len(recent_results) > 20:
        recent_results.pop(0)

    return result

@app.get("/latest")
def latest():
    return {"results": list(reversed(recent_results))}