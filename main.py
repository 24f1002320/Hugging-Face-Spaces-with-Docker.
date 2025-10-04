from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Initialize your model (this happens once when container starts)
classifier = pipeline("sentiment-analysis")

@app.get("/")
def read_root():
    return {"message": "Hello from Hugging Face Spaces!"}

@app.post("/analyze")
def analyze_sentiment(text: str):
    result = classifier(text)
    return {"sentiment": result[0]["label"], "score": result[0]["score"]}
