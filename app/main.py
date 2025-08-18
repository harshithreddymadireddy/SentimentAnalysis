from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from builtins import zip
from app.services.sentiment_service import analyzer_instance
from app.services.summary_service import summarizer_instance
import traceback

app = FastAPI(
    title="Influencer Sentiment Insight API with Debug",
    description="Sentiment analysis and insight NLG with error debugging."
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, statements: str = Form(...)):
    try:
        texts = [line.strip() for line in statements.splitlines() if line.strip()]
        print(f"Received statements for analyze: {texts}")
        sentiment_results = analyzer_instance.analyze(texts)
        print(f"Sentiment results: {sentiment_results}")
        summary_text, _ = summarizer_instance.generate_summary(texts, sentiment_results)
        print(f"Generated summary: {summary_text}")
        result = {
            "input_texts": texts,
            "sentiment_results": sentiment_results,
            "summary_text": summary_text
        }
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": result, 
            "input_text": statements, 
            "zip": zip
        })
    except Exception as e:
        print("Error in analyze endpoint:", e)
        traceback.print_exc()
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": None, 
            "input_text": statements, 
            "error": str(e), 
            "zip": zip
        })
