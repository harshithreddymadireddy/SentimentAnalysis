# Influencer Sentiment Insight API

A production-ready FastAPI application that provides AI-powered sentiment analysis and natural language generation (NLG) for influencer statements and social media content.

---

## 🚀 Features

- **Batch Sentiment Analysis:** Analyze multiple influencer statements simultaneously using the fast and accurate DistilBERT model.
- **Insight Generation:** Generate concise, summarizations using GPT-2 for actionable insights.
- **RESTful API:** Clean, well-documented endpoints with automatic OpenAPI (Swagger) and ReDoc documentation.
- **Health Checks:** Built-in service health endpoints for monitoring.
- **Comprehensive Testing:** Full test suite using `pytest` for confidence in code quality.
- **Production Ready:** Structured, modular codebase with proper error handling and logging.

---

## 📊 AI Models Used

- **Sentiment Analysis:**  
  `distilbert-base-uncased-finetuned-sst-2-english`  
  - Pre-trained on the Stanford Sentiment Treebank (SST-2) dataset  
  - Optimized for fast inference  
  - Performs binary sentiment classification (POSITIVE / NEGATIVE)

- **Text Generation:**  
  `gpt2`  
  - Used for natural language generation of insights  
  - Produces professional summaries from sentiment statistics  
  - Easily extensible for domain-specific fine-tuning

---

## 🏗️ Project Structure

sentiment-ai-api/
├── app/
│ ├── main.py # Entry point to FastAPI application
│ ├── models.py # Pydantic data validation and request/response schemas
│ ├── utils.py # Utility functions used throughout the app
│ ├── routers/ # API route modules for clean separation
│ │ ├── sentiment.py # Sentiment analysis endpoints
│ │ └── summary.py # Insight generation endpoints
│ └── services/ # Core business and AI logic
│ ├── sentiment_service.py # Sentiment classification logic and model
│ └── summary_service.py # Statistics calculation and NLG-based insight generation
├── test/ # Automated test suites
│ ├── test_sentiment.py
│ └── test_summary.py
├── requirements.txt # Python dependencies, pinned for reproducibility
└── README.md # Project documentation (this file)


---

## 📖 Usage

- Start the FastAPI server:

uvicorn app.main:app --reload

- Open your browser and go to: `http://localhost:8000`

- Enter multiple influencer statements and press the **Analyze&GenerateSummary** button to get sentiment classification and AI-generated business insight summaries.


## 📦 Installation

Install dependencies:


pip install -r requirements.txt
