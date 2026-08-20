from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertModel
from fastapi.concurrency import asynccontextmanager, run_in_threadpool
import nltk
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
from predict import preprocess_tweet

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading XGBoost model...")
    app.state.xgb_model = joblib.load("models/xgboost_sentiment_model.pkl")
    
    print("Loading TF-IDF vectorizer...")
    app.state.tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    app.state.tokenizer = None
    app.state.bert_model = None
    print("ALL models loaded successfully.")
    yield
    # Shutdown code
    print("Shutting down the application...")


#--------------------------------------------------------
# fast api setup
#--------------------------------------------------------
class Tweet(BaseModel):
    text: str

app = FastAPI(title="Market Sentiment Analysis API",lifespan=lifespan)

#--------------------------------------------------------
# root endpoint
#--------------------------------------------------------

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Market Sentiment Analysis API!"}


#-------------------------------------------------------------------------
# Prediction function
#-----------------------------------------------------------
def load_bert(app):

    if app.state.bert_model is None:

        print("Loading BERT model...")

        MODEL_ID = "YOUR_USERNAME/market-sentiment-bert"

        HF_TOKEN = os.getenv("HF_TOKEN")

        app.state.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN
        )

        app.state.bert_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                MODEL_ID,
                token=HF_TOKEN
            )
        )

        app.state.bert_model.eval()

        print("BERT model loaded successfully.")
        
def predict_sentiment(tweet: str, tokenizer, bert_model):
    inputs = tokenizer(
        tweet,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    prediction = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[0][prediction].item()

    return prediction, confidence

def predict_sentiment_xgb(
    tweet: str,
    xgb_model,
    tfidf):

    # Same preprocessing used during training
    preprocessed_tweet = preprocess_tweet(tweet)

    # Transform using saved TF-IDF
    tweet_vector = tfidf.transform(
        [preprocessed_tweet]
    )

    prediction = int(
    xgb_model.predict(tweet_vector)[0]
)

    confidence = float(
        max(xgb_model.predict_proba(tweet_vector)[0])
    )

    return prediction, confidence

@app.post("/predict_bert")
async def predict(request: Tweet):
    await run_in_threadpool(
        load_bert,
        app
    )
    prediction, confidence = await run_in_threadpool(
        predict_sentiment,
        request.text,
        app.state.tokenizer,
        app.state.bert_model    
    )

    sentiment = (
        "Positive"
        if prediction == 1
        else "Negative"
    )

    return {
        "tweet": request.text,
        "prediction": prediction,
        "sentiment": sentiment,
        "confidence": round(confidence, 4)
    }

@app.post("/predict_xgb")
async def predict_xgb(request: Tweet):

    prediction, confidence = await run_in_threadpool(
        predict_sentiment_xgb,
        request.text,
        app.state.xgb_model,
        app.state.tfidf
    )

    sentiment = (
        "Positive"
        if prediction == 1
        else "Negative"
    )

    return {
        "tweet": request.text,
        "prediction": prediction,
        "sentiment": sentiment,
        "confidence": round(confidence, 4)
    }
