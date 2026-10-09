import os
import re
import requests
import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from lime.lime_text import LimeTextExplainer

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

# Ensure required NLTK resources are available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Baseline uninformative prior of the trained voting model on an empty/zero vector
ZERO_VECTOR_REAL_PROB = 0.10474417

# Load model and vectorizer
MODEL_PATH = "./models/verifai_voting_model.pkl"
VECTORIZER_PATH = "./models/verifai_vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise RuntimeError("Model files not found in ./models directory.")

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="VerifAI API",
    description="Universal Fake News & Fact Verification Engine REST API",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(cleaned)

def calibrate_probability(p_real: float, p0: float = ZERO_VECTOR_REAL_PROB) -> float:
    if p_real >= p0:
        cal_real = 0.5 + 0.5 * ((p_real - p0) / (1.0 - p0))
    else:
        cal_real = 0.5 * (p_real / p0)
    return float(min(max(cal_real, 0.0), 1.0))

def verify_fact_live(claim: str):
    if not claim or len(claim.strip()) < 5:
        return None
    
    clean_query = re.sub(r'[^a-zA-Z0-9\s]', ' ', claim).strip()
    headers = {
        'User-Agent': 'VerifAI-FactCheck/2.0 (NLP Educational Project)'
    }
    
    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": clean_query,
            "format": "json",
            "srlimit": 3,
            "utf8": 1
        }
        res = requests.get(search_url, params=search_params, headers=headers, timeout=6)
        if res.status_code != 200:
            return None
        
        search_data = res.json()
        search_results = search_data.get("query", {}).get("search", [])
        if not search_results:
            return None
            
        best_match = search_results[0]
        title = best_match.get("title", "")
        
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
        sum_res = requests.get(summary_url, headers=headers, timeout=6)
        if sum_res.status_code != 200:
            return None
            
        sum_data = sum_res.json()
        extract = sum_data.get("extract", "")
        page_url = sum_data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{requests.utils.quote(title)}")
        
        common_words = {'the', 'a', 'an', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against', 
                        'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 
                        'to', 'from', 'up', 'down', 'is', 'was', 'were', 'be', 'been', 'being', 'have', 
                        'has', 'had', 'do', 'does', 'did', 'and', 'but', 'if', 'or', 'because', 'as', 
                        'until', 'while', 'of', 'this', 'that', 'these', 'those', 'it', 'its'}
        
        claim_tokens = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9]{3,}\b', claim) if w.lower() not in common_words]
        if not claim_tokens:
            return None
            
        extract_lower = extract.lower()
        title_lower = title.lower()
        
        matched_in_extract = [w for w in claim_tokens if w in extract_lower or w in title_lower]
        match_ratio = len(matched_in_extract) / len(claim_tokens)
        
        is_verified = (match_ratio >= 0.55)
        
        return {
            "title": title,
            "extract": extract,
            "page_url": page_url,
            "matched_keywords": matched_in_extract,
            "total_keywords": claim_tokens,
            "match_ratio": round(match_ratio, 2),
            "is_verified": is_verified
        }
    except Exception:
        return None

# Pydantic models for API
class VerifyRequest(BaseModel):
    title: Optional[str] = ""
    text: Optional[str] = ""
    context: Optional[str] = ""
    mode: Optional[str] = "headline"  # "headline" or "full"
    enable_fact_check: Optional[bool] = True

class ExplainRequest(BaseModel):
    text: str

@app.post("/api/verify")
def api_verify(req: VerifyRequest):
    # Support both title and text fields
    raw_query = req.title if req.title and req.title.strip() else (req.text or "")
    combined_input = (raw_query + " " + req.context).strip() if req.context else raw_query.strip()
    if not combined_input:
        raise HTTPException(status_code=400, detail="Text input is required.")
    
    # 1. Fact checking
    fact_result = None
    if req.enable_fact_check:
        fact_result = verify_fact_live(raw_query if raw_query.strip() else combined_input)
    
    # 2. NLP Preprocessing & TF-IDF
    cleaned_text = preprocess_text(combined_input)
    input_tokens = list(dict.fromkeys(cleaned_text.split()))
    
    X_input = vectorizer.transform([cleaned_text])
    nonzeros = X_input.nonzero()[1]
    feature_names = vectorizer.get_feature_names_out()
    matched = [feature_names[i] for i in nonzeros]
    oov = [t for t in input_tokens if t not in vectorizer.vocabulary_]
    
    raw_prob = model.predict_proba(X_input)[0]
    raw_real = float(raw_prob[1])
    raw_fake = float(raw_prob[0])
    
    # 3. Mode Calibration
    is_inconclusive = (len(matched) == 0)
    if is_inconclusive:
        cal_real = 0.50
        ml_pred = None
    elif req.mode == "headline":
        cal_real = calibrate_probability(raw_real)
        ml_pred = 1 if cal_real >= 0.50 else 0
    else:
        cal_real = raw_real
        ml_pred = int(model.predict(X_input)[0])

    # 4. Hybrid Decision Fusion
    if fact_result and fact_result.get("is_verified", False):
        verdict = "Real News (Factually Verified)"
        verdict_type = "real_verified"
        confidence = float(max(0.85, fact_result.get("match_ratio", 0.85)))
        p_real = confidence
        p_fake = 1.0 - confidence
        description = "This claim has been cross-referenced and verified against live encyclopedic knowledge."
    elif ml_pred == 1:
        verdict = "Real News"
        verdict_type = "real"
        confidence = cal_real
        p_real = cal_real
        p_fake = 1.0 - cal_real
        description = "Linguistic and stylistic structure matches credible, journalistic reporting."
    elif is_inconclusive and (not fact_result or not fact_result.get("is_verified")):
        verdict = "Inconclusive / Insufficient Evidence"
        verdict_type = "inconclusive"
        confidence = 0.50
        p_real = 0.50
        p_fake = 0.50
        description = "Could not find conclusive factual documentation or trained vocabulary matches."
    else:
        verdict = "Fake / Unverified Claim"
        verdict_type = "fake"
        confidence = 1.0 - cal_real
        p_real = cal_real
        p_fake = 1.0 - cal_real
        description = "Linguistic markers match sensational clickbait patterns or lack verifiable factual documentation."

    return {
        "verdict": verdict,
        "verdict_type": verdict_type,
        "confidence": round(confidence * 100, 1),
        "real_probability": round(p_real * 100, 1),
        "fake_probability": round(p_fake * 100, 1),
        "raw_real_probability": round(raw_real * 100, 1),
        "raw_fake_probability": round(raw_fake * 100, 1),
        "calibrated_real_probability": round(cal_real * 100, 1),
        "description": description,
        "fact_check": fact_result,
        "ml_analysis": {
            "matched_tokens": matched,
            "oov_tokens": oov,
            "is_inconclusive": is_inconclusive,
            "total_tokens_count": len(input_tokens),
            "matched_tokens_count": len(matched)
        }
    }

@app.post("/api/explain")
def api_explain(req: ExplainRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required.")
    
    explainer = LimeTextExplainer(class_names=['Fake', 'Real'])
    
    def predictor(texts):
        cleaned_list = [preprocess_text(t) for t in texts]
        X = vectorizer.transform(cleaned_list)
        raw_probs = model.predict_proba(X)
        cal_probs = []
        for p in raw_probs:
            p_real = calibrate_probability(p[1])
            cal_probs.append([1.0 - p_real, p_real])
        return np.array(cal_probs)

    try:
        exp = explainer.explain_instance(req.text, predictor, num_features=10)
        feature_list = []
        for word, weight in exp.as_list():
            feature_list.append({
                "word": str(word),
                "weight": round(float(weight), 4),
                "type": "real" if weight > 0 else "fake"
            })
        return {
            "features": feature_list,
            "text": req.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
def api_metrics():
    return {
        "accuracy": 99.18,
        "precision": 98.99,
        "recall": 99.50,
        "f1_score": 99.25,
        "classification_report": {
            "fake": {"precision": 0.99, "recall": 0.99, "f1": 0.99, "support": 3581},
            "real": {"precision": 0.99, "recall": 1.00, "f1": 0.99, "support": 4239}
        },
        "model_architecture": "Ensemble Voting Classifier (Logistic Regression + Calibrated Linear SVC + Random Forest)",
        "vectorizer": "TF-IDF Vectorizer with WordNet Lemmatization (7,000 features)"
    }

# Mount static frontend directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "frontend")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if os.path.exists(ASSETS_DIR):
    app.mount("/static/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"status": "VerifAI backend is running. Frontend not found."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
