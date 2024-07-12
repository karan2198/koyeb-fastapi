from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import asyncio
import os
from functools import lru_cache
import time

app = FastAPI()

# CORS Configuration
origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://my-scheam-gov.vercel.app",
    "https://my-scheam-o8ylu1eup-karan2198s-projects.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

DATA_FILE = 'recommendation_dataset.csv'
PICKLE_FILE = 'preprocessed_data.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
TFIDF_MATRIX_FILE = 'tfidf_matrix.pkl'

rec = None
vectorizer = None
tfidf_matrix = None

def preprocess_and_save_data():
    rec = pd.read_csv(DATA_FILE)
    rec = rec.dropna()

    age_mapping = {'Below 10': 0, '10-15': 1, '16-20': 2, '21-25': 3, '26-30': 4, 
                   '31-35': 5, '36-40': 6, '41-45': 7, '46-50': 8, 'Above 50': 9}
    rec['age'] = rec['age'].map(age_mapping)

    caste_mapping = {'SC': 0, 'ST': 1, 'OBC': 2}
    rec['social_category'] = rec['social_category'].map(caste_mapping)

    gender_mapping = {'M': 0, 'F': 1, 'T': 2}
    rec['gender'] = rec['gender'].map(gender_mapping)

    rec['domicile_of_tripura'] = rec['domicile_of_tripura'].map({'Y': 1, 'N': 0})

    rec['scheme_text'] = rec['scheme_name'] + ' ' + rec['description']

    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(rec, f)

    return rec

def fit_and_save_vectorizer(rec):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(rec['scheme_text'])

    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(TFIDF_MATRIX_FILE, 'wb') as f:
        pickle.dump(tfidf_matrix, f)

    return vectorizer, tfidf_matrix

def load_data_and_vectorizer():
    global rec, vectorizer, tfidf_matrix
    if os.path.exists(PICKLE_FILE) and os.path.exists(VECTORIZER_FILE) and os.path.exists(TFIDF_MATRIX_FILE):
        with open(PICKLE_FILE, 'rb') as f:
            rec = pickle.load(f)
        with open(VECTORIZER_FILE, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(TFIDF_MATRIX_FILE, 'rb') as f:
            tfidf_matrix = pickle.load(f)
    else:
        rec = preprocess_and_save_data()
        vectorizer, tfidf_matrix = fit_and_save_vectorizer(rec)

class RecommendationRequest(BaseModel):
    search_terms: str
    age: str
    social_category: str
    gender: str
    domicile_of_tripura: str
    num_recommendations: int = 5

@lru_cache(maxsize=128)
def get_query_vector(search_query):
    return vectorizer.transform([search_query])

def get_recommendations(search_terms: str, age: str, social_category: str, gender: str, domicile_of_tripura: str, num_recommendations: int = 5):
    search_query = f"{search_terms} {age} {social_category} {gender} {domicile_of_tripura}"
    query_vector = get_query_vector(search_query)
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_similar_indices = cosine_similarities.argsort()[::-1]

    seen = set()
    unique_indices = [index for index in top_similar_indices if not (rec.iloc[index]['scheme_name'] in seen or seen.add(rec.iloc[index]['scheme_name']))]
    unique_recommendations = unique_indices[:num_recommendations]
    
    recommendations = rec.iloc[unique_recommendations].fillna("").to_dict('records')
    
    return recommendations

@app.on_event("startup")
async def startup_event():
    load_data_and_vectorizer()

@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    try:
        start_time = time.time()
        recommendations = await asyncio.to_thread(get_recommendations,
            request.search_terms,
            request.age,
            request.social_category,
            request.gender,
            request.domicile_of_tripura,
            request.num_recommendations
        )
        elapsed_time = time.time() - start_time
        if elapsed_time > 10:
            raise HTTPException(status_code=500, detail=f"Request timed out after {elapsed_time:.2f} seconds")
        return {"recommendations": recommendations, "time_taken": elapsed_time}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
