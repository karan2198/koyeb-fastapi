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

app = FastAPI()

# CORS Configuration
origins = [
    "http://localhost",
    "http://localhost:3000",  # Add the origin of your React frontend
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

def preprocess_and_save_data():
    rec = pd.read_csv(DATA_FILE)
    rec = rec.dropna()

    # Map age ranges to numerical codes
    age_mapping = {'Below 10': 0, '10-15': 1, '16-20': 2, '21-25': 3, '26-30': 4, 
                   '31-35': 5, '36-40': 6, '41-45': 7, '46-50': 8, 'Above 50': 9}
    rec['age'] = rec['age'].map(age_mapping)

    # Convert categorical variables to numerical codes
    caste_mapping = {'SC': 0, 'ST': 1, 'OBC': 2}
    rec['social_category'] = rec['social_category'].map(caste_mapping)

    gender_mapping = {'M': 0, 'F': 1, 'T': 2}
    rec['gender'] = rec['gender'].map(gender_mapping)

    rec['domicile_of_tripura'] = rec['domicile_of_tripura'].map({'Y': 1, 'N': 0})

    rec['scheme_text'] = rec['scheme_name'] + ' ' + rec['description']

    # Save the preprocessed data
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(rec, f)

    return rec

def fit_and_save_vectorizer(rec):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(rec['scheme_text'])

    # Save the vectorizer and TF-IDF matrix
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(TFIDF_MATRIX_FILE, 'wb') as f:
        pickle.dump(tfidf_matrix, f)

    return vectorizer, tfidf_matrix

def load_data_and_vectorizer():
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

    return rec, vectorizer, tfidf_matrix

# Load data and vectorizer on startup
rec, vectorizer, tfidf_matrix = load_data_and_vectorizer()

class RecommendationRequest(BaseModel):
    search_terms: str
    age: str
    social_category: str
    gender: str
    domicile_of_tripura: str
    num_recommendations: int = 5

def get_recommendations(search_terms: str, age: str, social_category: str, gender: str, domicile_of_tripura: str, num_recommendations: int = 5):
    search_query = f"{search_terms} {age} {social_category} {gender} {domicile_of_tripura}"
    query_vector = vectorizer.transform([search_query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_similar_indices = cosine_similarities.argsort()[::-1]

    seen = set()
    unique_indices = [index for index in top_similar_indices if not (rec.iloc[index]['scheme_name'] in seen or seen.add(rec.iloc[index]['scheme_name']))]
    unique_recommendations = unique_indices[:num_recommendations]
    
    # Handle NaN values
    recommendations = rec.iloc[unique_recommendations].fillna("").to_dict('records')
    
    return recommendations

# Endpoint to fetch recommendations
@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    try:
        recommendations = await asyncio.to_thread(get_recommendations,
            request.search_terms,
            request.age,
            request.social_category,
            request.gender,
            request.domicile_of_tripura,
            request.num_recommendations
        )
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
