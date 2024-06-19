from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# Load and preprocess data
def load_and_preprocess_data():
    rec = pd.read_csv('recommendation_dataset.csv')
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

    return rec

# Load or fit TF-IDF vectorizer
def load_or_fit_vectorizer(rec):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(rec['scheme_text'])
    return vectorizer, tfidf_matrix

# Load and preprocess data
rec = load_and_preprocess_data()

# Load or fit TF-IDF vectorizer
vectorizer, tfidf_matrix = load_or_fit_vectorizer(rec)

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
def recommend(request: RecommendationRequest):
    try:
        recommendations = get_recommendations(
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
