from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load and preprocess the data
rec = pd.read_csv('recommendation_dataset.csv')
# Drop rows with missing values
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

# Fit TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(rec['scheme_text'])

class RecommendationRequest(BaseModel):
    search_terms: str
    age: str
    social_category: str
    gender: str
    domicile_of_tripura: str
    num_recommendations: int = 5

def get_recommendations(search_terms: str, age: str, social_category: str, gender: str, domicile_of_tripura: str, num_recommendations: int = 5):
    # Convert input parameters to a single string for TF-IDF
    search_query = f"{search_terms} {age} {social_category} {gender} {domicile_of_tripura}"
    
    # Vectorize the search query
    query_vector = vectorizer.transform([search_query])
    
    # Calculate cosine similarities
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get indices of top similar schemes
    top_similar_indices = cosine_similarities.argsort()[::-1]
    
    # Remove duplicates while maintaining order
    seen = set()
    unique_indices = [index for index in top_similar_indices if not (rec.iloc[index]['scheme_name'] in seen or seen.add(rec.iloc[index]['scheme_name']))]
    
    # Select top unique recommendations
    unique_recommendations = unique_indices[:num_recommendations]
    
    return rec.iloc[unique_recommendations].to_dict('records')

@app.post("/recommend")
def recommend(request: RecommendationRequest):
    recommendations = get_recommendations(
        request.search_terms,
        request.age,
        request.social_category,
        request.gender,
        request.domicile_of_tripura,
        request.num_recommendations
    )
    return {"recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
