import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

DATA_FILE = 'data/recommendation_dataset.csv'
PICKLE_FILE = 'data/preprocessed_data.pkl'
VECTORIZER_FILE = 'data/tfidf_vectorizer.pkl'
TFIDF_MATRIX_FILE = 'data/tfidf_matrix.pkl'

rec = None
vectorizer = None
tfidf_matrix = None

async def load_data_and_vectorizer():
    global rec, vectorizer, tfidf_matrix
    try:
        with open(PICKLE_FILE, 'rb') as f:
            rec = pickle.load(f)
        with open(VECTORIZER_FILE, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(TFIDF_MATRIX_FILE, 'rb') as f:
            tfidf_matrix = pickle.load(f)
    except Exception as e:
        print(f"Error loading data and vectorizer: {str(e)}")

def get_query_vector(search_query):
    global vectorizer
    return vectorizer.transform([search_query])

def get_recommendations(search_terms: str, age: str, social_category: str, gender: str, domicile_of_tripura: str, num_recommendations: int = 5):
    global rec, vectorizer, tfidf_matrix
    search_query = f"{search_terms} {age} {social_category} {gender} {domicile_of_tripura}"
    query_vector = get_query_vector(search_query)
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_similar_indices = cosine_similarities.argsort()[::-1]

    seen = set()
    unique_indices = [index for index in top_similar_indices if not (rec.iloc[index]['scheme_name'] in seen or seen.add(rec.iloc[index]['scheme_name']))]
    unique_recommendations = unique_indices[:num_recommendations]
    
    recommendations = rec.iloc[unique_recommendations].fillna("").to_dict('records')
    
    return recommendations
