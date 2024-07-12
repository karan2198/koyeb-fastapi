from pydantic import BaseModel

class RecommendationRequest(BaseModel):
    search_terms: str
    age: str
    social_category: str
    gender: str
    domicile_of_tripura: str
    num_recommendations: int = 5
