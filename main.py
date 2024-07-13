from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from recommendation.models import RecommendationRequest
from recommendation.recommend import get_recommendations

import asyncio
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
