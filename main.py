from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from recommendation.models import RecommendationRequest
from recommendation.recommend import get_recommendations, load_data_and_vectorizer

import asyncio
import logging
import time

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@app.on_event("startup")
async def on_startup():
    logger.info("Loading data and vectorizer...")
    await load_data_and_vectorizer()
    logger.info("Data and vectorizer loaded successfully.")

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
        end_time = time.time()
        logger.info(f"Request processed in {end_time - start_time} seconds")

        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error while getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
