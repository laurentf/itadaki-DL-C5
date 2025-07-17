import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core import config  # Import absolu
from features.api import api_router  # Import absolu
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create FastAPI app
app = FastAPI(
    title=config.PROJECT_NAME,
    description="API pour le système de recherche d'images de recettes Itadaki utilisant l'apprentissage profond",
    version="1.0.0",
    openapi_url=f"{config.API_V1_STR}/openapi.json",
)

# Configure CORS
origins = ["*"]  # In production, replace with your actual frontend domains

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Include routers
app.include_router(api_router, prefix=config.API_V1_STR)

@app.get("/")
async def root():
    return {"message": "🍜 Bienvenue sur l'API Itadaki. Visitez /docs pour la documentation API."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 