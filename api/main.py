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

@app.on_event("startup")
async def startup_event():
    """Load model and embeddings at application startup"""
    startup_logger = logging.getLogger(__name__)
    startup_logger.info("Itadaki API is starting up...")
    
    try:
        from features.search.services.search_service import SearchService
        success = await SearchService.initialize_model()
        
        if success:
            startup_logger.info("Startup completed successfully! API ready for fast searches.")
        else:
            startup_logger.error("Startup completed with errors. Model loading failed.")
            
    except Exception as e:
        startup_logger.error(f"Critical error during startup: {e}")

# Include routers
app.include_router(api_router, prefix=config.API_V1_STR)

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API Itadaki. Visitez /docs pour la documentation API."}

@app.get("/health")
async def health_check():
    """Health check endpoint with model status"""
    from features.search.services.search_service import SearchService
    search_service = SearchService.get_instance()
    
    return {
        "status": "healthy",
        "model_loaded": search_service.is_model_loaded(),
        "api_version": "1.0.0",
        "message": "Itadaki API is running"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 