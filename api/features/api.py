from fastapi import APIRouter
from .test.api.test import router as test_router
from .search.api.search import router as search_router

api_router = APIRouter()

# Include routers from all features
api_router.include_router(test_router, prefix="/test", tags=["test"])
api_router.include_router(search_router, prefix="/search", tags=["search"]) 