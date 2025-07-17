"""
Search API endpoints for recipe image retrieval
"""

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Dict, Any
import logging
from datetime import datetime

from ..schemas.search import (
    RecipeSearchRequest,
    RecipeSearchResponse,
    ErrorResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/search", response_model=RecipeSearchResponse, status_code=status.HTTP_200_OK)
async def search_recipes(
    image: UploadFile = File(..., description="Image file to search for similar recipes"),
    top_k: int = Form(default=3, ge=1, le=10, description="Number of top results to return")
):
    """
    Search for similar recipes based on uploaded image
    
    Args:
        image: Uploaded image file (JPEG, PNG, etc.)
        top_k: Number of top similar recipes to return (1-10)
        
    Returns:
        RecipeSearchResponse: List of similar recipes with metadata
    """
    try:
        from ..services.search_service import SearchService
        search_service = SearchService.get_instance()
        
        # Validate file type
        if not image.content_type or not image.content_type.startswith('image/'):
            logger.warning(f"Invalid file type: {image.content_type}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Read image data
        try:
            image_data = await image.read()
            logger.info(f"Image uploaded: {image.filename}, size: {len(image_data)} bytes, type: {image.content_type}")
        except Exception as e:
            logger.error(f"Error reading uploaded file: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Error reading uploaded image file"
            )
        
        # Create request object and convert to domain
        request = RecipeSearchRequest(topK=top_k)
        search_domain = request.to_domain(image_data)
        
        # Search for recipes
        logger.info(f"Starting recipe search with top_k={top_k}")
        result_domain = await search_service.search_recipes(search_domain)
        
        # Convert to response
        response = RecipeSearchResponse.from_domain(result_domain)
        
        if not result_domain.success:
            logger.error(f"Search failed: {result_domain.error_message}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    errorMessage=result_domain.error_message or "Search failed",
                    timestamp=datetime.now().isoformat()
                ).model_dump()
            )
        
        logger.info(f"Search completed successfully: {result_domain.total_results} results in {result_domain.processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error during recipe search: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                errorMessage="Internal server error during recipe search",
                timestamp=datetime.now().isoformat(),
                details={"error": str(e)}
            ).model_dump()
        ) 