"""
Search service for recipe image retrieval
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional
from PIL import Image
import io
import logging

from ..schemas.search import (
    RecipeSearchDomain, 
    RecipeSearchResultDomain, 
    RecipeResultDomain
)

logger = logging.getLogger(__name__)


class SearchService:
    """Service for recipe image search operations"""
    
    _instance = None
    _model_loader = None
    _is_loaded = False
    
    def __init__(self):
        pass
    
    @classmethod
    def get_instance(cls) -> 'SearchService':
        """Get singleton instance of SearchService"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    async def initialize_model(cls) -> bool:
        """Initialize model at application startup"""
        try:
            logger.info("Starting model initialization at application startup...")
            
            if cls._model_loader is None:
                from ..utils.model_loader import RecipeImageRetrievalFT
                cls._model_loader = RecipeImageRetrievalFT()
            
            success = cls._model_loader.load_system()
            if success:
                cls._is_loaded = True
                logger.info("Search system fully loaded and ready at startup!")
                return True
            else:
                logger.error("Failed to load search system at startup")
                return False
                
        except Exception as e:
            logger.error(f"Error during model initialization: {e}")
            return False
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self._is_loaded and SearchService._model_loader is not None
    
    async def search_recipes(self, search_domain: RecipeSearchDomain) -> RecipeSearchResultDomain:
        """
        Search for similar recipes based on uploaded image
        
        Args:
            search_domain: Domain model containing search parameters
            
        Returns:
            RecipeSearchResultDomain: Search results with recipes
        """
        start_time = time.time()
        
        try:
            # Model should already be loaded at startup
            if not self.is_model_loaded():
                error_msg = "Search model is not loaded. Please restart the application."
                logger.error(error_msg)
                return RecipeSearchResultDomain(
                    recipes=[],
                    total_results=0,
                    processing_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    success=False,
                    error_message=error_msg
                )
            
            try:
                image = Image.open(io.BytesIO(search_domain.image_data))
                logger.info(f"Image loaded: {image.size}, mode: {image.mode}")
            except Exception as e:
                error_msg = f"Invalid image format: {str(e)}"
                logger.error(error_msg)
                return RecipeSearchResultDomain(
                    recipes=[],
                    total_results=0,
                    processing_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    success=False,
                    error_message=error_msg
                )
            
            logger.info(f"Searching for top {search_domain.top_k} similar recipes...")
            similar_recipes = SearchService._model_loader.search_similar_recipes(
                image=image,
                top_k=search_domain.top_k
            )
            recipe_domains = []
            for recipe_data in similar_recipes:
                recipe_domain = RecipeResultDomain(
                    rank=recipe_data.get('rank', 0),
                    title=recipe_data.get('title', 'Unknown Recipe'),
                    image_path=recipe_data.get('image_path', ''),
                    ingredients=recipe_data.get('ingredients', ''),
                    instructions=recipe_data.get('instructions', ''),
                    similarity=recipe_data.get('similarity', 0.0)
                )
                recipe_domains.append(recipe_domain)
            
            processing_time = time.time() - start_time
            logger.info(f"Found {len(recipe_domains)} similar recipes in {processing_time:.2f}s")
            
            return RecipeSearchResultDomain(
                recipes=recipe_domains,
                total_results=len(recipe_domains),
                processing_time=processing_time,
                timestamp=datetime.now(),
                success=True
            )
            
        except Exception as e:
            error_msg = f"Error during recipe search: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return RecipeSearchResultDomain(
                recipes=[],
                total_results=0,
                processing_time=time.time() - start_time,
                timestamp=datetime.now(),
                success=False,
                error_message=error_msg
            ) 