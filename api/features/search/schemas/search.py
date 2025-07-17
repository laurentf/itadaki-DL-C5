from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import re

def to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase"""
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def to_snake(camel_str: str) -> str:
    """Convert camelCase to snake_case"""
    return re.sub(r'(?<!^)(?=[A-Z])', '_', camel_str).lower()

# Domain model (snake_case)
class RecipeSearchDomain(BaseModel):
    """Core domain model for recipe search with snake_case fields"""
    image_data: bytes
    top_k: int = 3
    timestamp: Optional[datetime] = None
    processing_time: Optional[float] = None
    
    class Config:
        populate_by_name = True
        alias_generator = to_camel

class RecipeResultDomain(BaseModel):
    """Domain model for a single recipe result"""
    rank: int
    title: str
    image_path: str
    ingredients: str
    instructions: str
    similarity: float
    
    class Config:
        populate_by_name = True
        alias_generator = to_camel

class RecipeSearchResultDomain(BaseModel):
    """Domain model for search results"""
    recipes: List[RecipeResultDomain]
    total_results: int
    processing_time: float
    timestamp: datetime
    success: bool = True
    error_message: Optional[str] = None
    
    class Config:
        populate_by_name = True
        alias_generator = to_camel

# API Request DTOs (camelCase)
class RecipeSearchRequest(BaseModel):
    """Request schema for recipe image search"""
    topK: int = Field(default=3, ge=1, le=10, description="Number of top results to return")
    
    def to_domain(self, image_data: bytes) -> RecipeSearchDomain:
        """Convert request to domain model"""
        return RecipeSearchDomain(
            image_data=image_data,
            top_k=self.topK,
            timestamp=datetime.now()
        )

# API Response DTOs (camelCase)
class RecipeResultResponse(BaseModel):
    """Single recipe result for API responses"""
    rank: int
    title: str
    imagePath: str
    ingredients: str
    instructions: str
    similarity: float
    
    @classmethod
    def from_domain(cls, domain: RecipeResultDomain) -> 'RecipeResultResponse':
        """Create response from domain model"""
        return cls(
            rank=domain.rank,
            title=domain.title,
            imagePath=domain.image_path,
            ingredients=domain.ingredients,
            instructions=domain.instructions,
            similarity=domain.similarity
        )

class RecipeSearchResponse(BaseModel):
    """Recipe search response"""
    recipes: List[RecipeResultResponse]
    totalResults: int
    processingTime: float
    timestamp: datetime
    success: bool = True
    errorMessage: Optional[str] = None
    
    @classmethod
    def from_domain(cls, domain: RecipeSearchResultDomain) -> 'RecipeSearchResponse':
        """Create response from domain model"""
        return cls(
            recipes=[RecipeResultResponse.from_domain(recipe) for recipe in domain.recipes],
            totalResults=domain.total_results,
            processingTime=domain.processing_time,
            timestamp=domain.timestamp,
            success=domain.success,
            errorMessage=domain.error_message
        )

class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = False
    errorMessage: str
    timestamp: str  # ✅ FIX: str au lieu de datetime pour éviter JSON serialization error
    details: Optional[Dict[str, Any]] = None 