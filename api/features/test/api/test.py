from fastapi import APIRouter, status
from features.test.schemas.test import TestResponse

router = APIRouter()

@router.get("/ping", response_model=TestResponse, status_code=status.HTTP_200_OK)
async def ping():
    """
    Simple ping endpoint
    """
    return TestResponse(
        status="ok", 
        message="🍜 pong",
        service="test"
    ) 