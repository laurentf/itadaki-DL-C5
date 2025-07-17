import time
from datetime import datetime
from typing import Dict, Any
from features.test.schemas.test import TestDomain
from core import config

class TestService:
    """Service for test operations"""
    
    _instance = None
    _start_time = None
    
    def __init__(self):
        if TestService._start_time is None:
            TestService._start_time = time.time()
    
    @classmethod
    def get_instance(cls) -> 'TestService':
        """Get singleton instance of TestService"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def get_health(self) -> Dict[str, Any]:
        """
        Get health information
        
        Returns:
            Dict: Health status and basic info
        """
        uptime = time.time() - self._start_time
        
        return {
            "status": "healthy",
            "uptime_seconds": uptime,
            "environment": config.ENVIRONMENT,
            "timestamp": datetime.now()
        }
    
    async def get_service_status(self) -> TestDomain:
        """
        Get detailed service status
        
        Returns:
            TestDomain: Service status information
        """
        uptime = time.time() - self._start_time
        
        return TestDomain(
            status="operational",
            message="Test service is running",
            service="test",
            uptime_seconds=uptime,
            environment=config.ENVIRONMENT
        ) 