from pydantic import BaseModel

def to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase"""
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

# Domain model (snake_case)
class TestDomain(BaseModel):
    """Core domain model for test with snake_case fields"""
    status: str
    message: str
    service: str
   
    class Config:
        populate_by_name = True
        alias_generator = to_camel

# API Response DTOs (camelCase)
class TestResponse(BaseModel):
    """Test response schema"""
    status: str
    message: str
    service: str
   
    @classmethod
    def from_domain(cls, domain: TestDomain) -> 'TestResponse':
        """Create response from domain model"""
        return cls(
            status=domain.status,
            message=domain.message,
            service=domain.service,
        ) 