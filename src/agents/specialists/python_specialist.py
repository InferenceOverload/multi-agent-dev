"""Python Specialist Agent - Expert in Python backend development."""

from typing import Dict, Any, List
from ..specialist_registry import SpecialistAgent


class PythonSpecialistAgent(SpecialistAgent):
    """Specialist agent for Python backend development."""
    
    @property
    def name(self) -> str:
        return "python_specialist"
    
    @property
    def supported_types(self) -> List[str]:
        return ["python", "django", "flask", "fastapi", "backend"]
    
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Python backend codebase."""
        analysis = {
            "framework": None,
            "database": None,
            "api_style": None,
            "authentication": None,
            "testing_framework": None,
            "deployment": None,
            "package_manager": None
        }
        
        frameworks = context.get("frameworks", [])
        tech_stack = context.get("tech_stack", {})
        
        # Detect framework
        if "Django" in frameworks:
            analysis["framework"] = "Django"
            analysis["database"] = "Likely PostgreSQL or SQLite"
            analysis["api_style"] = "Django REST Framework"
        elif "Flask" in frameworks:
            analysis["framework"] = "Flask"
            analysis["database"] = "SQLAlchemy ORM"
            analysis["api_style"] = "Flask-RESTful or custom"
        elif "FastAPI" in frameworks:
            analysis["framework"] = "FastAPI"
            analysis["database"] = "SQLAlchemy or Tortoise ORM"
            analysis["api_style"] = "OpenAPI/Swagger"
        
        # Detect package manager
        file_stats = context.get("file_statistics", {})
        if "requirements.txt" in str(context):
            analysis["package_manager"] = "pip (requirements.txt)"
        elif "Pipfile" in str(context):
            analysis["package_manager"] = "pipenv"
        elif "pyproject.toml" in str(context):
            analysis["package_manager"] = "poetry or pip"
        
        # Testing framework
        if context.get("has_tests"):
            analysis["testing_framework"] = "pytest or unittest"
        
        # Deployment
        if context.get("has_docker"):
            analysis["deployment"] = "Docker containerized"
        elif "serverless" in str(context).lower():
            analysis["deployment"] = "Serverless (AWS Lambda/GCP Functions)"
        
        return analysis
    
    async def create_story(self, requirement: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create Python backend-specific story."""
        
        requirement_lower = requirement.lower()
        
        # Determine story type
        story_type = "feature"
        if any(word in requirement_lower for word in ["api", "endpoint", "rest", "graphql"]):
            story_type = "api_endpoint"
        elif any(word in requirement_lower for word in ["database", "model", "migration", "schema"]):
            story_type = "database"
        elif any(word in requirement_lower for word in ["auth", "login", "permission", "security"]):
            story_type = "authentication"
        elif any(word in requirement_lower for word in ["async", "celery", "queue", "background"]):
            story_type = "async_task"
        
        story = {
            "title": f"[Backend] {requirement[:80]}",
            "type": story_type,
            "description": requirement,
            "acceptance_criteria": [],
            "implementation_details": {
                "models_to_create": [],
                "endpoints_to_create": [],
                "services_to_create": [],
                "migrations": [],
                "tests_to_write": [],
                "documentation": []
            },
            "technical_notes": [],
            "estimated_complexity": "medium"
        }
        
        # Add specific implementation details based on story type
        if story_type == "api_endpoint":
            story["implementation_details"]["endpoints_to_create"] = [
                "GET endpoint for listing",
                "POST endpoint for creation",
                "PUT/PATCH endpoint for updates",
                "DELETE endpoint for removal"
            ]
            story["implementation_details"]["services_to_create"] = [
                "Service layer for business logic",
                "Serializers/Schemas for validation",
                "Error handling middleware"
            ]
            story["acceptance_criteria"] = [
                "API returns correct status codes",
                "Request validation works properly",
                "Response format matches specification",
                "API documentation auto-generated",
                "Rate limiting applied if needed"
            ]
            
        elif story_type == "database":
            story["implementation_details"]["models_to_create"] = [
                "Database model with fields",
                "Model relationships and constraints",
                "Database indexes for performance"
            ]
            story["implementation_details"]["migrations"] = [
                "Create migration for new model",
                "Run migration in dev/staging",
                "Prepare rollback migration"
            ]
            story["acceptance_criteria"] = [
                "Model validates data correctly",
                "Relationships work as expected",
                "Queries perform efficiently",
                "Migration runs without errors"
            ]
            
        elif story_type == "authentication":
            story["implementation_details"]["services_to_create"] = [
                "Authentication service",
                "JWT/Session management",
                "Permission decorators/middleware"
            ]
            story["acceptance_criteria"] = [
                "Users can authenticate successfully",
                "Invalid credentials rejected properly",
                "Sessions/tokens expire correctly",
                "Permissions enforced on all endpoints"
            ]
            story["technical_notes"] = [
                "Follow OWASP security guidelines",
                "Implement proper password hashing",
                "Add rate limiting to auth endpoints"
            ]
        
        # Add testing requirements
        story["implementation_details"]["tests_to_write"] = [
            "Unit tests for business logic",
            "Integration tests for API endpoints",
            "Test edge cases and error conditions"
        ]
        
        # Estimate complexity based on implementation details
        total_items = sum(len(items) for items in story["implementation_details"].values())
        if total_items > 10:
            story["estimated_complexity"] = "high"
        elif total_items < 5:
            story["estimated_complexity"] = "low"
        
        return story