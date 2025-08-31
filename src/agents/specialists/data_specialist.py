"""Data Engineering Specialist Agent - Expert in data pipelines and analytics."""

from typing import Dict, Any, List
from ..specialist_registry import SpecialistAgent


class DataSpecialistAgent(SpecialistAgent):
    """Specialist agent for data engineering and analytics."""
    
    @property
    def name(self) -> str:
        return "data_specialist"
    
    @property
    def supported_types(self) -> List[str]:
        return ["sql", "spark", "airflow", "dbt", "data", "etl", "analytics"]
    
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data engineering codebase."""
        analysis = {
            "data_platform": None,
            "orchestration": None,
            "data_warehouse": None,
            "transformation_tool": None,
            "data_quality": None,
            "data_catalog": None
        }
        
        tech_stack = context.get("tech_stack", {})
        data_tech = tech_stack.get("data", [])
        
        # Detect data platform
        if "spark" in data_tech:
            analysis["data_platform"] = "Apache Spark"
        elif "databricks" in str(context).lower():
            analysis["data_platform"] = "Databricks"
        elif "snowflake" in str(context).lower():
            analysis["data_platform"] = "Snowflake"
        
        # Detect orchestration
        if "airflow" in data_tech:
            analysis["orchestration"] = "Apache Airflow"
        elif "prefect" in str(context).lower():
            analysis["orchestration"] = "Prefect"
        elif "dagster" in str(context).lower():
            analysis["orchestration"] = "Dagster"
        
        # Detect transformation tool
        if "dbt" in data_tech:
            analysis["transformation_tool"] = "dbt (data build tool)"
        elif "sql" in data_tech:
            analysis["transformation_tool"] = "SQL-based transformations"
        
        # Check for data quality
        if "great_expectations" in str(context).lower():
            analysis["data_quality"] = "Great Expectations"
        elif "deequ" in str(context).lower():
            analysis["data_quality"] = "AWS Deequ"
        
        return analysis
    
    async def create_story(self, requirement: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create data engineering-specific story."""
        
        requirement_lower = requirement.lower()
        
        # Determine story type
        story_type = "feature"
        if any(word in requirement_lower for word in ["pipeline", "etl", "ingestion", "extract"]):
            story_type = "data_pipeline"
        elif any(word in requirement_lower for word in ["transform", "aggregate", "join", "model"]):
            story_type = "transformation"
        elif any(word in requirement_lower for word in ["quality", "validation", "check", "monitor"]):
            story_type = "data_quality"
        elif any(word in requirement_lower for word in ["report", "dashboard", "visualization", "metric"]):
            story_type = "analytics"
        
        story = {
            "title": f"[Data] {requirement[:80]}",
            "type": story_type,
            "description": requirement,
            "acceptance_criteria": [],
            "implementation_details": {
                "pipelines_to_create": [],
                "transformations": [],
                "data_models": [],
                "quality_checks": [],
                "documentation": [],
                "monitoring": []
            },
            "technical_notes": [],
            "estimated_complexity": "medium",
            "data_considerations": {
                "source_systems": [],
                "data_volume": "unknown",
                "frequency": "daily",
                "sla": "6am completion"
            }
        }
        
        # Add specific implementation details based on story type
        if story_type == "data_pipeline":
            story["implementation_details"]["pipelines_to_create"] = [
                "Source data extraction logic",
                "Data validation and cleansing",
                "Load to staging area",
                "Error handling and retry logic"
            ]
            story["implementation_details"]["monitoring"] = [
                "Pipeline execution monitoring",
                "Data volume tracking",
                "Failure alerting"
            ]
            story["acceptance_criteria"] = [
                "Pipeline runs successfully end-to-end",
                "Data loaded matches source counts",
                "Failed records logged appropriately",
                "Pipeline completes within SLA",
                "Monitoring alerts configured"
            ]
            
        elif story_type == "transformation":
            story["implementation_details"]["transformations"] = [
                "Create staging models",
                "Build intermediate transformations",
                "Create final fact/dimension tables",
                "Add documentation to models"
            ]
            story["implementation_details"]["data_models"] = [
                "Define schema for new tables",
                "Create relationships between models",
                "Add appropriate indexes"
            ]
            story["acceptance_criteria"] = [
                "Transformations produce correct results",
                "Performance meets requirements",
                "Models are well-documented",
                "Tests pass for all models"
            ]
            
        elif story_type == "data_quality":
            story["implementation_details"]["quality_checks"] = [
                "Null value checks",
                "Referential integrity checks",
                "Business rule validations",
                "Anomaly detection"
            ]
            story["acceptance_criteria"] = [
                "Quality checks identify all known issues",
                "False positive rate < 5%",
                "Quality reports generated daily",
                "Alerts sent for critical issues"
            ]
            
        elif story_type == "analytics":
            story["implementation_details"]["data_models"] = [
                "Create analytical models",
                "Build aggregation tables",
                "Optimize for query performance"
            ]
            story["implementation_details"]["documentation"] = [
                "Document metrics definitions",
                "Create data dictionary",
                "Build user guide"
            ]
            story["acceptance_criteria"] = [
                "Metrics calculate correctly",
                "Query performance < 5 seconds",
                "Documentation complete and accurate",
                "Dashboard/reports functional"
            ]
        
        # Add data-specific technical notes
        story["technical_notes"] = [
            "Consider data volume and performance implications",
            "Implement idempotent operations",
            "Add data lineage tracking",
            "Follow data governance standards"
        ]
        
        # Estimate complexity based on data considerations
        if "large" in requirement_lower or "millions" in requirement_lower:
            story["estimated_complexity"] = "high"
            story["data_considerations"]["data_volume"] = "large (millions of records)"
        
        return story