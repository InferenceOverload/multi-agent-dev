"""Main application entry point for Hartford AI Agent System."""

import asyncio
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

from google.genai.adk.runtime import Runtime
from google.genai.adk import session

from agents.orchestrator import MainOrchestratorAgent
from services.memory_service import MemoryService
from config.config import get_config
from utils.logger import get_logger


# Load environment variables
load_dotenv()

logger = get_logger(__name__)
config = get_config()


class HartfordAIAgentSystem:
    """Main application class for Hartford AI Agent System."""
    
    def __init__(self):
        """Initialize the Hartford AI Agent System."""
        self.config = config
        self.memory_service = MemoryService(
            use_vertex_memory=config.google_genai_use_vertexai
        )
        self.orchestrator = MainOrchestratorAgent(
            memory_service=self.memory_service
        )
        self.runtime = None
        self.session = None
        
        logger.info("Hartford AI Agent System initialized")
    
    async def initialize(self):
        """Initialize runtime and session."""
        try:
            # Initialize ADK Runtime
            self.runtime = Runtime(
                project=config.google_cloud_project,
                location=config.google_cloud_location
            )
            
            # Create session
            self.session = session.Session()
            
            logger.info("Runtime and session initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize runtime: {e}")
            raise
    
    async def process_requirement(
        self,
        requirement: str,
        repository_url: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a requirement through the complete workflow.
        
        Args:
            requirement: Business requirement text
            repository_url: GitHub repository URL
            context: Optional additional context
            
        Returns:
            Workflow execution result
        """
        logger.info(f"Processing requirement: {requirement[:100]}...")
        
        # Prepare input for orchestrator
        input_data = {
            "requirement": requirement,
            "repository_url": repository_url,
            "context": context or {},
            "source": "api"
        }
        
        # Execute through orchestrator
        try:
            # Initialize session for this execution
            session_id = self.session.id if self.session else "default_session"
            
            # Initialize orchestrator context
            await self.orchestrator.initialize_context(session_id, input_data)
            
            # Process through complete workflow
            result = await self.orchestrator.process(input_data)
            
            logger.info(f"Workflow completed: {result.get('workflow_id')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def analyze_requirement(
        self,
        requirement: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze a requirement without full workflow.
        
        Args:
            requirement: Business requirement text
            context: Optional additional context
            
        Returns:
            Requirements analysis result
        """
        logger.info("Analyzing requirement...")
        
        session_id = self.session.id if self.session else "analysis_session"
        
        # Use requirements analyst directly
        req_analyst = self.orchestrator.requirements_analyst
        
        input_data = {
            "requirement_text": requirement,
            "context": context or {}
        }
        
        result = await req_analyst.execute(session_id, input_data)
        
        return result
    
    async def analyze_repository(
        self,
        repository_url: str,
        story_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a repository for story implementation.
        
        Args:
            repository_url: GitHub repository URL
            story_context: Story context for analysis
            
        Returns:
            Repository analysis result
        """
        logger.info(f"Analyzing repository: {repository_url}")
        
        session_id = self.session.id if self.session else "repo_session"
        
        # Use repository analyst directly
        repo_analyst = self.orchestrator.repository_analyst
        
        input_data = {
            "repo_url": repository_url,
            "story_context": story_context
        }
        
        result = await repo_analyst.execute(session_id, input_data)
        
        return result
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Workflow status
        """
        return await self.orchestrator.get_workflow_status(workflow_id)
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up resources...")
        
        # Cleanup old memories
        if self.memory_service:
            await self.memory_service.cleanup_old_memories(days=30)
        
        logger.info("Cleanup completed")


async def main():
    """Main entry point for the application."""
    # Initialize system
    system = HartfordAIAgentSystem()
    await system.initialize()
    
    # Example: Process a requirement
    example_requirement = """
    As a policyholder, I want to view my insurance policy details online
    so that I can access my coverage information anytime without calling customer service.
    The system should display policy number, coverage types, deductibles, premiums,
    and renewal dates. It should also show payment history and allow downloading
    policy documents in PDF format.
    """
    
    example_repo = "https://github.com/hartford-insurance/policy-management"
    
    # Process through complete workflow
    result = await system.process_requirement(
        requirement=example_requirement,
        repository_url=example_repo,
        context={
            "priority": "high",
            "target_sprint": "2024-Q1-Sprint-3",
            "team": "policy-team"
        }
    )
    
    # Print results
    if result.get("status") == "completed":
        print("\n✅ Workflow completed successfully!")
        print(f"Workflow ID: {result.get('workflow_id')}")
        
        summary = result.get("summary", {})
        print("\n📊 Summary:")
        print(f"  - Requirement: {summary.get('requirement')}")
        print(f"  - Work Item Type: {summary.get('work_item_type')}")
        print(f"  - Complexity: {summary.get('complexity')}")
        print(f"  - Domain: {summary.get('domain')}")
        print(f"  - Stories Created: {summary.get('stories_created')}")
        print(f"  - Total Story Points: {summary.get('total_story_points')}")
        print(f"  - Estimated Effort: {summary.get('estimated_effort')}")
        
        if result.get("stages", {}).get("pull_request", {}).get("pr_ready"):
            print("\n🔀 Pull Request Ready!")
            pr = result["stages"]["pull_request"]
            print(f"  - Branch: {pr.get('branch_name')}")
            print(f"  - Title: {pr.get('pr_title')}")
            print(f"  - Files Changed: {pr.get('files_changed')}")
    else:
        print(f"\n❌ Workflow failed: {result.get('error')}")
    
    # Cleanup
    await system.cleanup()


if __name__ == "__main__":
    # Run the main application
    asyncio.run(main())