"""Base agent implementation using Google ADK."""

from typing import Dict, Any, Optional, List, AsyncIterator
from abc import ABC, abstractmethod
import asyncio
import logging
import os
from datetime import datetime

from google import genai
from google.genai import types
# Comment out ADK imports that aren't needed for now
# from google.adk.agents import LlmAgent, BaseAgent
# from google.adk.runtime import get_runtime

from ..models.base import AgentContext, MemoryEntry
from ..config.config import get_config
from ..services.memory_service import MemoryService
from ..utils.logger import get_logger


logger = get_logger(__name__)
config = get_config()


class HartfordBaseAgent(ABC):
    """Base class for all Hartford AI agents."""
    
    def __init__(
        self,
        name: str,
        description: str,
        tools: Optional[List[Any]] = None,
        parent_agent: Optional[str] = None,
        memory_service: Optional[MemoryService] = None
    ):
        """Initialize base agent.
        
        Args:
            name: Agent name
            description: Agent description
            tools: List of tools available to the agent
            parent_agent: Parent agent name if in hierarchy
            memory_service: Memory service for context persistence
        """
        self.name = name
        self.description = description
        self.tools = tools or []
        self.parent_agent = parent_agent
        self.memory_service = memory_service or MemoryService()
        self.context: Optional[AgentContext] = None
        self.logger = get_logger(f"agent.{name}")
        
        # Initialize Gemini client
        self.client = genai.Client(
            vertexai=config.google_genai_use_vertexai,
            project=config.google_cloud_project,
            location=config.google_cloud_location
        )
        
        # Create ADK LLM agent - commented out as we don't use ADK classes here
        # self.llm_agent = self._create_llm_agent()
        self.llm_agent = None
    
    def _create_llm_agent(self):
        """Create ADK LLM agent with tools - simplified for now."""
        # Not using ADK LlmAgent class for now
        return None
    
    async def initialize_context(
        self,
        session_id: str,
        input_data: Dict[str, Any]
    ) -> AgentContext:
        """Initialize agent context for execution.
        
        Args:
            session_id: Session identifier
            input_data: Input data for agent
            
        Returns:
            Initialized agent context
        """
        self.context = AgentContext(
            session_id=session_id,
            agent_name=self.name,
            parent_agent=self.parent_agent,
            input_data=input_data,
            status="initializing"
        )
        
        # Load relevant memory
        if self.memory_service:
            memory = await self.memory_service.retrieve_relevant_memory(
                session_id, 
                self.name,
                input_data
            )
            self.context.memory = memory
        
        self.logger.info(f"Initialized context for session {session_id}")
        return self.context
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return results.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processing results
        """
        pass
    
    async def execute(
        self,
        session_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute agent with context management.
        
        Args:
            session_id: Session identifier
            input_data: Input data for processing
            
        Returns:
            Execution results
        """
        try:
            # Initialize context
            await self.initialize_context(session_id, input_data)
            self.context.status = "processing"
            
            # Process input
            start_time = datetime.utcnow()
            result = await self.process(input_data)
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update context
            self.context.output = result
            self.context.execution_time = execution_time
            self.context.status = "completed"
            
            # Store in memory
            if self.memory_service:
                await self.memory_service.store_execution_result(
                    self.context
                )
            
            self.logger.info(
                f"Completed execution for session {session_id} "
                f"in {execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            if self.context:
                self.context.status = "failed"
                self.context.error = str(e)
            raise
    
    async def save_to_memory(
        self,
        key: str,
        value: Any,
        memory_type: str = "short_term"
    ) -> None:
        """Save data to memory service.
        
        Args:
            key: Memory key
            value: Value to store
            memory_type: Type of memory (short_term, long_term, pattern)
        """
        if not self.memory_service or not self.context:
            return
        
        entry = MemoryEntry(
            session_id=self.context.session_id,
            agent_name=self.name,
            type=memory_type,
            key=key,
            value=value
        )
        
        await self.memory_service.store(entry)
    
    async def retrieve_from_memory(
        self,
        key: str,
        memory_type: Optional[str] = None
    ) -> Optional[Any]:
        """Retrieve data from memory service.
        
        Args:
            key: Memory key
            memory_type: Optional memory type filter
            
        Returns:
            Retrieved value or None
        """
        if not self.memory_service or not self.context:
            return None
        
        return await self.memory_service.retrieve(
            self.context.session_id,
            self.name,
            key,
            memory_type
        )
    
    def add_tool(self, tool: Any) -> None:
        """Add a tool to the agent.
        
        Args:
            tool: Tool to add
        """
        self.tools.append(tool)
        # Recreate LLM agent with updated tools
        self.llm_agent = self._create_llm_agent()
    
    async def collaborate_with_agent(
        self,
        agent_name: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collaborate with another agent.
        
        Args:
            agent_name: Name of agent to collaborate with
            input_data: Input data for the agent
            
        Returns:
            Results from the collaborating agent
        """
        # This would be implemented to work with the orchestrator
        # to invoke another agent and get results
        self.logger.info(f"Collaborating with agent: {agent_name}")
        # Placeholder for actual implementation
        return {"status": "collaboration_pending", "agent": agent_name}


class WorkflowAgent(HartfordBaseAgent):
    """Base class for workflow-based agents."""
    
    def __init__(
        self,
        name: str,
        description: str,
        workflow_type: str = "sequential",
        sub_agents: Optional[List[HartfordBaseAgent]] = None,
        **kwargs
    ):
        """Initialize workflow agent.
        
        Args:
            name: Agent name
            description: Agent description
            workflow_type: Type of workflow (sequential, parallel, loop)
            sub_agents: List of sub-agents in workflow
            **kwargs: Additional arguments for base class
        """
        super().__init__(name, description, **kwargs)
        self.workflow_type = workflow_type
        self.sub_agents = sub_agents or []
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through workflow.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Workflow results
        """
        if self.workflow_type == "sequential":
            return await self._process_sequential(input_data)
        elif self.workflow_type == "parallel":
            return await self._process_parallel(input_data)
        elif self.workflow_type == "loop":
            return await self._process_loop(input_data)
        else:
            raise ValueError(f"Unknown workflow type: {self.workflow_type}")
    
    async def _process_sequential(
        self,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process sub-agents sequentially.
        
        Args:
            input_data: Input data
            
        Returns:
            Sequential processing results
        """
        results = []
        current_input = input_data
        
        for sub_agent in self.sub_agents:
            self.logger.info(f"Processing with sub-agent: {sub_agent.name}")
            result = await sub_agent.execute(
                self.context.session_id,
                current_input
            )
            results.append(result)
            # Use output as input for next agent
            current_input = result
        
        return {
            "workflow_type": "sequential",
            "results": results,
            "final_output": results[-1] if results else None
        }
    
    async def _process_parallel(
        self,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process sub-agents in parallel.
        
        Args:
            input_data: Input data
            
        Returns:
            Parallel processing results
        """
        tasks = [
            sub_agent.execute(self.context.session_id, input_data)
            for sub_agent in self.sub_agents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        errors = []
        success_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({
                    "agent": self.sub_agents[i].name,
                    "error": str(result)
                })
            else:
                success_results.append(result)
        
        return {
            "workflow_type": "parallel",
            "results": success_results,
            "errors": errors
        }
    
    async def _process_loop(
        self,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process sub-agents in a loop.
        
        Args:
            input_data: Input data
            
        Returns:
            Loop processing results
        """
        max_iterations = input_data.get("max_iterations", 10)
        termination_condition = input_data.get("termination_condition")
        
        results = []
        current_input = input_data
        
        for iteration in range(max_iterations):
            self.logger.info(f"Loop iteration {iteration + 1}")
            
            for sub_agent in self.sub_agents:
                result = await sub_agent.execute(
                    self.context.session_id,
                    current_input
                )
                results.append(result)
                current_input = result
            
            # Check termination condition
            if termination_condition and termination_condition(current_input):
                self.logger.info("Termination condition met")
                break
        
        return {
            "workflow_type": "loop",
            "iterations": len(results) // len(self.sub_agents),
            "results": results,
            "final_output": results[-1] if results else None
        }