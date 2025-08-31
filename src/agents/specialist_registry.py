"""Specialist Agent Registry - Plugin architecture for tech-specific agents."""

from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod
import importlib
import os


class SpecialistAgent(ABC):
    """Base class for all specialist agents."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name."""
        pass
    
    @property
    @abstractmethod
    def supported_types(self) -> List[str]:
        """List of codebase types this agent supports."""
        pass
    
    @abstractmethod
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze codebase with specialized knowledge."""
        pass
    
    @abstractmethod
    async def create_story(self, requirement: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create story with implementation details."""
        pass


class SpecialistRegistry:
    """Registry for managing specialist agents."""
    
    def __init__(self):
        self.agents: Dict[str, SpecialistAgent] = {}
        self.type_mapping: Dict[str, List[str]] = {}
        self._load_builtin_agents()
    
    def register(self, agent: SpecialistAgent):
        """Register a specialist agent."""
        self.agents[agent.name] = agent
        
        # Map supported types to agent
        for supported_type in agent.supported_types:
            if supported_type not in self.type_mapping:
                self.type_mapping[supported_type] = []
            self.type_mapping[supported_type].append(agent.name)
    
    def get_agent_for_type(self, codebase_type: str) -> Optional[SpecialistAgent]:
        """Get the best specialist agent for a codebase type."""
        agent_names = self.type_mapping.get(codebase_type, [])
        if agent_names:
            return self.agents.get(agent_names[0])
        return None
    
    def get_agents_for_stack(self, tech_stack: Dict[str, List[str]]) -> List[SpecialistAgent]:
        """Get all relevant specialist agents for a tech stack."""
        relevant_agents = []
        seen = set()
        
        # Check each category and tech in the stack
        for category, techs in tech_stack.items():
            for tech in techs:
                # Get agents for this tech
                agent_names = self.type_mapping.get(tech, [])
                for agent_name in agent_names:
                    if agent_name not in seen:
                        seen.add(agent_name)
                        agent = self.agents.get(agent_name)
                        if agent:
                            relevant_agents.append(agent)
        
        return relevant_agents
    
    def _load_builtin_agents(self):
        """Load built-in specialist agents."""
        # This will load our built-in agents
        # We'll implement them next
        builtin_agents = [
            "react_specialist",
            "node_specialist",
            "python_specialist",
            "data_specialist"
        ]
        
        for agent_module in builtin_agents:
            try:
                # Try to import the specialist module
                module = importlib.import_module(f"src.agents.specialists.{agent_module}")
                # Get the agent class (convention: ModuleNameAgent)
                class_name = ''.join(word.capitalize() for word in agent_module.split('_')) + 'Agent'
                if hasattr(module, class_name):
                    agent_class = getattr(module, class_name)
                    agent_instance = agent_class()
                    self.register(agent_instance)
            except ImportError:
                # Agent not implemented yet
                pass
            except Exception as e:
                print(f"Error loading specialist {agent_module}: {e}")
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered specialist agents."""
        return [
            {
                "name": agent.name,
                "supported_types": agent.supported_types
            }
            for agent in self.agents.values()
        ]


class SmartCoordinator:
    """Coordinates between discovery, specialists, and story creation."""
    
    def __init__(self, registry: SpecialistRegistry):
        self.registry = registry
    
    async def analyze_and_plan(
        self, 
        discovery_results: Dict[str, Any],
        requirement: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Coordinate analysis using discovery results and specialist agents.
        
        Args:
            discovery_results: Results from Discovery Agent
            requirement: Optional requirement for story creation
            
        Returns:
            Comprehensive analysis and implementation plan
        """
        result = {
            "discovery": discovery_results,
            "specialist_analysis": {},
            "implementation_plan": None,
            "recommended_approach": None
        }
        
        # Get relevant specialist agents
        tech_stack = discovery_results.get("tech_stack", {})
        specialists = self.registry.get_agents_for_stack(tech_stack)
        
        # Run specialist analysis
        for specialist in specialists:
            analysis = await specialist.analyze(discovery_results)
            result["specialist_analysis"][specialist.name] = analysis
        
        # If requirement provided, create implementation plan
        if requirement:
            # Use the most relevant specialist for story creation
            primary_language = discovery_results.get("primary_language")
            primary_specialist = self.registry.get_agent_for_type(primary_language)
            
            if primary_specialist:
                story = await primary_specialist.create_story(requirement, discovery_results)
                result["implementation_plan"] = story
            else:
                # Fallback to generic story creation
                result["implementation_plan"] = {
                    "title": requirement[:100],
                    "description": requirement,
                    "implementation_notes": "No specialist agent available for detailed planning"
                }
        
        # Determine recommended approach
        if specialists:
            result["recommended_approach"] = {
                "primary_specialist": specialists[0].name if specialists else None,
                "all_specialists": [s.name for s in specialists],
                "confidence": "high" if len(specialists) > 0 else "low"
            }
        
        return result