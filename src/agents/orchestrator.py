"""Main Orchestrator Agent using ADK multi-agent patterns."""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
# Comment out ADK imports not needed
# from google.adk import llm_agent, agent_tool, workflow_agent

from ..agents.base_agent import HartfordBaseAgent, WorkflowAgent
from ..agents.requirements_analyst import RequirementsAnalystAgent
from ..agents.repository_analyst import RepositoryAnalystAgent
from ..models.base import WorkflowState, WorkItemType
from ..services.memory_service import MemoryService
from ..config.config import get_config, get_runtime_config
from ..utils.logger import get_logger


logger = get_logger(__name__)


class MainOrchestratorAgent(WorkflowAgent):
    """Main orchestrator that coordinates all Hartford AI agents."""
    
    def __init__(self, memory_service: Optional[MemoryService] = None):
        """Initialize Main Orchestrator Agent.
        
        Args:
            memory_service: Memory service for context persistence
        """
        super().__init__(
            name="MainOrchestrator",
            description=(
                "Orchestrates the entire Hartford Insurance AI development workflow "
                "from requirements analysis through pull request creation. "
                "Coordinates sub-agents and manages workflow state."
            ),
            workflow_type="sequential",
            memory_service=memory_service or MemoryService()
        )
        
        self.config = get_config()
        self.runtime_config = get_runtime_config()
        
        # Initialize sub-agents
        self._initialize_sub_agents()
        
        # Workflow state tracking
        self.workflow_states = {}
    
    def _initialize_sub_agents(self):
        """Initialize all sub-agents for the workflow."""
        # Requirements Analyst
        self.requirements_analyst = RequirementsAnalystAgent(
            memory_service=self.memory_service
        )
        
        # Repository Analyst
        self.repository_analyst = RepositoryAnalystAgent(
            memory_service=self.memory_service
        )
        
        # Story Creator (placeholder - would be implemented similarly)
        # self.story_creator = StoryCreatorAgent(memory_service=self.memory_service)
        
        # Developer (placeholder)
        # self.developer = DeveloperAgent(memory_service=self.memory_service)
        
        # PR Manager (placeholder)
        # self.pr_manager = PullRequestAgent(memory_service=self.memory_service)
        
        # Set sub-agents for workflow
        self.sub_agents = [
            self.requirements_analyst,
            self.repository_analyst,
            # self.story_creator,
            # self.developer,
            # self.pr_manager
        ]
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process complete workflow from requirement to implementation.
        
        Args:
            input_data: Contains 'requirement', 'repository_url', optional 'config'
            
        Returns:
            Complete workflow results
        """
        # Initialize workflow state
        workflow_id = self._generate_workflow_id()
        
        # Determine workflow type based on input
        is_enhancement = self._detect_enhancement_or_bugfix(input_data)
        has_repository = bool(input_data.get("repository_url"))
        
        # Adjust workflow stages based on type
        if is_enhancement and has_repository:
            initial_stage = "repository_analysis"
            stages = [
                "repository_analysis",
                "context_aware_requirements",
                "story_creation",
                "development",
                "pull_request"
            ]
        else:
            initial_stage = "requirements_analysis"
            stages = [
                "requirements_analysis",
                "repository_analysis",
                "story_creation",
                "development",
                "pull_request"
            ]
        
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            current_stage=initial_stage,
            stages_pending=stages,
            context=input_data
        )
        
        self.workflow_states[workflow_id] = workflow_state
        
        try:
            # Context-aware workflow execution
            if is_enhancement and has_repository:
                # For enhancements/bugfixes: Analyze repo FIRST
                logger.info(f"[{workflow_id}] Enhancement detected - Starting with repository analysis")
                
                # Stage 1: Repository Analysis FIRST
                logger.info(f"[{workflow_id}] Stage 1: Repository Analysis")
                repo_result = await self._process_repository_analysis(
                    workflow_state, None, input_data
                )
                
                # Stage 2: Context-Aware Requirements Analysis
                logger.info(f"[{workflow_id}] Stage 2: Context-Aware Requirements Analysis")
                requirements_result = await self._process_requirements_with_context(
                    workflow_state, input_data, repo_result
                )
            else:
                # For new features: Traditional flow
                logger.info(f"[{workflow_id}] New feature - Traditional workflow")
                
                # Stage 1: Requirements Analysis
                logger.info(f"[{workflow_id}] Stage 1: Requirements Analysis")
                requirements_result = await self._process_requirements(
                    workflow_state, input_data
                )
                
                # Stage 2: Repository Analysis (if repo provided)
                if has_repository:
                    logger.info(f"[{workflow_id}] Stage 2: Repository Analysis")
                    repo_result = await self._process_repository_analysis(
                        workflow_state, requirements_result, input_data
                    )
                else:
                    repo_result = None
            
            # Stage 3: Story Creation
            logger.info(f"[{workflow_id}] Stage 3: Story Creation")
            story_result = await self._process_story_creation(
                workflow_state, requirements_result, repo_result
            )
            
            # Stage 4: Development
            logger.info(f"[{workflow_id}] Stage 4: Development")
            dev_result = await self._process_development(
                workflow_state, story_result, repo_result
            )
            
            # Stage 5: Pull Request
            logger.info(f"[{workflow_id}] Stage 5: Pull Request Creation")
            pr_result = await self._process_pull_request(
                workflow_state, dev_result, story_result
            )
            
            # Mark workflow complete
            workflow_state.completed_at = datetime.utcnow()
            workflow_state.current_stage = "completed"
            
            # Prepare final result
            result = {
                "workflow_id": workflow_id,
                "status": "completed",
                "stages": {
                    "requirements": requirements_result,
                    "repository": repo_result,
                    "story": story_result,
                    "development": dev_result,
                    "pull_request": pr_result
                },
                "summary": self._generate_workflow_summary(
                    requirements_result, story_result, pr_result
                ),
                "execution_time": (
                    workflow_state.completed_at - workflow_state.started_at
                ).total_seconds()
            }
            
            # Store in long-term memory for learning
            await self.save_to_memory(
                f"workflow_{workflow_id}",
                result,
                "long_term"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[{workflow_id}] Workflow failed: {e}")
            workflow_state.errors.append({
                "stage": workflow_state.current_stage,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "completed_stages": workflow_state.stages_completed,
                "failed_stage": workflow_state.current_stage
            }
    
    async def _process_requirements(
        self,
        workflow_state: WorkflowState,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process requirements analysis stage.
        
        Args:
            workflow_state: Current workflow state
            input_data: Input data
            
        Returns:
            Requirements analysis result
        """
        workflow_state.current_stage = "requirements_analysis"
        
        # Prepare input for requirements analyst
        req_input = {
            "requirement_text": input_data.get("requirement", ""),
            "context": input_data.get("context", {}),
            "source": input_data.get("source", "user")
        }
        
        # Execute requirements analysis
        result = await self.requirements_analyst.execute(
            self.context.session_id,
            req_input
        )
        
        # Update workflow state
        workflow_state.stages_completed.append("requirements_analysis")
        workflow_state.context["requirements_result"] = result
        
        # Check if clarifications needed
        if result.get("clarifications_needed"):
            logger.info(f"Clarifications needed: {len(result['clarifications_needed'])}")
            # In production, would interact with user for clarifications
            # For now, proceed with available information
        
        return result
    
    def _detect_enhancement_or_bugfix(self, input_data: Dict[str, Any]) -> bool:
        """Detect if requirement is an enhancement or bugfix.
        
        Args:
            input_data: Input data
            
        Returns:
            True if enhancement/bugfix, False if new feature
        """
        requirement = input_data.get("requirement", "").lower()
        requirement_type = input_data.get("type", "").lower()
        
        # Explicit type check
        if requirement_type in ["enhancement", "bugfix", "bug", "fix", "improve", "update"]:
            return True
        
        # Keyword detection in requirement
        enhancement_keywords = [
            "fix", "bug", "issue", "problem", "error", "broken",
            "improve", "enhance", "update", "modify", "change",
            "refactor", "optimize", "performance", "speed up",
            "correct", "repair", "patch", "resolve"
        ]
        
        for keyword in enhancement_keywords:
            if keyword in requirement:
                return True
        
        return False
    
    async def _process_requirements_with_context(
        self,
        workflow_state: WorkflowState,
        input_data: Dict[str, Any],
        repo_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process requirements with repository context.
        
        Args:
            workflow_state: Current workflow state
            input_data: Input data
            repo_result: Repository analysis result
            
        Returns:
            Context-aware requirements analysis
        """
        workflow_state.current_stage = "context_aware_requirements"
        
        # Extract relevant context from repo analysis
        repo_context = {
            "architecture": repo_result.get("repository_info", {}).get("architecture_type"),
            "framework": repo_result.get("repository_info", {}).get("framework"),
            "patterns": repo_result.get("analysis", {}).get("patterns", []),
            "business_domains": repo_result.get("analysis", {}).get("business_domains", {}),
            "existing_components": repo_result.get("analysis", {}).get("relevant_files", []),
            "complexity_metrics": repo_result.get("analysis", {}).get("complexity_metrics", {}),
            "implementation_context": repo_result.get("implementation_context", {})
        }
        
        # Prepare input for requirements analyst with context
        req_input = {
            "requirement_text": input_data.get("requirement", ""),
            "context": input_data.get("context", {}),
            "repository_context": repo_context,
            "is_enhancement": True,
            "source": input_data.get("source", "user")
        }
        
        # Execute requirements analysis with repository knowledge
        result = await self.requirements_analyst.execute(
            self.context.session_id,
            req_input
        )
        
        # Enhance result with repository-specific insights
        result["repository_insights"] = {
            "affected_components": self._identify_affected_components(
                result, repo_result
            ),
            "implementation_approach": repo_result.get("implementation_context", {}).get(
                "suggested_approach", "Follow existing patterns"
            ),
            "existing_patterns": repo_context["patterns"][:5] if repo_context["patterns"] else [],
            "estimated_impact": self._estimate_change_impact(result, repo_result)
        }
        
        # Update workflow state
        workflow_state.stages_completed.append("context_aware_requirements")
        workflow_state.context["requirements_result"] = result
        
        return result
    
    async def _process_repository_analysis(
        self,
        workflow_state: WorkflowState,
        requirements_result: Optional[Dict[str, Any]],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process repository analysis stage.
        
        Args:
            workflow_state: Current workflow state
            requirements_result: Requirements analysis result (optional)
            input_data: Original input data
            
        Returns:
            Repository analysis result
        """
        workflow_state.current_stage = "repository_analysis"
        
        # Prepare input for repository analyst
        repo_input = {
            "repo_url": input_data.get("repository_url", ""),
            "branch": input_data.get("branch", "main")
        }
        
        # Add story context if requirements were analyzed first
        if requirements_result:
            repo_input["story_context"] = {
                "title": requirements_result["requirement"]["title"],
                "description": requirements_result["requirement"]["description"],
                "domain": requirements_result["analysis"].get("domain"),
                "complexity": requirements_result["analysis"].get("complexity"),
                "technical_areas": requirements_result["analysis"].get("technical_areas", [])
            }
        else:
            # For enhancement flow, use raw requirement as context
            repo_input["story_context"] = {
                "title": input_data.get("requirement", "")[:100],
                "description": input_data.get("requirement", "")
            }
        
        # Execute repository analysis
        result = await self.repository_analyst.execute(
            self.context.session_id,
            repo_input
        )
        
        # Update workflow state
        workflow_state.stages_completed.append("repository_analysis")
        workflow_state.context["repository_result"] = result
        
        return result
    
    def _identify_affected_components(
        self,
        requirements_result: Dict[str, Any],
        repo_result: Dict[str, Any]
    ) -> List[str]:
        """Identify components affected by requirement.
        
        Args:
            requirements_result: Requirements analysis
            repo_result: Repository analysis
            
        Returns:
            List of affected components
        """
        affected = []
        
        # Get technical areas from requirements
        tech_areas = requirements_result.get("analysis", {}).get("technical_areas", [])
        
        # Map to actual components in repo
        relevant_files = repo_result.get("implementation_context", {}).get("relevant_files", [])
        
        components = set()
        for file_ctx in relevant_files[:10]:
            path = file_ctx.get("path", "")
            if "/" in path:
                component = path.split("/")[0]
                components.add(component)
        
        return list(components)
    
    def _estimate_change_impact(
        self,
        requirements_result: Dict[str, Any],
        repo_result: Dict[str, Any]
    ) -> str:
        """Estimate impact of changes.
        
        Args:
            requirements_result: Requirements analysis
            repo_result: Repository analysis
            
        Returns:
            Impact assessment
        """
        complexity = requirements_result.get("analysis", {}).get("complexity", "medium")
        files_count = len(repo_result.get("implementation_context", {}).get("relevant_files", []))
        
        if complexity in ["very_high", "high"] or files_count > 10:
            return "high - Multiple components affected, careful testing required"
        elif complexity == "medium" or files_count > 5:
            return "medium - Moderate changes required, standard testing"
        else:
            return "low - Localized changes, minimal impact"
    
    async def _process_story_creation(
        self,
        workflow_state: WorkflowState,
        requirements_result: Dict[str, Any],
        repo_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process story creation stage.
        
        Args:
            workflow_state: Current workflow state
            requirements_result: Requirements analysis result
            repo_result: Repository analysis result
            
        Returns:
            Story creation result
        """
        workflow_state.current_stage = "story_creation"
        
        # For now, create a mock story result
        # In production, would use StoryCreatorAgent
        work_item_type = requirements_result.get("work_item_type", "Story")
        
        story_result = {
            "work_item_type": work_item_type,
            "stories": [],
            "implementation_plans": []
        }
        
        if work_item_type == WorkItemType.STORY.value:
            # Single story
            story = {
                "title": requirements_result["requirement"]["title"],
                "description": requirements_result["requirement"]["description"],
                "user_story": self._generate_user_story(requirements_result),
                "acceptance_criteria": self._generate_acceptance_criteria(requirements_result),
                "story_points": self._estimate_story_points(
                    requirements_result["analysis"]["complexity"]
                ),
                "implementation_plan": repo_result.get("implementation_context", {})
            }
            story_result["stories"].append(story)
            
        elif work_item_type == WorkItemType.FEATURE.value:
            # Multiple stories for feature
            stories_count = requirements_result["analysis"]["scope"]["estimated_stories"][0]
            for i in range(min(stories_count, 3)):  # Limit to 3 for demo
                story = {
                    "title": f"{requirements_result['requirement']['title']} - Part {i+1}",
                    "description": f"Implementation part {i+1}",
                    "story_points": 5,
                    "implementation_plan": {}
                }
                story_result["stories"].append(story)
        
        # Update workflow state
        workflow_state.stages_completed.append("story_creation")
        workflow_state.context["story_result"] = story_result
        
        return story_result
    
    async def _process_development(
        self,
        workflow_state: WorkflowState,
        story_result: Dict[str, Any],
        repo_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process development stage.
        
        Args:
            workflow_state: Current workflow state
            story_result: Story creation result
            repo_result: Repository analysis result
            
        Returns:
            Development result
        """
        workflow_state.current_stage = "development"
        
        # For now, create a mock development result
        # In production, would use DeveloperAgent
        dev_result = {
            "status": "ready_for_implementation",
            "implementation_approach": repo_result.get("implementation_context", {}).get(
                "suggested_approach", "Follow existing patterns"
            ),
            "files_to_modify": [],
            "new_files": [],
            "tests_required": repo_result.get("implementation_context", {}).get(
                "test_strategy", "Unit and integration tests"
            ),
            "estimated_effort": story_result["stories"][0].get("story_points", 5) if story_result["stories"] else 5
        }
        
        # Extract files from repository analysis
        if "implementation_context" in repo_result:
            relevant_files = repo_result["implementation_context"].get("relevant_files", [])
            for file_context in relevant_files[:5]:  # Top 5 files
                dev_result["files_to_modify"].append({
                    "path": file_context.get("path"),
                    "changes_needed": "Implement new functionality",
                    "relevance": file_context.get("relevance_score", 0)
                })
        
        # Update workflow state
        workflow_state.stages_completed.append("development")
        workflow_state.context["development_result"] = dev_result
        
        return dev_result
    
    async def _process_pull_request(
        self,
        workflow_state: WorkflowState,
        dev_result: Dict[str, Any],
        story_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process pull request creation stage.
        
        Args:
            workflow_state: Current workflow state
            dev_result: Development result
            story_result: Story result
            
        Returns:
            Pull request result
        """
        workflow_state.current_stage = "pull_request"
        
        # For now, create a mock PR result
        # In production, would use PullRequestAgent
        pr_result = {
            "status": "ready_to_create",
            "pr_title": f"feat: {story_result['stories'][0]['title']}" if story_result["stories"] else "feat: New feature",
            "pr_description": self._generate_pr_description(
                story_result, dev_result
            ),
            "branch_name": self._generate_branch_name(story_result),
            "files_changed": len(dev_result.get("files_to_modify", [])),
            "reviewers": ["team-lead", "senior-dev"],
            "labels": ["enhancement", "insurance-domain"],
            "rally_link": f"https://rally1.rallydev.com/story/{workflow_state.workflow_id}"
        }
        
        # Update workflow state
        workflow_state.stages_completed.append("pull_request")
        workflow_state.context["pr_result"] = pr_result
        
        return pr_result
    
    def _generate_workflow_id(self) -> str:
        """Generate unique workflow ID.
        
        Returns:
            Workflow ID
        """
        import uuid
        return f"wf_{uuid.uuid4().hex[:12]}"
    
    def _generate_user_story(self, requirements_result: Dict[str, Any]) -> str:
        """Generate user story format.
        
        Args:
            requirements_result: Requirements analysis result
            
        Returns:
            User story
        """
        user_types = requirements_result["analysis"].get("user_types", ["user"])
        primary_user = user_types[0] if user_types else "user"
        
        return (
            f"As a {primary_user}, "
            f"I want {requirements_result['requirement']['title'].lower()}, "
            f"so that I can achieve the intended business value"
        )
    
    def _generate_acceptance_criteria(
        self,
        requirements_result: Dict[str, Any]
    ) -> List[str]:
        """Generate acceptance criteria.
        
        Args:
            requirements_result: Requirements analysis result
            
        Returns:
            List of acceptance criteria
        """
        criteria = []
        
        # Basic criteria based on requirement
        criteria.append(
            f"Given a {requirements_result['analysis'].get('domain', 'system')} context, "
            f"When the feature is implemented, "
            f"Then it should meet the specified requirements"
        )
        
        # Add compliance criteria if needed
        if requirements_result.get("compliance_requirements"):
            criteria.append(
                f"Given compliance requirements, "
                f"When the feature is used, "
                f"Then it should comply with {', '.join(requirements_result['compliance_requirements'])}"
            )
        
        # Add performance criteria for complex features
        if requirements_result["analysis"].get("complexity") in ["high", "very_high"]:
            criteria.append(
                "Given performance requirements, "
                "When under normal load, "
                "Then response time should be under 2 seconds"
            )
        
        return criteria
    
    def _estimate_story_points(self, complexity: str) -> int:
        """Estimate story points based on complexity.
        
        Args:
            complexity: Complexity level
            
        Returns:
            Story points (Fibonacci)
        """
        points_map = {
            "low": 3,
            "medium": 5,
            "high": 8,
            "very_high": 13
        }
        return points_map.get(complexity, 5)
    
    def _generate_pr_description(
        self,
        story_result: Dict[str, Any],
        dev_result: Dict[str, Any]
    ) -> str:
        """Generate PR description.
        
        Args:
            story_result: Story result
            dev_result: Development result
            
        Returns:
            PR description
        """
        story = story_result["stories"][0] if story_result["stories"] else {}
        
        description = f"""## Summary
{story.get('description', 'Implementation of new feature')}

## Changes
- Implementation approach: {dev_result.get('implementation_approach', 'Standard implementation')}
- Files modified: {len(dev_result.get('files_to_modify', []))}
- New files: {len(dev_result.get('new_files', []))}

## Testing
{dev_result.get('tests_required', 'Unit and integration tests included')}

## Story Points
{story.get('story_points', 5)}

## Rally Link
[View in Rally](https://rally1.rallydev.com)

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Security review completed
- [ ] Performance impact assessed
"""
        return description
    
    def _generate_branch_name(self, story_result: Dict[str, Any]) -> str:
        """Generate branch name.
        
        Args:
            story_result: Story result
            
        Returns:
            Branch name
        """
        if story_result["stories"]:
            title = story_result["stories"][0]["title"]
            # Clean title for branch name
            branch_name = title.lower()
            branch_name = branch_name.replace(" ", "-")
            branch_name = ''.join(c for c in branch_name if c.isalnum() or c == '-')
            return f"feature/{branch_name[:50]}"
        return f"feature/new-feature"
    
    def _generate_workflow_summary(
        self,
        requirements_result: Dict[str, Any],
        story_result: Dict[str, Any],
        pr_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate workflow summary.
        
        Args:
            requirements_result: Requirements result
            story_result: Story result
            pr_result: PR result
            
        Returns:
            Workflow summary
        """
        return {
            "requirement": requirements_result["requirement"]["title"],
            "work_item_type": requirements_result["work_item_type"],
            "complexity": requirements_result["analysis"]["complexity"],
            "domain": requirements_result["analysis"].get("domain"),
            "stories_created": len(story_result.get("stories", [])),
            "total_story_points": sum(
                s.get("story_points", 0) for s in story_result.get("stories", [])
            ),
            "pr_ready": pr_result["status"] == "ready_to_create",
            "estimated_effort": requirements_result["analysis"]["effort_estimate"]["time_estimate"]
        }
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Workflow status or None
        """
        if workflow_id in self.workflow_states:
            state = self.workflow_states[workflow_id]
            return {
                "workflow_id": workflow_id,
                "current_stage": state.current_stage,
                "completed_stages": state.stages_completed,
                "pending_stages": state.stages_pending,
                "errors": state.errors,
                "started_at": state.started_at.isoformat(),
                "completed_at": state.completed_at.isoformat() if state.completed_at else None
            }
        return None
    
    async def handle_interactive_clarification(
        self,
        workflow_id: str,
        clarifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle interactive clarifications from user.
        
        Args:
            workflow_id: Workflow ID
            clarifications: User-provided clarifications
            
        Returns:
            Updated workflow result
        """
        if workflow_id not in self.workflow_states:
            return {"error": "Workflow not found"}
        
        state = self.workflow_states[workflow_id]
        
        # Update context with clarifications
        state.context["clarifications"] = clarifications
        
        # Resume workflow from current stage
        # This would be implemented based on specific stage requirements
        
        return {"status": "clarifications_received", "workflow_id": workflow_id}