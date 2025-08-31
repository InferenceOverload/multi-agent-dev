"""Base models for Hartford AI Agent System."""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, validator


class StoryState(str, Enum):
    """Rally story states."""
    DEFINED = "Defined"
    IN_PROGRESS = "In-Progress"
    COMPLETED = "Completed"
    ACCEPTED = "Accepted"
    BLOCKED = "Blocked"


class WorkItemType(str, Enum):
    """Rally work item types."""
    EPIC = "Epic"
    FEATURE = "Feature"
    STORY = "Story"
    TASK = "Task"
    DEFECT = "Defect"


class ComplexityLevel(str, Enum):
    """Story complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ComplianceType(str, Enum):
    """Insurance compliance types."""
    HIPAA = "HIPAA"
    SOX = "SOX"
    STATE_REGULATIONS = "State Regulations"
    GDPR = "GDPR"
    CCPA = "CCPA"
    INTERNAL = "Internal"


class BusinessDomain(str, Enum):
    """Hartford business domains."""
    POLICY_MANAGEMENT = "policy_management"
    CLAIMS_PROCESSING = "claims_processing"
    CUSTOMER_MANAGEMENT = "customer_management"
    BILLING_FINANCE = "billing_finance"
    REGULATORY_COMPLIANCE = "regulatory_compliance"


class Requirement(BaseModel):
    """Business requirement model."""
    id: Optional[str] = None
    title: str
    description: str
    source: str = Field(default="user")
    business_value: Optional[str] = None
    acceptance_criteria: List[str] = Field(default_factory=list)
    domain: Optional[BusinessDomain] = None
    complexity: Optional[ComplexityLevel] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Story(BaseModel):
    """Rally story model."""
    id: Optional[str] = None
    formatted_id: Optional[str] = None
    title: str
    description: str
    user_story: str
    acceptance_criteria: List[str]
    story_points: Optional[int] = None
    state: StoryState = StoryState.DEFINED
    parent_id: Optional[str] = None
    owner: Optional[str] = None
    iteration: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    domain: Optional[BusinessDomain] = None
    complexity: Optional[ComplexityLevel] = None
    risk_level: Optional[RiskLevel] = None
    compliance_type: Optional[ComplianceType] = None
    implementation_plan: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    github_pr: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RepositoryInfo(BaseModel):
    """Repository information model."""
    url: str
    name: str
    organization: str
    default_branch: str = "main"
    language_stats: Dict[str, float] = Field(default_factory=dict)
    total_files: int = 0
    total_lines: int = 0
    architecture_type: Optional[str] = None
    framework: Optional[str] = None
    last_analyzed: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FileContext(BaseModel):
    """File context for code analysis."""
    path: str
    content: Optional[str] = None
    language: str
    size: int
    relevance_score: float = 0.0
    chunk_type: Optional[str] = None
    chunk_name: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    dependencies: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CodePattern(BaseModel):
    """Identified code pattern."""
    name: str
    type: str
    description: str
    file_path: str
    line_range: tuple[int, int]
    confidence: float
    usage_count: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ImplementationContext(BaseModel):
    """Context for story implementation."""
    story_id: str
    domain: BusinessDomain
    relevant_files: List[FileContext]
    patterns: List[CodePattern]
    suggested_approach: str
    estimated_complexity: ComplexityLevel
    dependencies: List[str]
    test_strategy: Optional[str] = None
    deployment_notes: Optional[str] = None
    risk_assessment: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PullRequest(BaseModel):
    """GitHub pull request model."""
    id: Optional[int] = None
    number: Optional[int] = None
    title: str
    description: str
    source_branch: str
    target_branch: str = "main"
    story_id: Optional[str] = None
    rally_link: Optional[str] = None
    files_changed: List[str] = Field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    reviewers: List[str] = Field(default_factory=list)
    labels: List[str] = Field(default_factory=list)
    status: str = "open"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentContext(BaseModel):
    """Context for agent execution."""
    session_id: str
    agent_name: str
    parent_agent: Optional[str] = None
    input_data: Dict[str, Any]
    state: Dict[str, Any] = Field(default_factory=dict)
    memory: Dict[str, Any] = Field(default_factory=dict)
    tools_used: List[str] = Field(default_factory=list)
    execution_time: Optional[float] = None
    status: str = "pending"
    error: Optional[str] = None
    output: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    """Workflow execution state."""
    workflow_id: str
    current_stage: str
    stages_completed: List[str] = Field(default_factory=list)
    stages_pending: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('stages_pending')
    def validate_stages(cls, v, values):
        """Ensure no duplicate stages."""
        if 'stages_completed' in values:
            completed = set(values['stages_completed'])
            return [s for s in v if s not in completed]
        return v


class MemoryEntry(BaseModel):
    """Memory storage entry."""
    id: Optional[str] = None
    session_id: str
    agent_name: str
    type: str  # "short_term", "long_term", "pattern", "success"
    key: str
    value: Any
    relevance_score: float = 1.0
    usage_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)