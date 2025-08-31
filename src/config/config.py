"""Configuration management for Hartford AI Agents."""

import os
from typing import Dict, Any, Optional
from pydantic import Field
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings
from functools import lru_cache


class AgentConfig(BaseSettings):
    """Configuration for Hartford AI Agent System."""
    
    # Google Cloud Configuration
    google_cloud_project: str = Field(default="hartford-ai-agents")
    google_cloud_location: str = Field(default="us-central1")
    google_genai_use_vertexai: bool = Field(default=True)
    google_genai_api_key: Optional[str] = Field(default=None)
    
    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(default=None)
    
    # Rally Configuration
    rally_server: str = Field(default="https://rally1.rallydev.com")
    rally_username: str = Field(default="")
    rally_password: str = Field(default="")
    rally_workspace: str = Field(default="Hartford Insurance")
    rally_project: str = Field(default="Policy Management System")
    
    # GitHub Configuration
    github_token: str = Field(default="")
    github_org: str = Field(default="hartford-insurance")
    github_repos: str = Field(default="policy-management,claims-processing,customer-portal")
    
    # Agent Configuration
    agent_model: str = Field(default="gemini-2.0-flash")
    backup_model: str = Field(default="claude-3-sonnet")
    sandbox_type: str = Field(default="docker")
    max_context_length: int = Field(default=1000000)
    enable_parallel_execution: bool = Field(default=True)
    context_compression_ratio: float = Field(default=0.3)
    
    # Redis Configuration
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    
    # Database Configuration
    database_url: str = Field(default="postgresql://user:password@localhost:5432/hartford_agents")
    
    # Logging Configuration
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class InsuranceDomainConfig:
    """Insurance domain-specific configuration."""
    
    DOMAINS = {
        "policy_management": {
            "keywords": ["policy", "coverage", "premium", "deductible", "policyholder", "renewal"],
            "file_patterns": ["*Policy*", "*Coverage*", "*Premium*", "*Underwriting*"],
            "business_rules": ["rating", "underwriting", "renewal", "endorsement"],
            "rally_tags": ["policy", "underwriting", "renewal"]
        },
        "claims_processing": {
            "keywords": ["claim", "settlement", "adjuster", "loss", "damage", "investigation"],
            "file_patterns": ["*Claim*", "*Settlement*", "*Adjuster*", "*Loss*"],
            "business_rules": ["evaluation", "approval", "payment", "subrogation"],
            "rally_tags": ["claims", "settlement", "adjuster"]
        },
        "customer_management": {
            "keywords": ["customer", "agent", "broker", "service", "portal", "profile"],
            "file_patterns": ["*Customer*", "*Agent*", "*Portal*", "*Profile*"],
            "business_rules": ["authentication", "preferences", "communication", "onboarding"],
            "rally_tags": ["customer", "portal", "agent"]
        },
        "billing_finance": {
            "keywords": ["billing", "payment", "invoice", "commission", "accounting", "reconciliation"],
            "file_patterns": ["*Bill*", "*Payment*", "*Finance*", "*Invoice*"],
            "business_rules": ["calculation", "collection", "reporting", "reconciliation"],
            "rally_tags": ["billing", "payment", "finance"]
        },
        "regulatory_compliance": {
            "keywords": ["compliance", "regulatory", "audit", "HIPAA", "SOX", "privacy"],
            "file_patterns": ["*Compliance*", "*Regulatory*", "*Audit*", "*Privacy*"],
            "business_rules": ["validation", "reporting", "monitoring", "certification"],
            "rally_tags": ["compliance", "regulatory", "audit"]
        }
    }
    
    COMPLEXITY_FACTORS = {
        "low": {"story_points": [1, 2, 3], "time_estimate": "1-2 days"},
        "medium": {"story_points": [5, 8], "time_estimate": "3-5 days"},
        "high": {"story_points": [13, 21], "time_estimate": "1-2 weeks"},
        "very_high": {"story_points": [34, 55], "time_estimate": "2-4 weeks"}
    }
    
    RISK_LEVELS = {
        "critical": {"impact": "Business critical", "mitigation_priority": 1},
        "high": {"impact": "Significant impact", "mitigation_priority": 2},
        "medium": {"impact": "Moderate impact", "mitigation_priority": 3},
        "low": {"impact": "Minimal impact", "mitigation_priority": 4}
    }


class ADKRuntimeConfig:
    """ADK Runtime specific configuration."""
    
    AGENT_HIERARCHY = {
        "main_orchestrator": {
            "type": "coordinator",
            "sub_agents": [
                "requirements_analyst",
                "repository_analyst",
                "story_creator",
                "developer",
                "pr_manager"
            ]
        },
        "requirements_analyst": {
            "type": "llm_agent",
            "tools": ["rally_api", "domain_analyzer", "clarification_generator"]
        },
        "repository_analyst": {
            "type": "llm_agent",
            "tools": ["git_clone", "code_parser", "pattern_detector", "context_builder"]
        },
        "story_creator": {
            "type": "llm_agent",
            "tools": ["rally_api", "story_template", "sizing_calculator", "dependency_analyzer"]
        },
        "developer": {
            "type": "llm_agent",
            "tools": ["code_generator", "test_generator", "sandbox_executor", "linter"]
        },
        "pr_manager": {
            "type": "llm_agent",
            "tools": ["github_api", "pr_template", "documentation_generator", "rally_linker"]
        }
    }
    
    WORKFLOW_PATTERNS = {
        "sequential": ["requirements", "analysis", "planning", "implementation", "review"],
        "parallel": ["code_generation", "test_generation", "documentation"],
        "iterative": ["development", "testing", "refinement"]
    }


@lru_cache()
def get_config() -> AgentConfig:
    """Get cached configuration instance."""
    return AgentConfig()


@lru_cache()
def get_domain_config() -> InsuranceDomainConfig:
    """Get cached insurance domain configuration."""
    return InsuranceDomainConfig()


@lru_cache()
def get_runtime_config() -> ADKRuntimeConfig:
    """Get cached ADK runtime configuration."""
    return ADKRuntimeConfig()