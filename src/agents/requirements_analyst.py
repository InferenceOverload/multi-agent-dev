"""Requirements Analysis Agent for Hartford Insurance AI System."""

from typing import Dict, Any, List, Optional, Tuple
import re
from datetime import datetime

from ..agents.base_agent import HartfordBaseAgent
from ..models.base import (
    Requirement, WorkItemType, ComplexityLevel, 
    BusinessDomain, ComplianceType
)
from ..config.config import get_domain_config
from ..tools.clarification_tool import ClarificationTool
from ..tools.domain_analyzer import DomainAnalyzer
from ..utils.logger import get_logger


logger = get_logger(__name__)


class RequirementsAnalystAgent(HartfordBaseAgent):
    """Agent for analyzing and categorizing business requirements."""
    
    COMPLEXITY_INDICATORS = {
        "low": {
            "keywords": ["simple", "basic", "minor", "small", "fix", "update"],
            "scope_indicators": ["single", "one", "specific"],
            "effort_days": (1, 2)
        },
        "medium": {
            "keywords": ["moderate", "standard", "typical", "normal", "enhance"],
            "scope_indicators": ["multiple", "several", "few"],
            "effort_days": (3, 5)
        },
        "high": {
            "keywords": ["complex", "significant", "major", "comprehensive", "integrate"],
            "scope_indicators": ["many", "various", "across"],
            "effort_days": (6, 10)
        },
        "very_high": {
            "keywords": ["highly complex", "enterprise", "critical", "transform", "overhaul"],
            "scope_indicators": ["all", "entire", "complete", "full"],
            "effort_days": (11, None)
        }
    }
    
    def __init__(self, **kwargs):
        """Initialize Requirements Analyst Agent."""
        super().__init__(
            name="RequirementsAnalyst",
            description=(
                "Analyzes business requirements to determine complexity, "
                "categorization (EPIC/Feature/Story), and generates clarifying questions. "
                "Specializes in insurance domain requirements."
            ),
            tools=[
                ClarificationTool(),
                DomainAnalyzer()
            ],
            **kwargs
        )
        self.domain_config = get_domain_config()
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process requirement and return analysis.
        
        Args:
            input_data: Contains 'requirement_text', optional 'context', and 'repository_context'
            
        Returns:
            Analysis results including categorization and clarifications
        """
        requirement_text = input_data.get("requirement_text", "")
        context = input_data.get("context", {})
        repository_context = input_data.get("repository_context", {})
        is_enhancement = input_data.get("is_enhancement", False)
        
        # Create requirement object
        requirement = Requirement(
            title=self._extract_title(requirement_text),
            description=requirement_text,
            source=input_data.get("source", "user")
        )
        
        # Analyze requirement (with repository context if available)
        if repository_context:
            analysis_results = await self._analyze_requirement_with_repo_context(
                requirement, context, repository_context
            )
        else:
            analysis_results = await self._analyze_requirement(requirement, context)
        
        # Determine work item type (considering repo context)
        work_item_type = self._categorize_requirement_with_context(
            requirement, 
            analysis_results,
            repository_context,
            is_enhancement
        )
        
        # Generate clarifying questions if needed
        clarifications = await self._generate_clarifications_with_context(
            requirement,
            analysis_results,
            work_item_type,
            repository_context
        )
        
        # Identify compliance requirements
        compliance_types = self._identify_compliance_requirements(requirement)
        
        # Build response
        response = {
            "requirement": requirement.dict(),
            "work_item_type": work_item_type.value,
            "analysis": analysis_results,
            "clarifications_needed": clarifications,
            "compliance_requirements": compliance_types,
            "recommended_action": self._get_recommended_action_with_context(
                work_item_type,
                analysis_results,
                repository_context
            )
        }
        
        # Add repository-aware insights if context available
        if repository_context:
            response["repository_aware_insights"] = self._generate_repo_aware_insights(
                requirement, analysis_results, repository_context
            )
        
        # Store in memory for future reference
        await self.save_to_memory(
            f"requirement_{requirement.id}",
            response,
            "long_term"
        )
        
        return response
    
    def _extract_title(self, requirement_text: str) -> str:
        """Extract title from requirement text.
        
        Args:
            requirement_text: Full requirement text
            
        Returns:
            Extracted or generated title
        """
        # Try to extract first sentence or line
        lines = requirement_text.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # Limit to reasonable title length
            if len(first_line) <= 100:
                return first_line
            else:
                # Truncate and add ellipsis
                return first_line[:97] + "..."
        return "Untitled Requirement"
    
    async def _analyze_requirement(
        self,
        requirement: Requirement,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze requirement for various attributes.
        
        Args:
            requirement: Requirement object
            context: Additional context
            
        Returns:
            Analysis results
        """
        text_lower = requirement.description.lower()
        
        # Identify business domain
        domain = self._identify_business_domain(requirement)
        
        # Assess complexity
        complexity = self._assess_complexity(requirement, context)
        
        # Estimate scope
        scope = self._estimate_scope(requirement)
        
        # Identify user types
        user_types = self._identify_user_types(requirement)
        
        # Estimate effort
        effort_estimate = self._estimate_effort(complexity, scope)
        
        # Identify dependencies
        dependencies = self._identify_dependencies(requirement, context)
        
        # Risk assessment
        risk_level = self._assess_risk(requirement, complexity, dependencies)
        
        return {
            "domain": domain.value if domain else None,
            "complexity": complexity.value,
            "scope": scope,
            "user_types": user_types,
            "effort_estimate": effort_estimate,
            "dependencies": dependencies,
            "risk_level": risk_level,
            "business_value": self._assess_business_value(requirement),
            "technical_areas": self._identify_technical_areas(requirement)
        }
    
    def _identify_business_domain(self, requirement: Requirement) -> Optional[BusinessDomain]:
        """Identify the business domain of the requirement.
        
        Args:
            requirement: Requirement object
            
        Returns:
            Identified business domain or None
        """
        text_lower = requirement.description.lower()
        domain_scores = {}
        
        for domain_name, config in self.domain_config.DOMAINS.items():
            score = 0
            
            # Check for keyword matches
            for keyword in config["keywords"]:
                if keyword.lower() in text_lower:
                    score += 2
            
            # Check for business rule mentions
            for rule in config["business_rules"]:
                if rule.lower() in text_lower:
                    score += 3
            
            if score > 0:
                domain_scores[domain_name] = score
        
        if domain_scores:
            # Return domain with highest score
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            try:
                return BusinessDomain(best_domain[0])
            except ValueError:
                logger.warning(f"Unknown domain: {best_domain[0]}")
                return None
        
        return None
    
    def _assess_complexity(
        self,
        requirement: Requirement,
        context: Dict[str, Any]
    ) -> ComplexityLevel:
        """Assess the complexity of the requirement.
        
        Args:
            requirement: Requirement object
            context: Additional context
            
        Returns:
            Assessed complexity level
        """
        text_lower = requirement.description.lower()
        complexity_scores = {"low": 0, "medium": 0, "high": 0, "very_high": 0}
        
        for level, indicators in self.COMPLEXITY_INDICATORS.items():
            # Check keywords
            for keyword in indicators["keywords"]:
                if keyword in text_lower:
                    complexity_scores[level] += 2
            
            # Check scope indicators
            for scope_word in indicators["scope_indicators"]:
                if scope_word in text_lower:
                    complexity_scores[level] += 1
        
        # Consider requirement length as a factor
        word_count = len(requirement.description.split())
        if word_count < 50:
            complexity_scores["low"] += 1
        elif word_count < 150:
            complexity_scores["medium"] += 1
        elif word_count < 300:
            complexity_scores["high"] += 1
        else:
            complexity_scores["very_high"] += 1
        
        # Consider number of acceptance criteria if provided
        if hasattr(requirement, 'acceptance_criteria') and requirement.acceptance_criteria:
            criteria_count = len(requirement.acceptance_criteria)
            if criteria_count <= 3:
                complexity_scores["low"] += 1
            elif criteria_count <= 6:
                complexity_scores["medium"] += 1
            elif criteria_count <= 10:
                complexity_scores["high"] += 1
            else:
                complexity_scores["very_high"] += 1
        
        # Return level with highest score
        best_level = max(complexity_scores.items(), key=lambda x: x[1])
        return ComplexityLevel(best_level[0])
    
    def _estimate_scope(self, requirement: Requirement) -> Dict[str, Any]:
        """Estimate the scope of the requirement.
        
        Args:
            requirement: Requirement object
            
        Returns:
            Scope estimation
        """
        text_lower = requirement.description.lower()
        
        # Count potential features/components mentioned
        component_keywords = [
            "module", "component", "service", "api", "interface",
            "database", "ui", "frontend", "backend", "integration"
        ]
        components_mentioned = sum(1 for kw in component_keywords if kw in text_lower)
        
        # Count user stories or scenarios
        user_story_patterns = [
            r"as a\s+\w+",
            r"i want to",
            r"so that",
            r"given.*when.*then"
        ]
        user_stories = sum(
            1 for pattern in user_story_patterns 
            if re.search(pattern, text_lower)
        )
        
        # Estimate based on indicators
        if components_mentioned <= 1 and user_stories <= 1:
            scope_size = "small"
        elif components_mentioned <= 3 and user_stories <= 3:
            scope_size = "medium"
        elif components_mentioned <= 5 and user_stories <= 5:
            scope_size = "large"
        else:
            scope_size = "very_large"
        
        return {
            "size": scope_size,
            "components": components_mentioned,
            "user_stories": user_stories,
            "estimated_stories": self._estimate_story_count(scope_size)
        }
    
    def _estimate_story_count(self, scope_size: str) -> Tuple[int, int]:
        """Estimate number of stories based on scope.
        
        Args:
            scope_size: Size of scope
            
        Returns:
            Tuple of (min_stories, max_stories)
        """
        story_estimates = {
            "small": (1, 3),
            "medium": (4, 8),
            "large": (9, 15),
            "very_large": (16, 30)
        }
        return story_estimates.get(scope_size, (1, 5))
    
    def _identify_user_types(self, requirement: Requirement) -> List[str]:
        """Identify user types mentioned in the requirement.
        
        Args:
            requirement: Requirement object
            
        Returns:
            List of identified user types
        """
        text_lower = requirement.description.lower()
        
        # Common insurance user types
        user_types = []
        insurance_users = {
            "policyholder": ["policyholder", "insured", "customer", "client"],
            "agent": ["agent", "broker", "producer"],
            "adjuster": ["adjuster", "claims adjuster", "claims handler"],
            "underwriter": ["underwriter", "risk assessor"],
            "administrator": ["admin", "administrator", "system admin"],
            "manager": ["manager", "supervisor", "team lead"],
            "auditor": ["auditor", "compliance officer", "reviewer"],
            "service_rep": ["service rep", "csr", "customer service"]
        }
        
        for user_type, keywords in insurance_users.items():
            if any(keyword in text_lower for keyword in keywords):
                user_types.append(user_type)
        
        return user_types if user_types else ["general_user"]
    
    def _estimate_effort(
        self,
        complexity: ComplexityLevel,
        scope: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate effort required for the requirement.
        
        Args:
            complexity: Complexity level
            scope: Scope information
            
        Returns:
            Effort estimation
        """
        # Base effort days by complexity
        base_effort = {
            ComplexityLevel.LOW: 2,
            ComplexityLevel.MEDIUM: 5,
            ComplexityLevel.HIGH: 10,
            ComplexityLevel.VERY_HIGH: 20
        }
        
        # Scope multiplier
        scope_multiplier = {
            "small": 1.0,
            "medium": 1.5,
            "large": 2.0,
            "very_large": 3.0
        }
        
        base_days = base_effort[complexity]
        multiplier = scope_multiplier[scope["size"]]
        
        estimated_days = base_days * multiplier
        
        # Convert to time ranges
        if estimated_days <= 5:
            time_estimate = "1 week"
        elif estimated_days <= 10:
            time_estimate = "2 weeks"
        elif estimated_days <= 20:
            time_estimate = "1 month"
        elif estimated_days <= 40:
            time_estimate = "2 months"
        elif estimated_days <= 60:
            time_estimate = "3 months"
        else:
            time_estimate = "3+ months"
        
        return {
            "estimated_days": estimated_days,
            "time_estimate": time_estimate,
            "confidence": self._calculate_confidence(complexity, scope)
        }
    
    def _calculate_confidence(
        self,
        complexity: ComplexityLevel,
        scope: Dict[str, Any]
    ) -> str:
        """Calculate confidence in estimation.
        
        Args:
            complexity: Complexity level
            scope: Scope information
            
        Returns:
            Confidence level (high, medium, low)
        """
        if complexity == ComplexityLevel.LOW and scope["size"] == "small":
            return "high"
        elif complexity == ComplexityLevel.VERY_HIGH or scope["size"] == "very_large":
            return "low"
        else:
            return "medium"
    
    def _identify_dependencies(
        self,
        requirement: Requirement,
        context: Dict[str, Any]
    ) -> List[str]:
        """Identify potential dependencies.
        
        Args:
            requirement: Requirement object
            context: Additional context
            
        Returns:
            List of identified dependencies
        """
        dependencies = []
        text_lower = requirement.description.lower()
        
        # Technical dependencies
        tech_dependencies = {
            "database": ["database", "data storage", "persistence"],
            "api": ["api", "service", "endpoint", "integration"],
            "authentication": ["login", "authentication", "authorization", "security"],
            "external_system": ["third-party", "external", "integration", "interface"],
            "reporting": ["report", "analytics", "dashboard", "metrics"],
            "messaging": ["email", "notification", "alert", "message"]
        }
        
        for dep_type, keywords in tech_dependencies.items():
            if any(keyword in text_lower for keyword in keywords):
                dependencies.append(dep_type)
        
        # Business dependencies
        if "approval" in text_lower or "review" in text_lower:
            dependencies.append("approval_workflow")
        
        if "compliance" in text_lower or "regulatory" in text_lower:
            dependencies.append("compliance_review")
        
        return dependencies
    
    def _assess_risk(
        self,
        requirement: Requirement,
        complexity: ComplexityLevel,
        dependencies: List[str]
    ) -> str:
        """Assess risk level of the requirement.
        
        Args:
            requirement: Requirement object
            complexity: Complexity level
            dependencies: List of dependencies
            
        Returns:
            Risk level (critical, high, medium, low)
        """
        risk_score = 0
        text_lower = requirement.description.lower()
        
        # Complexity contribution
        complexity_scores = {
            ComplexityLevel.LOW: 1,
            ComplexityLevel.MEDIUM: 2,
            ComplexityLevel.HIGH: 3,
            ComplexityLevel.VERY_HIGH: 4
        }
        risk_score += complexity_scores[complexity]
        
        # Dependencies contribution
        risk_score += len(dependencies) * 0.5
        
        # Critical keywords
        critical_keywords = [
            "critical", "essential", "mandatory", "required",
            "compliance", "regulatory", "audit", "security"
        ]
        critical_count = sum(1 for kw in critical_keywords if kw in text_lower)
        risk_score += critical_count
        
        # Determine risk level
        if risk_score >= 8:
            return "critical"
        elif risk_score >= 6:
            return "high"
        elif risk_score >= 3:
            return "medium"
        else:
            return "low"
    
    def _assess_business_value(self, requirement: Requirement) -> str:
        """Assess business value of the requirement.
        
        Args:
            requirement: Requirement object
            
        Returns:
            Business value assessment
        """
        text_lower = requirement.description.lower()
        value_indicators = {
            "high": [
                "revenue", "cost saving", "efficiency", "customer satisfaction",
                "competitive advantage", "compliance", "risk reduction"
            ],
            "medium": [
                "improvement", "enhancement", "optimization", "streamline",
                "automate", "simplify"
            ],
            "low": [
                "minor", "cosmetic", "ui", "formatting", "cleanup"
            ]
        }
        
        value_scores = {"high": 0, "medium": 0, "low": 0}
        
        for level, keywords in value_indicators.items():
            for keyword in keywords:
                if keyword in text_lower:
                    value_scores[level] += 1
        
        # Return level with highest score
        if value_scores["high"] > 0:
            return "high"
        elif value_scores["medium"] > 0:
            return "medium"
        else:
            return "low"
    
    def _identify_technical_areas(self, requirement: Requirement) -> List[str]:
        """Identify technical areas involved.
        
        Args:
            requirement: Requirement object
            
        Returns:
            List of technical areas
        """
        text_lower = requirement.description.lower()
        technical_areas = []
        
        area_keywords = {
            "frontend": ["ui", "user interface", "frontend", "screen", "page", "form"],
            "backend": ["backend", "server", "api", "service", "database"],
            "database": ["database", "data", "table", "query", "storage"],
            "integration": ["integration", "interface", "api", "service", "third-party"],
            "security": ["security", "authentication", "authorization", "encryption"],
            "infrastructure": ["infrastructure", "deployment", "server", "cloud", "network"],
            "testing": ["test", "testing", "qa", "quality", "validation"],
            "documentation": ["documentation", "document", "guide", "manual"]
        }
        
        for area, keywords in area_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                technical_areas.append(area)
        
        return technical_areas if technical_areas else ["general"]
    
    def _categorize_requirement(
        self,
        requirement: Requirement,
        analysis: Dict[str, Any]
    ) -> WorkItemType:
        """Categorize requirement as EPIC, Feature, or Story.
        
        Args:
            requirement: Requirement object
            analysis: Analysis results
            
        Returns:
            Work item type
        """
        effort = analysis["effort_estimate"]["estimated_days"]
        scope = analysis["scope"]["size"]
        user_types = len(analysis["user_types"])
        
        # Decision logic based on CLAUDE.md specifications
        if effort > 60 and user_types > 3:
            # Large initiative spanning multiple user types
            return WorkItemType.EPIC
        elif effort > 20 or (scope in ["large", "very_large"] and user_types > 1):
            # Significant functionality for specific workflow
            return WorkItemType.FEATURE
        else:
            # Specific, implementable requirement
            return WorkItemType.STORY
    
    async def _generate_clarifications(
        self,
        requirement: Requirement,
        analysis: Dict[str, Any],
        work_item_type: WorkItemType
    ) -> List[Dict[str, str]]:
        """Generate clarifying questions for the requirement.
        
        Args:
            requirement: Requirement object
            analysis: Analysis results
            work_item_type: Categorized work item type
            
        Returns:
            List of clarification questions
        """
        clarifications = []
        
        # Domain-specific clarifications
        if not analysis.get("domain"):
            clarifications.append({
                "category": "domain",
                "question": "Which business area does this requirement primarily affect? (Policy Management, Claims, Customer Service, Billing, or Compliance)",
                "importance": "high"
            })
        
        # User type clarifications
        if not analysis.get("user_types") or analysis["user_types"] == ["general_user"]:
            clarifications.append({
                "category": "users",
                "question": "Who are the primary users of this functionality? (Policyholders, Agents, Adjusters, Underwriters, etc.)",
                "importance": "high"
            })
        
        # Acceptance criteria clarifications
        if work_item_type == WorkItemType.STORY and not requirement.acceptance_criteria:
            clarifications.append({
                "category": "acceptance",
                "question": "What are the specific acceptance criteria for this requirement? Please provide in Given/When/Then format.",
                "importance": "high"
            })
        
        # Technical clarifications
        if "integration" in analysis.get("dependencies", []):
            clarifications.append({
                "category": "technical",
                "question": "Which external systems or APIs need to be integrated?",
                "importance": "medium"
            })
        
        # Compliance clarifications
        if analysis.get("risk_level") in ["high", "critical"]:
            clarifications.append({
                "category": "compliance",
                "question": "Are there specific regulatory or compliance requirements that must be met?",
                "importance": "high"
            })
        
        # Priority clarifications
        if work_item_type in [WorkItemType.EPIC, WorkItemType.FEATURE]:
            clarifications.append({
                "category": "priority",
                "question": "What is the business priority and target timeline for this initiative?",
                "importance": "medium"
            })
        
        # Success metrics clarifications
        if analysis.get("business_value") == "high":
            clarifications.append({
                "category": "metrics",
                "question": "How will success be measured for this requirement? What are the key metrics?",
                "importance": "medium"
            })
        
        return clarifications
    
    def _identify_compliance_requirements(
        self,
        requirement: Requirement
    ) -> List[str]:
        """Identify compliance requirements.
        
        Args:
            requirement: Requirement object
            
        Returns:
            List of compliance types
        """
        text_lower = requirement.description.lower()
        compliance_types = []
        
        compliance_keywords = {
            ComplianceType.HIPAA: ["hipaa", "health information", "phi", "protected health"],
            ComplianceType.SOX: ["sox", "sarbanes", "financial reporting", "audit trail"],
            ComplianceType.STATE_REGULATIONS: ["state regulation", "insurance regulation", "department of insurance"],
            ComplianceType.GDPR: ["gdpr", "data protection", "eu privacy"],
            ComplianceType.CCPA: ["ccpa", "california privacy", "consumer privacy"],
            ComplianceType.INTERNAL: ["internal policy", "company policy", "hartford standard"]
        }
        
        for compliance_type, keywords in compliance_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                compliance_types.append(compliance_type.value)
        
        return compliance_types
    
    def _get_recommended_action(
        self,
        work_item_type: WorkItemType,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get recommended next action.
        
        Args:
            work_item_type: Type of work item
            analysis: Analysis results
            
        Returns:
            Recommended action
        """
        if work_item_type == WorkItemType.EPIC:
            return {
                "action": "create_epic",
                "description": "Create an EPIC in Rally and break down into Features",
                "estimated_features": (3, 8),
                "next_steps": [
                    "Define EPIC objectives and success criteria",
                    "Identify Features within the EPIC",
                    "Establish timeline and milestones"
                ]
            }
        elif work_item_type == WorkItemType.FEATURE:
            return {
                "action": "create_feature",
                "description": "Create a Feature in Rally and define Stories",
                "estimated_stories": analysis["scope"]["estimated_stories"],
                "next_steps": [
                    "Define Feature scope and boundaries",
                    "Break down into implementable Stories",
                    "Prioritize Stories for implementation"
                ]
            }
        else:  # STORY
            return {
                "action": "create_story",
                "description": "Create a Story in Rally and proceed with implementation",
                "story_points": self._estimate_story_points(analysis["complexity"]),
                "next_steps": [
                    "Finalize acceptance criteria",
                    "Perform repository analysis",
                    "Create implementation plan",
                    "Begin development"
                ]
            }
    
    def _estimate_story_points(self, complexity: ComplexityLevel) -> int:
        """Estimate story points based on complexity.
        
        Args:
            complexity: Complexity level
            
        Returns:
            Estimated story points (Fibonacci)
        """
        points_map = {
            ComplexityLevel.LOW: 3,
            ComplexityLevel.MEDIUM: 5,
            ComplexityLevel.HIGH: 8,
            ComplexityLevel.VERY_HIGH: 13
        }
        return points_map[complexity]
    
    async def _analyze_requirement_with_repo_context(
        self,
        requirement: Requirement,
        context: Dict[str, Any],
        repository_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze requirement with repository context.
        
        Args:
            requirement: Requirement object
            context: Additional context
            repository_context: Repository analysis context
            
        Returns:
            Context-aware analysis results
        """
        # Start with basic analysis
        basic_analysis = await self._analyze_requirement(requirement, context)
        
        # Enhance with repository context
        enhanced_analysis = basic_analysis.copy()
        
        # Adjust complexity based on existing codebase
        existing_complexity = repository_context.get("complexity_metrics", {}).get("average_complexity", 0)
        if existing_complexity > 15:
            # High complexity codebase makes changes harder
            if enhanced_analysis["complexity"] == "low":
                enhanced_analysis["complexity"] = "medium"
            elif enhanced_analysis["complexity"] == "medium":
                enhanced_analysis["complexity"] = "high"
        
        # Identify implementation patterns from repo
        repo_patterns = repository_context.get("patterns", [])
        if repo_patterns:
            enhanced_analysis["available_patterns"] = repo_patterns[:5]
            enhanced_analysis["suggested_patterns"] = self._suggest_patterns_for_requirement(
                requirement, repo_patterns
            )
        
        # Map to existing components
        existing_components = repository_context.get("existing_components", [])
        if existing_components:
            enhanced_analysis["affected_components"] = self._identify_affected_components_from_requirement(
                requirement, existing_components
            )
        
        # Adjust effort based on repo structure
        if repository_context.get("architecture") == "microservices":
            # Microservices might require changes in multiple services
            enhanced_analysis["effort_estimate"]["coordination_overhead"] = "high"
        
        # Consider existing business domains
        repo_domains = repository_context.get("business_domains", {})
        if repo_domains:
            # Find matching domain
            for domain_name, domain_info in repo_domains.items():
                if domain_info.get("primary"):
                    enhanced_analysis["primary_repo_domain"] = domain_name
                    break
        
        return enhanced_analysis
    
    def _suggest_patterns_for_requirement(
        self,
        requirement: Requirement,
        repo_patterns: List
    ) -> List[str]:
        """Suggest patterns from repo for requirement.
        
        Args:
            requirement: Requirement object
            repo_patterns: Available patterns in repo
            
        Returns:
            Suggested patterns
        """
        suggested = []
        req_lower = requirement.description.lower()
        
        pattern_keywords = {
            "repository-pattern": ["data", "database", "storage", "crud"],
            "service-layer": ["business", "logic", "process", "service"],
            "rest-api": ["api", "endpoint", "rest", "http"],
            "async": ["async", "background", "queue", "scheduled"],
            "error-handling": ["error", "exception", "validation", "check"]
        }
        
        for pattern in repo_patterns:
            if hasattr(pattern, 'name'):
                pattern_name = pattern.name
            else:
                pattern_name = str(pattern)
            
            # Check if pattern is relevant to requirement
            if pattern_name in pattern_keywords:
                keywords = pattern_keywords[pattern_name]
                if any(keyword in req_lower for keyword in keywords):
                    suggested.append(pattern_name)
        
        return suggested[:3]
    
    def _identify_affected_components_from_requirement(
        self,
        requirement: Requirement,
        existing_components: List
    ) -> List[str]:
        """Identify components affected by requirement.
        
        Args:
            requirement: Requirement object
            existing_components: Existing components in repo
            
        Returns:
            Affected components
        """
        affected = []
        req_lower = requirement.description.lower()
        
        for component in existing_components[:20]:
            if isinstance(component, dict):
                path = component.get("path", "").lower()
            else:
                path = str(component).lower()
            
            # Extract component name
            if "/" in path:
                component_name = path.split("/")[0]
            else:
                component_name = path
            
            # Check if component might be affected
            if component_name in req_lower or any(
                keyword in component_name 
                for keyword in ["service", "controller", "model", "api"]
                if keyword in req_lower
            ):
                if component_name not in affected:
                    affected.append(component_name)
        
        return affected[:5]
    
    def _categorize_requirement_with_context(
        self,
        requirement: Requirement,
        analysis: Dict[str, Any],
        repository_context: Dict[str, Any],
        is_enhancement: bool
    ) -> WorkItemType:
        """Categorize requirement with repository context.
        
        Args:
            requirement: Requirement object
            analysis: Analysis results
            repository_context: Repository context
            is_enhancement: Whether this is an enhancement
            
        Returns:
            Work item type
        """
        # For enhancements/bugfixes, bias toward Story
        if is_enhancement:
            effort = analysis["effort_estimate"]["estimated_days"]
            
            # Enhancements are usually Stories unless very large
            if effort > 30:
                return WorkItemType.FEATURE
            else:
                return WorkItemType.STORY
        
        # For new features, use standard categorization
        return self._categorize_requirement(requirement, analysis)
    
    async def _generate_clarifications_with_context(
        self,
        requirement: Requirement,
        analysis: Dict[str, Any],
        work_item_type: WorkItemType,
        repository_context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate clarifications with repository context.
        
        Args:
            requirement: Requirement object
            analysis: Analysis results
            work_item_type: Work item type
            repository_context: Repository context
            
        Returns:
            Clarification questions
        """
        # Start with basic clarifications
        clarifications = await self._generate_clarifications(
            requirement, analysis, work_item_type
        )
        
        # Add repository-specific clarifications
        if repository_context:
            # Check if affected components are identified
            if analysis.get("affected_components"):
                clarifications.append({
                    "category": "components",
                    "question": f"The following components might be affected: {', '.join(analysis['affected_components'])}. Are there others to consider?",
                    "importance": "medium"
                })
            
            # Check architecture compatibility
            architecture = repository_context.get("architecture")
            if architecture:
                clarifications.append({
                    "category": "architecture",
                    "question": f"The system uses {architecture} architecture. How should this requirement fit within that structure?",
                    "importance": "high"
                })
            
            # Check for pattern usage
            if analysis.get("suggested_patterns"):
                clarifications.append({
                    "category": "patterns",
                    "question": f"Should we use these existing patterns: {', '.join(analysis['suggested_patterns'])}?",
                    "importance": "medium"
                })
        
        return clarifications
    
    def _get_recommended_action_with_context(
        self,
        work_item_type: WorkItemType,
        analysis: Dict[str, Any],
        repository_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get recommended action with repository context.
        
        Args:
            work_item_type: Type of work item
            analysis: Analysis results
            repository_context: Repository context
            
        Returns:
            Recommended action
        """
        base_action = self._get_recommended_action(work_item_type, analysis)
        
        if repository_context:
            # Add repository-specific next steps
            repo_steps = []
            
            if analysis.get("affected_components"):
                repo_steps.append(f"Review affected components: {', '.join(analysis['affected_components'][:3])}")
            
            if analysis.get("suggested_patterns"):
                repo_steps.append(f"Implement using patterns: {', '.join(analysis['suggested_patterns'][:2])}")
            
            if repository_context.get("implementation_context"):
                repo_steps.append("Follow the suggested implementation approach from repository analysis")
            
            # Prepend repo steps to next steps
            if repo_steps:
                base_action["next_steps"] = repo_steps + base_action.get("next_steps", [])
        
        return base_action
    
    def _generate_repo_aware_insights(
        self,
        requirement: Requirement,
        analysis: Dict[str, Any],
        repository_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate repository-aware insights.
        
        Args:
            requirement: Requirement object
            analysis: Analysis results
            repository_context: Repository context
            
        Returns:
            Repository-aware insights
        """
        insights = {
            "implementation_confidence": "high" if repository_context else "medium",
            "existing_patterns_applicable": bool(analysis.get("suggested_patterns")),
            "architecture_alignment": self._assess_architecture_alignment(
                requirement, repository_context
            ),
            "estimated_code_changes": self._estimate_code_changes(
                analysis, repository_context
            ),
            "reusability_opportunities": self._identify_reusability(
                analysis, repository_context
            )
        }
        
        # Add specific recommendations
        recommendations = []
        
        if insights["existing_patterns_applicable"]:
            recommendations.append("Leverage existing patterns for consistency")
        
        if analysis.get("affected_components"):
            recommendations.append(f"Focus testing on {len(analysis['affected_components'])} affected components")
        
        if repository_context.get("complexity_metrics", {}).get("average_complexity", 0) > 10:
            recommendations.append("Consider refactoring complex areas while implementing")
        
        insights["recommendations"] = recommendations
        
        return insights
    
    def _assess_architecture_alignment(
        self,
        requirement: Requirement,
        repository_context: Dict[str, Any]
    ) -> str:
        """Assess how well requirement aligns with architecture.
        
        Args:
            requirement: Requirement object
            repository_context: Repository context
            
        Returns:
            Alignment assessment
        """
        architecture = repository_context.get("architecture", "unknown")
        
        if architecture == "microservices":
            # Check if requirement is well-bounded
            if "service" in requirement.description.lower() or "api" in requirement.description.lower():
                return "good - fits service boundaries"
            else:
                return "review - may span multiple services"
        elif architecture == "monolithic":
            return "good - single codebase change"
        else:
            return "unknown - architecture type not determined"
    
    def _estimate_code_changes(
        self,
        analysis: Dict[str, Any],
        repository_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate code changes required.
        
        Args:
            analysis: Analysis results
            repository_context: Repository context
            
        Returns:
            Code change estimates
        """
        affected_count = len(analysis.get("affected_components", []))
        complexity = analysis.get("complexity", "medium")
        
        if complexity == "low" and affected_count <= 2:
            return {
                "files": "1-5",
                "lines": "10-50",
                "risk": "low"
            }
        elif complexity == "medium" or affected_count <= 5:
            return {
                "files": "5-15",
                "lines": "50-200",
                "risk": "medium"
            }
        else:
            return {
                "files": "15+",
                "lines": "200+",
                "risk": "high"
            }
    
    def _identify_reusability(
        self,
        analysis: Dict[str, Any],
        repository_context: Dict[str, Any]
    ) -> List[str]:
        """Identify reusability opportunities.
        
        Args:
            analysis: Analysis results
            repository_context: Repository context
            
        Returns:
            Reusability opportunities
        """
        opportunities = []
        
        if analysis.get("suggested_patterns"):
            opportunities.append("Existing patterns can be reused")
        
        if repository_context.get("framework"):
            opportunities.append(f"Leverage {repository_context['framework']} features")
        
        if analysis.get("available_patterns"):
            opportunities.append("Similar implementations exist in codebase")
        
        return opportunities