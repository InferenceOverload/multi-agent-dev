"""Intelligent Router - LLM-based decision making for choosing the right approach."""

from typing import Dict, Any, Optional, List
from enum import Enum


class AnalysisApproach(Enum):
    """Different approaches the agent can take."""
    QUICK_DISCOVERY = "quick_discovery"  # Fast, surface-level analysis
    DEEP_RAG = "deep_rag"  # Full RAG corpus creation and indexing
    SPECIALIST_ONLY = "specialist_only"  # Use specialist agents without RAG
    HYBRID = "hybrid"  # Combination of approaches
    SIMPLE_SEARCH = "simple_search"  # Just grep/search without RAG


class IntelligentRouter:
    """
    Routes requests to the appropriate analysis method based on:
    - Query complexity
    - Codebase size
    - User intent
    - Available resources
    """
    
    def __init__(self):
        self.decision_history = []
        self.rag_corpus_cache = {}  # Track existing RAG corpora
        
    async def decide_approach(
        self, 
        user_query: str,
        repo_url: Optional[str] = None,
        codebase_size: Optional[Dict] = None,
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Use LLM intelligence to decide the best approach.
        
        This is where the agent's intelligence shines - it decides
        based on context, not hard-coded rules.
        """
        
        # Build context for decision making
        decision_context = self._build_decision_context(
            user_query, repo_url, codebase_size, conversation_history
        )
        
        # Let the LLM analyze and decide
        decision = await self._llm_decide(decision_context)
        
        # Record decision for learning
        self.decision_history.append({
            "query": user_query,
            "decision": decision,
            "context": decision_context
        })
        
        return decision
    
    def _build_decision_context(
        self, 
        query: str,
        repo_url: Optional[str],
        codebase_size: Optional[Dict],
        history: List[Dict]
    ) -> Dict:
        """Build context for LLM to make intelligent decision."""
        
        context = {
            "query": query,
            "query_type": self._classify_query(query),
            "has_repository": repo_url is not None,
            "repository_url": repo_url,
            "conversation_depth": len(history) if history else 0,
            "previous_rag_exists": repo_url in self.rag_corpus_cache if repo_url else False
        }
        
        # Add codebase metrics if available
        if codebase_size:
            context.update({
                "total_files": codebase_size.get("total_files", 0),
                "size_category": codebase_size.get("size_category", "unknown"),
                "primary_language": codebase_size.get("primary_language"),
                "complexity_estimate": self._estimate_complexity(codebase_size)
            })
        
        # Check if this is a follow-up question
        if history and len(history) > 0:
            context["is_followup"] = True
            context["previous_approach"] = history[-1].get("approach_used")
        
        return context
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        
        # Specific code search (check first as it's most specific)
        if any(word in query_lower for word in ["where is", "find", "locate", "search for", "show me"]):
            return "specific_search"
        
        # Modification request
        if any(word in query_lower for word in ["change", "modify", "update", "fix", "add", "color", "button"]):
            return "modification"
        
        # General understanding
        if any(word in query_lower for word in ["what does", "explain", "understand", "analyze this repository"]):
            return "general_understanding"
        
        # High-level question
        if any(word in query_lower for word in ["architecture", "design", "structure", "pattern"]):
            return "high_level"
        
        # API/Database specific
        if any(word in query_lower for word in ["api", "endpoint", "database", "sql", "query"]):
            return "specific_search"
        
        return "unknown"
    
    def _estimate_complexity(self, codebase_size: Dict) -> str:
        """Estimate codebase complexity."""
        total_files = codebase_size.get("total_files", 0)
        
        if total_files < 50:
            return "simple"
        elif total_files < 500:
            return "moderate"
        elif total_files < 5000:
            return "complex"
        else:
            return "very_complex"
    
    async def _llm_decide(self, context: Dict) -> Dict[str, Any]:
        """
        This is where we use LLM intelligence to decide.
        In production, this would call Gemini/Claude to make the decision.
        """
        
        # For now, implementing intelligent heuristics
        # In production, this would be an LLM call
        
        query_type = context.get("query_type")
        complexity = context.get("complexity_estimate", "unknown")
        is_followup = context.get("is_followup", False)
        previous_rag_exists = context.get("previous_rag_exists", False)
        
        # Decision logic (this would be LLM-based in production)
        approach = AnalysisApproach.QUICK_DISCOVERY
        reasoning = ""
        
        # If RAG corpus already exists, prefer using it
        if previous_rag_exists:
            approach = AnalysisApproach.DEEP_RAG
            reasoning = "RAG corpus already exists for this repository, reusing it"
        
        # For specific searches in large codebases
        elif query_type == "specific_search" and complexity in ["complex", "very_complex"]:
            approach = AnalysisApproach.DEEP_RAG
            reasoning = "Large codebase with specific search query - RAG will be most effective"
        
        # For modifications, we need precise code location
        elif query_type == "modification":
            if complexity in ["simple", "moderate"]:
                approach = AnalysisApproach.SIMPLE_SEARCH
                reasoning = "Small codebase modification - simple search is sufficient"
            else:
                approach = AnalysisApproach.DEEP_RAG
                reasoning = "Large codebase modification - need RAG for accurate code location"
        
        # For general understanding
        elif query_type == "general_understanding":
            approach = AnalysisApproach.QUICK_DISCOVERY
            reasoning = "General understanding request - quick discovery is sufficient"
        
        # For high-level architecture questions
        elif query_type == "high_level":
            approach = AnalysisApproach.SPECIALIST_ONLY
            reasoning = "Architecture question - specialist agents are best suited"
        
        # For follow-up questions
        elif is_followup:
            # Escalate if previous approach wasn't sufficient
            if context.get("previous_approach") == AnalysisApproach.QUICK_DISCOVERY.value:
                approach = AnalysisApproach.DEEP_RAG
                reasoning = "Follow-up question after quick discovery - escalating to RAG"
        
        return {
            "approach": approach.value,
            "reasoning": reasoning,
            "confidence": 0.85,  # Would be calculated by LLM
            "alternative_approaches": self._get_alternatives(approach),
            "estimated_time": self._estimate_time(approach, complexity),
            "recommended_tools": self._get_recommended_tools(approach)
        }
    
    def _get_alternatives(self, primary: AnalysisApproach) -> List[str]:
        """Get alternative approaches if primary fails."""
        alternatives = []
        
        if primary == AnalysisApproach.SIMPLE_SEARCH:
            alternatives = [AnalysisApproach.DEEP_RAG.value]
        elif primary == AnalysisApproach.QUICK_DISCOVERY:
            alternatives = [AnalysisApproach.DEEP_RAG.value, AnalysisApproach.SPECIALIST_ONLY.value]
        elif primary == AnalysisApproach.SPECIALIST_ONLY:
            alternatives = [AnalysisApproach.DEEP_RAG.value]
        
        return alternatives
    
    def _estimate_time(self, approach: AnalysisApproach, complexity: str) -> str:
        """Estimate time for the approach."""
        time_matrix = {
            AnalysisApproach.QUICK_DISCOVERY: {"simple": "5s", "moderate": "10s", "complex": "20s"},
            AnalysisApproach.SIMPLE_SEARCH: {"simple": "2s", "moderate": "5s", "complex": "10s"},
            AnalysisApproach.DEEP_RAG: {"simple": "30s", "moderate": "1m", "complex": "2m"},
            AnalysisApproach.SPECIALIST_ONLY: {"simple": "10s", "moderate": "15s", "complex": "30s"},
            AnalysisApproach.HYBRID: {"simple": "1m", "moderate": "2m", "complex": "3m"}
        }
        
        return time_matrix.get(approach, {}).get(complexity, "unknown")
    
    def _get_recommended_tools(self, approach: AnalysisApproach) -> List[str]:
        """Get recommended tools for the approach."""
        tools_map = {
            AnalysisApproach.QUICK_DISCOVERY: ["DiscoveryAgent", "CoreTools.get_directory_structure"],
            AnalysisApproach.SIMPLE_SEARCH: ["CoreTools.search_pattern", "CoreTools.find_files_by_extension"],
            AnalysisApproach.DEEP_RAG: ["VertexAI.RAGEngine", "CodeRAGBot"],
            AnalysisApproach.SPECIALIST_ONLY: ["SpecialistRegistry", "SmartCoordinator"],
            AnalysisApproach.HYBRID: ["All available tools"]
        }
        
        return tools_map.get(approach, [])