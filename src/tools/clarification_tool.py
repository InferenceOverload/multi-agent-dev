"""Clarification tool for requirements analysis."""

from typing import Dict, Any, List
# Comment out ADK tool import as it's not needed
# from google.adk import tool


class ClarificationTool:
    """Tool for generating clarifying questions for requirements."""
    
    def __init__(self):
        """Initialize clarification tool."""
        self.question_templates = {
            "scope": [
                "What specific functionality should be included in this requirement?",
                "Are there any features that should be explicitly excluded?",
                "What are the boundaries of this implementation?"
            ],
            "users": [
                "Who are the primary users of this feature?",
                "Are there different user roles with different permissions?",
                "What is the expected user volume?"
            ],
            "technical": [
                "Are there specific technical constraints or requirements?",
                "Which systems need to be integrated?",
                "What are the performance requirements?"
            ],
            "business": [
                "What is the business priority of this requirement?",
                "What is the expected ROI or business value?",
                "Are there regulatory or compliance considerations?"
            ],
            "timeline": [
                "What is the target completion date?",
                "Are there any dependencies on other projects?",
                "Is this blocking other initiatives?"
            ]
        }
    
    def generate_clarifications(
        self,
        requirement: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate clarifying questions based on requirement analysis.
        
        Args:
            requirement: Requirement details
            analysis: Analysis results
            
        Returns:
            List of clarification questions
        """
        clarifications = []
        
        # Check for missing information
        if not analysis.get("user_types") or analysis["user_types"] == ["general_user"]:
            clarifications.append({
                "category": "users",
                "question": self.question_templates["users"][0],
                "importance": "high",
                "reason": "User types not clearly identified"
            })
        
        if analysis.get("complexity") in ["high", "very_high"]:
            clarifications.append({
                "category": "scope",
                "question": self.question_templates["scope"][1],
                "importance": "high",
                "reason": "High complexity requires clear scope boundaries"
            })
        
        if analysis.get("dependencies"):
            clarifications.append({
                "category": "technical",
                "question": self.question_templates["technical"][1],
                "importance": "medium",
                "reason": "Integration dependencies identified"
            })
        
        if not analysis.get("business_value") or analysis["business_value"] == "low":
            clarifications.append({
                "category": "business",
                "question": self.question_templates["business"][1],
                "importance": "high",
                "reason": "Business value needs clarification"
            })
        
        return clarifications
    
    def prioritize_clarifications(
        self,
        clarifications: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Prioritize clarifications by importance.
        
        Args:
            clarifications: List of clarifications
            
        Returns:
            Prioritized list of clarifications
        """
        importance_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        
        return sorted(
            clarifications,
            key=lambda x: importance_order.get(x.get("importance", "low"), 3)
        )