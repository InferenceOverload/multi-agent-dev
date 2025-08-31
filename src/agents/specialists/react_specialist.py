"""React Specialist Agent - Expert in React/Frontend development."""

from typing import Dict, Any, List
from ..specialist_registry import SpecialistAgent


class ReactSpecialistAgent(SpecialistAgent):
    """Specialist agent for React and frontend development."""
    
    @property
    def name(self) -> str:
        return "react_specialist"
    
    @property
    def supported_types(self) -> List[str]:
        return ["react", "javascript", "typescript", "frontend", "vue", "angular"]
    
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze React/frontend codebase."""
        analysis = {
            "component_architecture": None,
            "state_management": None,
            "routing": None,
            "styling_approach": None,
            "build_setup": None,
            "testing_setup": None
        }
        
        # Analyze component architecture
        file_stats = context.get("file_statistics", {})
        by_extension = file_stats.get("by_extension", {})
        
        if ".jsx" in by_extension or ".tsx" in by_extension:
            analysis["component_architecture"] = "JSX/TSX components detected"
        
        # Detect state management
        tech_stack = context.get("tech_stack", {})
        frameworks = context.get("frameworks", [])
        
        if "React" in frameworks:
            analysis["state_management"] = "React (hooks, context)"
            analysis["routing"] = "Likely React Router"
        elif "Vue" in frameworks:
            analysis["state_management"] = "Vue (Vuex or Pinia)"
            analysis["routing"] = "Vue Router"
        elif "Angular" in frameworks:
            analysis["state_management"] = "Angular (RxJS, Services)"
            analysis["routing"] = "Angular Router"
        
        # Detect styling approach
        if ".scss" in by_extension or ".sass" in by_extension:
            analysis["styling_approach"] = "SASS/SCSS"
        elif ".less" in by_extension:
            analysis["styling_approach"] = "LESS"
        elif "styled-components" in str(context):
            analysis["styling_approach"] = "CSS-in-JS (styled-components)"
        else:
            analysis["styling_approach"] = "CSS Modules or plain CSS"
        
        # Build setup
        if context.get("has_webpack"):
            analysis["build_setup"] = "Webpack"
        elif "Next" in frameworks:
            analysis["build_setup"] = "Next.js"
        elif "Gatsby" in frameworks:
            analysis["build_setup"] = "Gatsby"
        else:
            analysis["build_setup"] = "Create React App or Vite"
        
        # Testing setup
        if context.get("has_tests"):
            analysis["testing_setup"] = "Tests detected (likely Jest + React Testing Library)"
        
        return analysis
    
    async def create_story(self, requirement: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create frontend-specific story with implementation details."""
        
        # Analyze requirement for UI-specific needs
        requirement_lower = requirement.lower()
        
        # Determine story type
        story_type = "feature"
        if any(word in requirement_lower for word in ["button", "form", "modal", "component", "ui"]):
            story_type = "ui_component"
        elif any(word in requirement_lower for word in ["api", "fetch", "data", "state"]):
            story_type = "data_integration"
        elif any(word in requirement_lower for word in ["style", "css", "theme", "color", "design"]):
            story_type = "styling"
        
        story = {
            "title": f"[Frontend] {requirement[:80]}",
            "type": story_type,
            "description": requirement,
            "acceptance_criteria": [],
            "implementation_details": {
                "components_to_create": [],
                "components_to_modify": [],
                "state_changes": [],
                "api_calls": [],
                "styling_changes": [],
                "tests_to_write": []
            },
            "technical_notes": [],
            "estimated_complexity": "medium"
        }
        
        # Add specific implementation details based on story type
        if story_type == "ui_component":
            story["implementation_details"]["components_to_create"] = [
                "New React component with props interface",
                "Component styles (CSS/SCSS file)",
                "Component tests"
            ]
            story["acceptance_criteria"] = [
                "Component renders correctly with all props",
                "Component is responsive on mobile/tablet/desktop",
                "Component passes accessibility checks",
                "Component has unit tests with >80% coverage"
            ]
            story["technical_notes"] = [
                "Follow existing component patterns in src/components",
                "Use existing design system tokens for styling",
                "Ensure component is reusable and well-documented"
            ]
            
        elif story_type == "data_integration":
            story["implementation_details"]["api_calls"] = [
                "Create API service function",
                "Add Redux/Context actions and reducers",
                "Handle loading and error states"
            ]
            story["acceptance_criteria"] = [
                "Data fetches correctly from API",
                "Loading state displays while fetching",
                "Error states handled gracefully",
                "Data updates reflected in UI immediately"
            ]
            
        elif story_type == "styling":
            story["implementation_details"]["styling_changes"] = [
                "Update theme configuration",
                "Modify component styles",
                "Ensure consistency across breakpoints"
            ]
            story["acceptance_criteria"] = [
                "Styles applied consistently across app",
                "No visual regressions in existing components",
                "Maintains accessibility standards (WCAG 2.1 AA)"
            ]
        
        # Add files to modify based on context
        if context.get("entry_points"):
            story["implementation_details"]["files_to_modify"] = context["entry_points"][:3]
        
        # Estimate complexity
        if len(story["implementation_details"]["components_to_create"]) > 3:
            story["estimated_complexity"] = "high"
        elif len(story["implementation_details"]["components_to_create"]) == 0:
            story["estimated_complexity"] = "low"
        
        return story