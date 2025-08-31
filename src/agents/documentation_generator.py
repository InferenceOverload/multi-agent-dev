"""Documentation Generation Agent for legacy codebases."""

import os
import ast
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import sqlparse
from collections import defaultdict

from ..agents.base_agent import HartfordBaseAgent
from ..agents.repository_analyst import RepositoryAnalystAgent
from ..models.base import FileContext, CodePattern
from ..utils.logger import get_logger


logger = get_logger(__name__)


class DocumentationGeneratorAgent(HartfordBaseAgent):
    """Agent for generating comprehensive documentation from legacy codebases."""
    
    def __init__(self, **kwargs):
        """Initialize Documentation Generator Agent."""
        super().__init__(
            name="DocumentationGenerator",
            description=(
                "Generates comprehensive documentation from legacy codebases including "
                "architecture diagrams, sequence diagrams, flowcharts, and human-friendly "
                "documentation. Handles Python, Java, and SQL codebases."
            ),
            **kwargs
        )
        self.repo_analyst = RepositoryAnalystAgent(memory_service=self.memory_service)
        self.supported_languages = ["Python", "Java", "SQL", "JavaScript", "TypeScript", "C#"]
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process repository and generate documentation.
        
        Args:
            input_data: Contains 'repo_url' or 'repo_path', optional 'output_format'
            
        Returns:
            Generated documentation and diagrams
        """
        repo_url = input_data.get("repo_url")
        repo_path = input_data.get("repo_path")
        output_format = input_data.get("output_format", "markdown")
        
        # First, analyze the repository using our existing analyst
        logger.info("Starting repository analysis for documentation...")
        
        if repo_url:
            repo_analysis = await self.repo_analyst.execute(
                self.context.session_id,
                {"repo_url": repo_url, "story_context": {"title": "Documentation Generation"}}
            )
        else:
            repo_analysis = await self._analyze_local_repo(repo_path)
        
        # Generate various documentation components
        logger.info("Generating documentation components...")
        
        documentation = {
            "overview": await self._generate_overview(repo_analysis),
            "architecture": await self._generate_architecture_diagram(repo_analysis),
            "components": await self._document_components(repo_analysis),
            "database": await self._document_database(repo_analysis),
            "api": await self._document_apis(repo_analysis),
            "flows": await self._generate_flow_diagrams(repo_analysis),
            "dependencies": await self._document_dependencies(repo_analysis),
            "tech_debt": await self._assess_tech_debt(repo_analysis),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Generate final documentation
        if output_format == "markdown":
            documentation["markdown"] = await self._generate_markdown_docs(documentation)
        
        # Save documentation
        await self._save_documentation(documentation, repo_analysis.get("repository_info", {}))
        
        return documentation
    
    async def _generate_overview(self, repo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate repository overview.
        
        Args:
            repo_analysis: Repository analysis results
            
        Returns:
            Overview documentation
        """
        repo_info = repo_analysis.get("repository_info", {})
        analysis = repo_analysis.get("analysis", {})
        
        overview = {
            "name": repo_info.get("name", "Unknown"),
            "description": self._infer_description(repo_analysis),
            "architecture_type": repo_info.get("architecture_type", "Unknown"),
            "primary_language": self._get_primary_language(repo_info.get("language_stats", {})),
            "frameworks": [repo_info.get("framework")] if repo_info.get("framework") else [],
            "total_files": repo_info.get("total_files", 0),
            "total_lines": repo_info.get("total_lines", 0),
            "key_technologies": self._identify_technologies(repo_analysis),
            "business_domains": list(analysis.get("business_domains", {}).keys()),
            "complexity_rating": self._calculate_overall_complexity(analysis)
        }
        
        return overview
    
    async def _generate_architecture_diagram(self, repo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate architecture diagram in Mermaid format.
        
        Args:
            repo_analysis: Repository analysis results
            
        Returns:
            Architecture diagram
        """
        components = await self._identify_components(repo_analysis)
        
        # Generate Mermaid diagram
        mermaid_diagram = "graph TB\n"
        
        # Add components
        for comp_id, component in components.items():
            label = component['name'].replace("-", "_")
            mermaid_diagram += f"    {label}[{component['name']}]\n"
        
        # Add relationships
        relationships = await self._identify_relationships(components, repo_analysis)
        for rel in relationships:
            from_label = rel['from'].replace("-", "_")
            to_label = rel['to'].replace("-", "_")
            mermaid_diagram += f"    {from_label} --> {to_label}\n"
        
        # Add styling
        mermaid_diagram += "\n    %% Styling\n"
        for comp_id, component in components.items():
            label = component['name'].replace("-", "_")
            if component['type'] == 'frontend':
                mermaid_diagram += f"    style {label} fill:#e1f5fe\n"
            elif component['type'] == 'backend':
                mermaid_diagram += f"    style {label} fill:#f3e5f5\n"
            elif component['type'] == 'database':
                mermaid_diagram += f"    style {label} fill:#fff3e0\n"
        
        return {
            "mermaid": mermaid_diagram,
            "components": components,
            "relationships": relationships,
            "description": "High-level architecture diagram showing system components and their relationships"
        }
    
    async def _document_components(self, repo_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Document individual components/modules.
        
        Args:
            repo_analysis: Repository analysis results
            
        Returns:
            Component documentation
        """
        components = []
        
        # Analyze file structure to identify components
        files = repo_analysis.get("analysis", {}).get("relevant_files", [])
        
        # Group files by directory/module
        modules = defaultdict(list)
        for file_context in files:
            path_parts = Path(file_context.get("path", "")).parts
            if len(path_parts) > 0:
                module = path_parts[0] if len(path_parts) > 1 else "root"
                modules[module].append(file_context)
        
        # Document each module
        for module_name, module_files in modules.items():
            component = {
                "name": module_name,
                "type": self._infer_component_type(module_name, module_files),
                "description": self._generate_component_description(module_name, module_files),
                "files_count": len(module_files),
                "main_language": self._get_module_language(module_files),
                "key_classes": await self._extract_key_classes(module_files),
                "key_functions": await self._extract_key_functions(module_files),
                "dependencies": self._extract_module_dependencies(module_files),
                "patterns": self._identify_module_patterns(module_files)
            }
            components.append(component)
        
        return components
    
    async def _document_database(self, repo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Document database schema and relationships.
        
        Args:
            repo_analysis: Repository analysis results
            
        Returns:
            Database documentation
        """
        db_docs = {
            "tables": [],
            "relationships": [],
            "stored_procedures": [],
            "views": [],
            "indexes": [],
            "er_diagram": None
        }
        
        # Find SQL files
        files = repo_analysis.get("analysis", {}).get("relevant_files", [])
        sql_files = [f for f in files if f.get("language") == "SQL" or f.get("path", "").endswith(".sql")]
        
        for sql_file in sql_files:
            content = sql_file.get("content", "")
            
            # Parse SQL
            parsed = sqlparse.parse(content)
            for statement in parsed:
                stmt_type = statement.get_type()
                
                if stmt_type == "CREATE":
                    # Extract table definitions
                    table_info = self._extract_table_info(str(statement))
                    if table_info:
                        db_docs["tables"].append(table_info)
                
                elif "PROCEDURE" in str(statement).upper():
                    proc_info = self._extract_procedure_info(str(statement))
                    if proc_info:
                        db_docs["stored_procedures"].append(proc_info)
                
                elif "VIEW" in str(statement).upper():
                    view_info = self._extract_view_info(str(statement))
                    if view_info:
                        db_docs["views"].append(view_info)
        
        # Generate ER diagram if tables found
        if db_docs["tables"]:
            db_docs["er_diagram"] = self._generate_er_diagram(db_docs["tables"])
        
        return db_docs
    
    async def _document_apis(self, repo_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Document API endpoints.
        
        Args:
            repo_analysis: Repository analysis results
            
        Returns:
            API documentation
        """
        apis = []
        files = repo_analysis.get("analysis", {}).get("relevant_files", [])
        
        for file_context in files:
            content = file_context.get("content", "")
            language = file_context.get("language", "")
            
            # Extract API endpoints based on language/framework
            if language == "Python":
                # Flask/FastAPI patterns
                endpoints = self._extract_python_endpoints(content)
                apis.extend(endpoints)
            
            elif language == "Java":
                # Spring Boot patterns
                endpoints = self._extract_java_endpoints(content)
                apis.extend(endpoints)
            
            elif language in ["JavaScript", "TypeScript"]:
                # Express/Node patterns
                endpoints = self._extract_js_endpoints(content)
                apis.extend(endpoints)
        
        # Deduplicate and organize
        unique_apis = self._deduplicate_apis(apis)
        
        return unique_apis
    
    async def _generate_flow_diagrams(self, repo_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate flow diagrams for key processes.
        
        Args:
            repo_analysis: Repository analysis results
            
        Returns:
            Flow diagrams
        """
        flows = []
        
        # Identify key business processes
        processes = await self._identify_business_processes(repo_analysis)
        
        for process in processes:
            # Generate sequence diagram
            sequence_diagram = self._generate_sequence_diagram(process)
            
            # Generate flowchart
            flowchart = self._generate_flowchart(process)
            
            flows.append({
                "name": process["name"],
                "description": process["description"],
                "sequence_diagram": sequence_diagram,
                "flowchart": flowchart,
                "involved_components": process.get("components", [])
            })
        
        return flows
    
    def _generate_sequence_diagram(self, process: Dict[str, Any]) -> str:
        """Generate sequence diagram in Mermaid format.
        
        Args:
            process: Process information
            
        Returns:
            Mermaid sequence diagram
        """
        diagram = "sequenceDiagram\n"
        
        # Add participants
        for component in process.get("components", []):
            diagram += f"    participant {component}\n"
        
        # Add interactions
        for step in process.get("steps", []):
            if step.get("type") == "call":
                diagram += f"    {step['from']}->>+{step['to']}: {step['action']}\n"
                if step.get("response"):
                    diagram += f"    {step['to']}-->>-{step['from']}: {step['response']}\n"
            elif step.get("type") == "note":
                diagram += f"    Note over {step['component']}: {step['text']}\n"
        
        return diagram
    
    def _generate_flowchart(self, process: Dict[str, Any]) -> str:
        """Generate flowchart in Mermaid format.
        
        Args:
            process: Process information
            
        Returns:
            Mermaid flowchart
        """
        flowchart = "flowchart TD\n"
        
        step_counter = 0
        for step in process.get("flow", []):
            step_id = f"step{step_counter}"
            next_id = f"step{step_counter + 1}"
            
            if step["type"] == "start":
                flowchart += f"    {step_id}([{step['label']}])\n"
            elif step["type"] == "process":
                flowchart += f"    {step_id}[{step['label']}]\n"
            elif step["type"] == "decision":
                flowchart += f"    {step_id}{{{step['label']}}}\n"
            elif step["type"] == "end":
                flowchart += f"    {step_id}([{step['label']}])\n"
            
            if step_counter < len(process.get("flow", [])) - 1:
                if step["type"] == "decision":
                    flowchart += f"    {step_id} -->|Yes| {next_id}\n"
                    flowchart += f"    {step_id} -->|No| step{step_counter + 2}\n"
                else:
                    flowchart += f"    {step_id} --> {next_id}\n"
            
            step_counter += 1
        
        return flowchart
    
    async def _document_dependencies(self, repo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Document project dependencies.
        
        Args:
            repo_analysis: Repository analysis results
            
        Returns:
            Dependencies documentation
        """
        dependencies = {
            "languages": repo_analysis.get("repository_info", {}).get("language_stats", {}),
            "frameworks": [],
            "libraries": [],
            "external_services": [],
            "dependency_graph": None
        }
        
        files = repo_analysis.get("analysis", {}).get("relevant_files", [])
        
        # Look for dependency files
        for file_context in files:
            path = file_context.get("path", "")
            content = file_context.get("content", "")
            
            # Python dependencies
            if "requirements.txt" in path or "pyproject.toml" in path:
                deps = self._extract_python_dependencies(content)
                dependencies["libraries"].extend(deps)
            
            # Java dependencies
            elif "pom.xml" in path or "build.gradle" in path:
                deps = self._extract_java_dependencies(content)
                dependencies["libraries"].extend(deps)
            
            # JavaScript dependencies
            elif "package.json" in path:
                deps = self._extract_js_dependencies(content)
                dependencies["libraries"].extend(deps)
        
        # Generate dependency graph
        if dependencies["libraries"]:
            dependencies["dependency_graph"] = self._generate_dependency_graph(dependencies)
        
        return dependencies
    
    async def _assess_tech_debt(self, repo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess technical debt in the codebase.
        
        Args:
            repo_analysis: Repository analysis results
            
        Returns:
            Technical debt assessment
        """
        tech_debt = {
            "overall_score": 0,
            "issues": [],
            "recommendations": [],
            "metrics": {}
        }
        
        complexity_metrics = repo_analysis.get("analysis", {}).get("complexity_metrics", {})
        
        # Assess complexity
        avg_complexity = complexity_metrics.get("average_complexity", 0)
        if avg_complexity > 10:
            tech_debt["issues"].append({
                "type": "high_complexity",
                "severity": "high",
                "description": f"Average cyclomatic complexity is {avg_complexity:.1f} (should be < 10)",
                "files_affected": len(complexity_metrics.get("file_complexities", []))
            })
            tech_debt["recommendations"].append("Refactor complex methods to reduce cyclomatic complexity")
        
        # Check for code duplication patterns
        patterns = repo_analysis.get("analysis", {}).get("patterns", [])
        duplicate_patterns = [p for p in patterns if p.usage_count > 5]
        if duplicate_patterns:
            tech_debt["issues"].append({
                "type": "code_duplication",
                "severity": "medium",
                "description": f"Found {len(duplicate_patterns)} patterns repeated more than 5 times",
                "patterns": [p.name for p in duplicate_patterns[:5]]
            })
            tech_debt["recommendations"].append("Extract common patterns into reusable components")
        
        # Check for outdated patterns
        files = repo_analysis.get("analysis", {}).get("relevant_files", [])
        deprecated_count = sum(1 for f in files if "deprecated" in f.get("content", "").lower())
        if deprecated_count > 0:
            tech_debt["issues"].append({
                "type": "deprecated_code",
                "severity": "medium",
                "description": f"Found deprecated code in {deprecated_count} files",
                "action_needed": "Update or remove deprecated code"
            })
        
        # Calculate overall score (0-100, higher is worse)
        severity_weights = {"critical": 25, "high": 15, "medium": 8, "low": 3}
        for issue in tech_debt["issues"]:
            tech_debt["overall_score"] += severity_weights.get(issue["severity"], 5)
        
        tech_debt["overall_score"] = min(tech_debt["overall_score"], 100)
        
        # Set metrics
        tech_debt["metrics"] = {
            "average_complexity": avg_complexity,
            "total_files": repo_analysis.get("repository_info", {}).get("total_files", 0),
            "total_lines": repo_analysis.get("repository_info", {}).get("total_lines", 0),
            "debt_rating": self._get_debt_rating(tech_debt["overall_score"])
        }
        
        return tech_debt
    
    async def _generate_markdown_docs(self, documentation: Dict[str, Any]) -> str:
        """Generate comprehensive markdown documentation.
        
        Args:
            documentation: All documentation components
            
        Returns:
            Markdown formatted documentation
        """
        overview = documentation["overview"]
        
        md = f"""# {overview['name']} - System Documentation

*Generated on: {documentation['generated_at']}*

## 📋 Executive Summary

**Architecture Type**: {overview['architecture_type']}  
**Primary Language**: {overview['primary_language']}  
**Total Files**: {overview['total_files']:,}  
**Total Lines of Code**: {overview['total_lines']:,}  
**Complexity Rating**: {overview['complexity_rating']}

### Key Technologies
{self._format_list(overview['key_technologies'])}

### Business Domains
{self._format_list(overview['business_domains'])}

---

## 🏗 System Architecture

{documentation['architecture']['description']}

```mermaid
{documentation['architecture']['mermaid']}
```

---

## 📦 Components

"""
        
        # Add component documentation
        for component in documentation["components"]:
            md += f"""### {component['name']}
**Type**: {component['type']}  
**Language**: {component['main_language']}  
**Files**: {component['files_count']}

{component['description']}

"""
            if component['key_classes']:
                md += f"**Key Classes**:\n{self._format_list(component['key_classes'][:5])}\n\n"
            
            if component['key_functions']:
                md += f"**Key Functions**:\n{self._format_list(component['key_functions'][:5])}\n\n"
        
        # Add database documentation
        if documentation["database"]["tables"]:
            md += """---

## 🗄 Database Schema

### Tables
"""
            for table in documentation["database"]["tables"][:10]:
                md += f"- **{table['name']}**: {table.get('description', 'No description')}\n"
            
            if documentation["database"]["er_diagram"]:
                md += f"\n### Entity Relationship Diagram\n\n```mermaid\n{documentation['database']['er_diagram']}\n```\n\n"
        
        # Add API documentation
        if documentation["api"]:
            md += """---

## 🔌 API Endpoints

"""
            for api in documentation["api"][:20]:
                md += f"### {api['method']} {api['path']}\n"
                md += f"{api.get('description', 'No description')}\n\n"
        
        # Add flow diagrams
        if documentation["flows"]:
            md += """---

## 🔄 Business Processes

"""
            for flow in documentation["flows"][:5]:
                md += f"### {flow['name']}\n\n{flow['description']}\n\n"
                
                if flow.get("sequence_diagram"):
                    md += f"#### Sequence Diagram\n\n```mermaid\n{flow['sequence_diagram']}\n```\n\n"
                
                if flow.get("flowchart"):
                    md += f"#### Process Flow\n\n```mermaid\n{flow['flowchart']}\n```\n\n"
        
        # Add technical debt assessment
        tech_debt = documentation["tech_debt"]
        md += f"""---

## ⚠️ Technical Debt Assessment

**Overall Score**: {tech_debt['overall_score']}/100 ({tech_debt['metrics']['debt_rating']})  
**Average Complexity**: {tech_debt['metrics']['average_complexity']:.1f}

### Key Issues
"""
        for issue in tech_debt["issues"][:5]:
            md += f"- **{issue['type']}** ({issue['severity']}): {issue['description']}\n"
        
        md += "\n### Recommendations\n"
        for rec in tech_debt["recommendations"][:5]:
            md += f"- {rec}\n"
        
        # Add dependencies
        md += """---

## 📚 Dependencies

### Language Distribution
"""
        for lang, percent in documentation["dependencies"]["languages"].items():
            md += f"- {lang}: {percent:.1f}%\n"
        
        if documentation["dependencies"]["libraries"]:
            md += f"\n### Key Libraries\n{self._format_list(documentation['dependencies']['libraries'][:20])}\n"
        
        return md
    
    # Helper methods
    
    def _infer_description(self, repo_analysis: Dict[str, Any]) -> str:
        """Infer repository description from analysis."""
        domains = repo_analysis.get("analysis", {}).get("business_domains", {})
        if domains:
            primary_domain = max(domains.items(), key=lambda x: x[1].get("score", 0))
            return f"System focused on {primary_domain[0].replace('_', ' ')}"
        return "Legacy system requiring documentation"
    
    def _get_primary_language(self, language_stats: Dict[str, float]) -> str:
        """Get primary programming language."""
        if not language_stats:
            return "Unknown"
        return max(language_stats.items(), key=lambda x: x[1])[0]
    
    def _identify_technologies(self, repo_analysis: Dict[str, Any]) -> List[str]:
        """Identify key technologies used."""
        technologies = []
        
        # Add framework if detected
        framework = repo_analysis.get("repository_info", {}).get("framework")
        if framework:
            technologies.append(framework)
        
        # Add languages
        languages = list(repo_analysis.get("repository_info", {}).get("language_stats", {}).keys())
        technologies.extend(languages[:3])
        
        # Add patterns
        patterns = repo_analysis.get("analysis", {}).get("patterns", [])
        for pattern in patterns[:5]:
            if hasattr(pattern, 'name'):
                technologies.append(pattern.name)
        
        return technologies[:10]
    
    def _calculate_overall_complexity(self, analysis: Dict[str, Any]) -> str:
        """Calculate overall complexity rating."""
        complexity_metrics = analysis.get("complexity_metrics", {})
        avg_complexity = complexity_metrics.get("average_complexity", 0)
        
        if avg_complexity <= 5:
            return "Low"
        elif avg_complexity <= 10:
            return "Medium"
        elif avg_complexity <= 20:
            return "High"
        else:
            return "Very High"
    
    async def _identify_components(self, repo_analysis: Dict[str, Any]) -> Dict[str, Dict]:
        """Identify system components."""
        components = {}
        
        # Basic component identification based on directory structure
        files = repo_analysis.get("analysis", {}).get("relevant_files", [])
        
        directories = set()
        for file_context in files:
            path = Path(file_context.get("path", ""))
            if len(path.parts) > 0:
                directories.add(path.parts[0])
        
        for dir_name in directories:
            components[dir_name] = {
                "name": dir_name,
                "type": self._infer_component_type(dir_name, [])
            }
        
        return components
    
    def _infer_component_type(self, name: str, files: List) -> str:
        """Infer component type from name and files."""
        name_lower = name.lower()
        
        if any(x in name_lower for x in ["ui", "frontend", "web", "client"]):
            return "frontend"
        elif any(x in name_lower for x in ["api", "service", "backend", "server"]):
            return "backend"
        elif any(x in name_lower for x in ["db", "database", "data", "sql"]):
            return "database"
        elif any(x in name_lower for x in ["test", "spec"]):
            return "test"
        elif any(x in name_lower for x in ["doc", "docs"]):
            return "documentation"
        else:
            return "module"
    
    async def _identify_relationships(self, components: Dict, repo_analysis: Dict) -> List[Dict]:
        """Identify relationships between components."""
        relationships = []
        
        # Simple heuristic-based relationship detection
        component_names = list(components.keys())
        
        for i, comp1 in enumerate(component_names):
            for comp2 in component_names[i+1:]:
                # Check if components might be related
                if self._are_components_related(comp1, comp2, repo_analysis):
                    relationships.append({
                        "from": comp1,
                        "to": comp2,
                        "type": "uses"
                    })
        
        return relationships
    
    def _are_components_related(self, comp1: str, comp2: str, repo_analysis: Dict) -> bool:
        """Check if two components are related."""
        # Simple heuristic - would need more sophisticated analysis in production
        files = repo_analysis.get("analysis", {}).get("relevant_files", [])
        
        # Check for imports/dependencies between components
        for file_context in files:
            if comp1 in file_context.get("path", ""):
                deps = file_context.get("dependencies", [])
                if any(comp2 in dep for dep in deps):
                    return True
        
        return False
    
    def _generate_component_description(self, name: str, files: List) -> str:
        """Generate component description."""
        file_count = len(files)
        languages = set(f.get("language") for f in files if f.get("language"))
        
        return f"Module containing {file_count} files, primarily written in {', '.join(languages) if languages else 'various languages'}"
    
    def _get_module_language(self, files: List) -> str:
        """Get primary language for module."""
        language_counts = defaultdict(int)
        for f in files:
            if f.get("language"):
                language_counts[f["language"]] += 1
        
        if language_counts:
            return max(language_counts.items(), key=lambda x: x[1])[0]
        return "Unknown"
    
    async def _extract_key_classes(self, files: List) -> List[str]:
        """Extract key class names from files."""
        classes = []
        
        for file_context in files[:10]:  # Limit to first 10 files
            content = file_context.get("content", "")
            language = file_context.get("language", "")
            
            if language == "Python":
                # Extract Python classes
                class_pattern = re.compile(r'class\s+(\w+)')
                classes.extend(class_pattern.findall(content))
            elif language == "Java":
                # Extract Java classes
                class_pattern = re.compile(r'(?:public\s+)?class\s+(\w+)')
                classes.extend(class_pattern.findall(content))
        
        return list(set(classes))[:10]
    
    async def _extract_key_functions(self, files: List) -> List[str]:
        """Extract key function names from files."""
        functions = []
        
        for file_context in files[:10]:
            content = file_context.get("content", "")
            language = file_context.get("language", "")
            
            if language == "Python":
                func_pattern = re.compile(r'def\s+(\w+)')
                functions.extend(func_pattern.findall(content))
            elif language == "Java":
                func_pattern = re.compile(r'(?:public\s+)?(?:static\s+)?\w+\s+(\w+)\s*\(')
                functions.extend(func_pattern.findall(content))
        
        return list(set(functions))[:10]
    
    def _extract_module_dependencies(self, files: List) -> List[str]:
        """Extract module dependencies."""
        all_deps = []
        for f in files:
            all_deps.extend(f.get("dependencies", []))
        return list(set(all_deps))[:10]
    
    def _identify_module_patterns(self, files: List) -> List[str]:
        """Identify patterns in module."""
        all_patterns = []
        for f in files:
            all_patterns.extend(f.get("patterns", []))
        
        # Count pattern occurrences
        pattern_counts = defaultdict(int)
        for p in all_patterns:
            pattern_counts[p] += 1
        
        # Return most common patterns
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        return [p[0] for p in sorted_patterns[:5]]
    
    def _extract_table_info(self, sql: str) -> Optional[Dict[str, Any]]:
        """Extract table information from CREATE TABLE statement."""
        table_pattern = re.compile(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', re.IGNORECASE)
        match = table_pattern.search(sql)
        
        if match:
            return {
                "name": match.group(1),
                "columns": self._extract_columns(sql),
                "primary_key": self._extract_primary_key(sql),
                "foreign_keys": self._extract_foreign_keys(sql)
            }
        return None
    
    def _extract_columns(self, sql: str) -> List[Dict]:
        """Extract column definitions from SQL."""
        columns = []
        # Simplified column extraction - would need more robust parsing in production
        column_pattern = re.compile(r'(\w+)\s+(VARCHAR|INT|TEXT|DATE|TIMESTAMP|DECIMAL|BOOLEAN)', re.IGNORECASE)
        for match in column_pattern.finditer(sql):
            columns.append({
                "name": match.group(1),
                "type": match.group(2)
            })
        return columns
    
    def _extract_primary_key(self, sql: str) -> Optional[str]:
        """Extract primary key from SQL."""
        pk_pattern = re.compile(r'PRIMARY\s+KEY\s*\(([^)]+)\)', re.IGNORECASE)
        match = pk_pattern.search(sql)
        return match.group(1) if match else None
    
    def _extract_foreign_keys(self, sql: str) -> List[Dict]:
        """Extract foreign keys from SQL."""
        foreign_keys = []
        fk_pattern = re.compile(r'FOREIGN\s+KEY\s*\((\w+)\)\s+REFERENCES\s+(\w+)\s*\((\w+)\)', re.IGNORECASE)
        for match in fk_pattern.finditer(sql):
            foreign_keys.append({
                "column": match.group(1),
                "references_table": match.group(2),
                "references_column": match.group(3)
            })
        return foreign_keys
    
    def _extract_procedure_info(self, sql: str) -> Optional[Dict]:
        """Extract stored procedure information."""
        proc_pattern = re.compile(r'CREATE\s+PROCEDURE\s+(\w+)', re.IGNORECASE)
        match = proc_pattern.search(sql)
        
        if match:
            return {
                "name": match.group(1),
                "parameters": self._extract_procedure_params(sql)
            }
        return None
    
    def _extract_procedure_params(self, sql: str) -> List[str]:
        """Extract procedure parameters."""
        # Simplified extraction
        param_pattern = re.compile(r'\(([^)]+)\)')
        match = param_pattern.search(sql)
        if match:
            params = match.group(1).split(',')
            return [p.strip() for p in params]
        return []
    
    def _extract_view_info(self, sql: str) -> Optional[Dict]:
        """Extract view information."""
        view_pattern = re.compile(r'CREATE\s+VIEW\s+(\w+)', re.IGNORECASE)
        match = view_pattern.search(sql)
        
        if match:
            return {
                "name": match.group(1),
                "definition": sql
            }
        return None
    
    def _generate_er_diagram(self, tables: List[Dict]) -> str:
        """Generate ER diagram in Mermaid format."""
        diagram = "erDiagram\n"
        
        for table in tables:
            # Add table
            diagram += f"    {table['name']} {{\n"
            
            # Add columns
            for column in table.get("columns", [])[:10]:
                diagram += f"        {column['type']} {column['name']}\n"
            
            diagram += "    }\n"
            
            # Add relationships
            for fk in table.get("foreign_keys", []):
                diagram += f"    {table['name']} ||--o{{ {fk['references_table']} : has\n"
        
        return diagram
    
    def _extract_python_endpoints(self, content: str) -> List[Dict]:
        """Extract Python API endpoints."""
        endpoints = []
        
        # Flask patterns
        flask_pattern = re.compile(r'@app\.route\([\'"]([^\'\"]+)[\'"].*?methods=\[([^\]]+)\]')
        for match in flask_pattern.finditer(content):
            methods = match.group(2).replace("'", "").replace('"', "").split(',')
            for method in methods:
                endpoints.append({
                    "path": match.group(1),
                    "method": method.strip(),
                    "framework": "Flask"
                })
        
        # FastAPI patterns
        fastapi_patterns = [
            (re.compile(r'@app\.get\([\'"]([^\'\"]+)[\'"]'), 'GET'),
            (re.compile(r'@app\.post\([\'"]([^\'\"]+)[\'"]'), 'POST'),
            (re.compile(r'@app\.put\([\'"]([^\'\"]+)[\'"]'), 'PUT'),
            (re.compile(r'@app\.delete\([\'"]([^\'\"]+)[\'"]'), 'DELETE')
        ]
        
        for pattern, method in fastapi_patterns:
            for match in pattern.finditer(content):
                endpoints.append({
                    "path": match.group(1),
                    "method": method,
                    "framework": "FastAPI"
                })
        
        return endpoints
    
    def _extract_java_endpoints(self, content: str) -> List[Dict]:
        """Extract Java API endpoints."""
        endpoints = []
        
        # Spring Boot patterns
        mapping_patterns = [
            (re.compile(r'@GetMapping\([\'"]([^\'\"]+)[\'"]'), 'GET'),
            (re.compile(r'@PostMapping\([\'"]([^\'\"]+)[\'"]'), 'POST'),
            (re.compile(r'@PutMapping\([\'"]([^\'\"]+)[\'"]'), 'PUT'),
            (re.compile(r'@DeleteMapping\([\'"]([^\'\"]+)[\'"]'), 'DELETE'),
            (re.compile(r'@RequestMapping\([\'"]([^\'\"]+)[\'"]'), 'GET')
        ]
        
        for pattern, method in mapping_patterns:
            for match in pattern.finditer(content):
                endpoints.append({
                    "path": match.group(1),
                    "method": method,
                    "framework": "Spring Boot"
                })
        
        return endpoints
    
    def _extract_js_endpoints(self, content: str) -> List[Dict]:
        """Extract JavaScript API endpoints."""
        endpoints = []
        
        # Express patterns
        express_patterns = [
            (re.compile(r'app\.get\([\'"]([^\'\"]+)[\'"]'), 'GET'),
            (re.compile(r'app\.post\([\'"]([^\'\"]+)[\'"]'), 'POST'),
            (re.compile(r'app\.put\([\'"]([^\'\"]+)[\'"]'), 'PUT'),
            (re.compile(r'app\.delete\([\'"]([^\'\"]+)[\'"]'), 'DELETE'),
            (re.compile(r'router\.get\([\'"]([^\'\"]+)[\'"]'), 'GET'),
            (re.compile(r'router\.post\([\'"]([^\'\"]+)[\'"]'), 'POST')
        ]
        
        for pattern, method in express_patterns:
            for match in pattern.finditer(content):
                endpoints.append({
                    "path": match.group(1),
                    "method": method,
                    "framework": "Express"
                })
        
        return endpoints
    
    def _deduplicate_apis(self, apis: List[Dict]) -> List[Dict]:
        """Deduplicate API endpoints."""
        seen = set()
        unique_apis = []
        
        for api in apis:
            key = f"{api['method']}:{api['path']}"
            if key not in seen:
                seen.add(key)
                unique_apis.append(api)
        
        return unique_apis
    
    async def _identify_business_processes(self, repo_analysis: Dict[str, Any]) -> List[Dict]:
        """Identify key business processes from code."""
        processes = []
        
        # Look for common business process patterns
        business_keywords = {
            "authentication": ["login", "authenticate", "authorize"],
            "payment": ["payment", "charge", "billing", "invoice"],
            "registration": ["register", "signup", "create_account"],
            "order": ["order", "purchase", "checkout"],
            "claim": ["claim", "submit_claim", "process_claim"]
        }
        
        files = repo_analysis.get("analysis", {}).get("relevant_files", [])
        
        for process_name, keywords in business_keywords.items():
            process_files = []
            for file_context in files:
                content_lower = file_context.get("content", "").lower()
                if any(keyword in content_lower for keyword in keywords):
                    process_files.append(file_context)
            
            if process_files:
                processes.append({
                    "name": process_name.replace("_", " ").title(),
                    "description": f"Process for handling {process_name}",
                    "files": process_files[:5],
                    "components": self._extract_process_components(process_files),
                    "steps": self._extract_process_steps(process_files),
                    "flow": self._extract_process_flow(process_files)
                })
        
        return processes[:5]
    
    def _extract_process_components(self, files: List) -> List[str]:
        """Extract components involved in process."""
        components = set()
        for f in files:
            path_parts = Path(f.get("path", "")).parts
            if path_parts:
                components.add(path_parts[0])
        return list(components)[:5]
    
    def _extract_process_steps(self, files: List) -> List[Dict]:
        """Extract process steps for sequence diagram."""
        # Simplified extraction - would need more sophisticated analysis
        return [
            {"type": "call", "from": "Client", "to": "API", "action": "Request"},
            {"type": "call", "from": "API", "to": "Service", "action": "Process"},
            {"type": "call", "from": "Service", "to": "Database", "action": "Query"},
            {"type": "call", "from": "Database", "to": "Service", "action": "Result"},
            {"type": "call", "from": "Service", "to": "API", "action": "Response"},
            {"type": "call", "from": "API", "to": "Client", "action": "Result"}
        ]
    
    def _extract_process_flow(self, files: List) -> List[Dict]:
        """Extract process flow for flowchart."""
        # Simplified flow - would need more sophisticated analysis
        return [
            {"type": "start", "label": "Start"},
            {"type": "process", "label": "Receive Request"},
            {"type": "decision", "label": "Valid Request?"},
            {"type": "process", "label": "Process Request"},
            {"type": "process", "label": "Return Response"},
            {"type": "end", "label": "End"}
        ]
    
    def _extract_python_dependencies(self, content: str) -> List[str]:
        """Extract Python dependencies."""
        deps = []
        
        # requirements.txt format
        if "==" in content or ">=" in content:
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name
                    pkg = re.split(r'[><=!]', line)[0].strip()
                    if pkg:
                        deps.append(pkg)
        
        # pyproject.toml format
        elif "[tool.poetry.dependencies]" in content:
            # Simplified extraction
            in_deps = False
            lines = content.split('\n')
            for line in lines:
                if "[tool.poetry.dependencies]" in line:
                    in_deps = True
                elif line.startswith('[') and in_deps:
                    break
                elif in_deps and '=' in line:
                    pkg = line.split('=')[0].strip()
                    if pkg and pkg != "python":
                        deps.append(pkg)
        
        return deps
    
    def _extract_java_dependencies(self, content: str) -> List[str]:
        """Extract Java dependencies."""
        deps = []
        
        # Maven pom.xml
        if "<dependency>" in content:
            dep_pattern = re.compile(r'<artifactId>([^<]+)</artifactId>')
            deps.extend(dep_pattern.findall(content))
        
        # Gradle build.gradle
        elif "dependencies {" in content:
            # Simplified extraction
            dep_pattern = re.compile(r'(?:compile|implementation)\s+[\'"]([^:]+):([^:]+)')
            for match in dep_pattern.finditer(content):
                deps.append(f"{match.group(1)}.{match.group(2)}")
        
        return deps
    
    def _extract_js_dependencies(self, content: str) -> List[str]:
        """Extract JavaScript dependencies."""
        deps = []
        
        try:
            # Parse package.json
            import json
            package = json.loads(content)
            
            deps.extend(package.get("dependencies", {}).keys())
            deps.extend(package.get("devDependencies", {}).keys())
        except:
            pass
        
        return deps
    
    def _generate_dependency_graph(self, dependencies: Dict) -> str:
        """Generate dependency graph in Mermaid format."""
        graph = "graph LR\n"
        graph += "    App[Application]\n"
        
        # Add top libraries
        for lib in dependencies["libraries"][:10]:
            lib_id = lib.replace("-", "_").replace(".", "_")
            graph += f"    App --> {lib_id}[{lib}]\n"
        
        return graph
    
    def _get_debt_rating(self, score: int) -> str:
        """Get technical debt rating."""
        if score <= 20:
            return "Good"
        elif score <= 40:
            return "Fair"
        elif score <= 60:
            return "Poor"
        else:
            return "Critical"
    
    def _format_list(self, items: List) -> str:
        """Format list for markdown."""
        if not items:
            return "- None identified\n"
        return "\n".join(f"- {item}" for item in items) + "\n"
    
    async def _save_documentation(self, documentation: Dict, repo_info: Dict):
        """Save documentation to file."""
        # Save to memory for retrieval
        await self.save_to_memory(
            f"documentation_{repo_info.get('name', 'unknown')}",
            documentation,
            "long_term"
        )
        
        logger.info(f"Documentation generated for {repo_info.get('name', 'repository')}")
    
    async def _analyze_local_repo(self, repo_path: str) -> Dict[str, Any]:
        """Analyze local repository without cloning."""
        # Simplified local analysis - would use repo_analyst methods
        return {
            "repository_info": {"name": Path(repo_path).name},
            "analysis": {"relevant_files": [], "patterns": []}
        }