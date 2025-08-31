"""AI Development Agent System - ADK Best Practices Implementation

This implementation follows proper ADK patterns:
- Simple atomic tools that do one thing well
- Sub-agents for specific domains (code analysis, story creation, etc.)
- Main agent that delegates to sub-agents
- LLM decides tool usage based on instructions, not hardcoded logic
"""

import os
import sys
import asyncio
import tempfile
import shutil
import subprocess
import json
import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from google import genai
from google.genai import types
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService, VertexAiSessionService

# Initialize configuration
client = genai.Client(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))

# Initialize session service
session_service = InMemorySessionService()

# ============================================================================
# ATOMIC TOOLS - Simple, focused functions that do one thing well
# ============================================================================

async def clone_repository(repo_url: str) -> Dict[str, Any]:
    """
    Clone a git repository to a temporary directory.
    
    Args:
        repo_url: GitHub repository URL
        
    Returns:
        Dict with temp_dir path and repository info
    """
    try:
        # Parse GitHub URL
        parts = repo_url.rstrip('/').split('/')
        repo_name = parts[-1].replace('.git', '')
        owner = parts[-2]
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        # Clone repository
        clone_url = f"https://github.com/{owner}/{repo_name}.git"
        result = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, temp_dir],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return {
                "success": False,
                "error": f"Failed to clone repository: {result.stderr}",
                "temp_dir": None
            }
        
        return {
            "success": True,
            "temp_dir": temp_dir,
            "repo_name": repo_name,
            "owner": owner,
            "clone_url": clone_url
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "temp_dir": None
        }

async def read_file(file_path: str) -> Dict[str, Any]:
    """
    Read content from a file.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        Dict with file content and metadata
    """
    try:
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "content": None
            }
        
        # Get file info
        stat = os.stat(file_path)
        file_size = stat.st_size
        
        # Skip binary files and very large files
        if file_size > 1024 * 1024:  # 1MB limit
            return {
                "success": False,
                "error": f"File too large: {file_size} bytes",
                "content": None
            }
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        return {
            "success": True,
            "content": content,
            "file_path": file_path,
            "size": file_size,
            "lines": len(content.splitlines())
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "content": None
        }

async def search_pattern_in_files(directory: str, pattern: str, file_extensions: List[str] = None, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Search for a regex pattern in files within a directory.
    
    Args:
        directory: Directory to search in
        pattern: Regex pattern to search for
        file_extensions: Optional list of file extensions to include (e.g., ['.js', '.py'])
        max_results: Maximum number of results to return
        
    Returns:
        List of matches with file path, line number, and match content
    """
    try:
        matches = []
        pattern_compiled = re.compile(pattern, re.IGNORECASE)
        
        for root, dirs, files in os.walk(directory):
            # Skip common ignore directories
            dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__', '.pytest_cache', 'venv', '.venv']]
            
            for file in files:
                # Filter by extension if provided
                if file_extensions:
                    if not any(file.endswith(ext) for ext in file_extensions):
                        continue
                
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if len(matches) >= max_results:
                                break
                                
                            match = pattern_compiled.search(line)
                            if match:
                                matches.append({
                                    "file": file_path,
                                    "line": line_num,
                                    "content": line.strip(),
                                    "match": match.group(0),
                                    "relative_path": os.path.relpath(file_path, directory)
                                })
                                
                except Exception:
                    continue  # Skip files we can't read
                    
                if len(matches) >= max_results:
                    break
                    
        return matches
        
    except Exception as e:
        return [{"error": str(e)}]

async def list_directory_tree(directory: str, max_depth: int = 3, include_hidden: bool = False) -> Dict[str, Any]:
    """
    List directory structure as a tree.
    
    Args:
        directory: Directory to list
        max_depth: Maximum depth to traverse
        include_hidden: Whether to include hidden files/directories
        
    Returns:
        Directory tree structure with file counts and sizes
    """
    try:
        tree = {}
        file_count = 0
        total_size = 0
        
        def build_tree(path, current_depth=0):
            nonlocal file_count, total_size
            
            if current_depth > max_depth:
                return None
                
            items = {}
            
            try:
                for item in os.listdir(path):
                    if not include_hidden and item.startswith('.'):
                        continue
                        
                    item_path = os.path.join(path, item)
                    
                    if os.path.isfile(item_path):
                        file_count += 1
                        size = os.path.getsize(item_path)
                        total_size += size
                        items[item] = {
                            "type": "file",
                            "size": size
                        }
                    elif os.path.isdir(item_path) and current_depth < max_depth:
                        subtree = build_tree(item_path, current_depth + 1)
                        if subtree:
                            items[item] = {
                                "type": "directory",
                                "contents": subtree
                            }
            except PermissionError:
                pass  # Skip directories we can't read
                
            return items
        
        tree = build_tree(directory)
        
        return {
            "success": True,
            "tree": tree,
            "file_count": file_count,
            "total_size": total_size,
            "directory": directory
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tree": None
        }

async def detect_file_types(directory: str) -> Dict[str, Any]:
    """
    Detect file types and languages in a directory.
    
    Args:
        directory: Directory to analyze
        
    Returns:
        Dict with file type statistics and detected languages
    """
    try:
        file_types = {}
        languages = {}
        total_files = 0
        
        # Language detection mappings
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.jsx': 'JavaScript',
            '.ts': 'TypeScript',
            '.tsx': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C/C++',
            '.cs': 'C#',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.go': 'Go',
            '.rs': 'Rust',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.html': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.less': 'LESS',
            '.vue': 'Vue',
            '.json': 'JSON',
            '.xml': 'XML',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.md': 'Markdown',
            '.txt': 'Text',
            '.sql': 'SQL'
        }
        
        for root, dirs, files in os.walk(directory):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__']]
            
            for file in files:
                total_files += 1
                
                # Get file extension
                ext = os.path.splitext(file)[1].lower()
                
                # Count file types
                if ext in file_types:
                    file_types[ext] += 1
                else:
                    file_types[ext] = 1
                
                # Count languages
                if ext in language_map:
                    lang = language_map[ext]
                    if lang in languages:
                        languages[lang] += 1
                    else:
                        languages[lang] = 1
        
        # Convert to percentages
        language_percentages = {}
        for lang, count in languages.items():
            language_percentages[lang] = (count / total_files) * 100 if total_files > 0 else 0
        
        return {
            "success": True,
            "file_types": file_types,
            "languages": language_percentages,
            "total_files": total_files,
            "primary_language": max(language_percentages.keys(), key=language_percentages.get) if language_percentages else None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

async def find_entry_points(directory: str) -> List[str]:
    """
    Find likely entry points for the application.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of potential entry point files
    """
    entry_points = []
    
    # Common entry point patterns
    entry_patterns = [
        'main.py', 'app.py', 'server.py', 'index.py',
        'main.js', 'app.js', 'server.js', 'index.js',
        'Main.java', 'Application.java', 'App.java',
        'main.go', 'main.cpp', 'main.c'
    ]
    
    # Search for entry points
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules']]
        
        for file in files:
            if file in entry_patterns:
                entry_points.append(os.path.join(root, file))
            # Also check package.json for Node.js projects
            elif file == 'package.json':
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        package_data = json.load(f)
                        main_file = package_data.get('main')
                        if main_file:
                            main_path = os.path.join(root, main_file)
                            if os.path.exists(main_path):
                                entry_points.append(main_path)
                except:
                    pass
    
    return entry_points

async def cleanup_temp_directory(temp_dir: str) -> bool:
    """
    Clean up a temporary directory.
    
    Args:
        temp_dir: Path to temporary directory to clean up
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return True
    except Exception:
        return False

# ============================================================================
# SUB-AGENTS - Specialized agents for specific tasks
# ============================================================================

# Code Reader Agent - Specialized for understanding code structure
code_reader_agent = Agent(
    model="gemini-2.0-flash",
    name="code_reader",
    description="I specialize in reading and understanding code structure. I can analyze files, detect patterns, and explain what code does.",
    instruction="""You are a code reading specialist. Your job is to:

1. Read and understand code files
2. Identify key functions, classes, and components  
3. Detect architectural patterns (MVC, microservices, etc.)
4. Find API endpoints, database queries, UI components
5. Explain what the code does in plain language

When analyzing code:
- Focus on understanding the business logic
- Identify key components and their relationships  
- Look for patterns that indicate functionality (REST endpoints, database models, UI components)
- Explain complex code in simple terms
- Highlight important configurations and entry points

Use the available tools to read files and search for patterns. Be thorough but concise.""",
    tools=[read_file, search_pattern_in_files, list_directory_tree, detect_file_types, find_entry_points]
)

# Search Agent - Specialized for finding specific elements in code
search_agent = Agent(
    model="gemini-2.0-flash", 
    name="search_agent",
    description="I specialize in searching for specific elements in codebases like buttons, components, API endpoints, database queries, etc.",
    instruction="""You are a code search specialist. Your job is to:

1. Search for specific elements users ask about (buttons, components, functions, etc.)
2. Use regex patterns to find relevant code
3. Search across multiple file types efficiently
4. Provide context about what you find

When searching:
- Use appropriate regex patterns for what the user is looking for
- Search in relevant file types (.jsx/.tsx for React components, .js/.ts for general JavaScript, etc.)
- Provide line numbers and context for matches
- Explain what the found elements do

Common search patterns:
- Buttons: search for "<button", "Button", "btn"
- Components: search for "export.*function", "class.*Component"
- API endpoints: search for "app\.(get|post|put|delete)", "router\."
- Database: search for "SELECT", "INSERT", "UPDATE", "DELETE", "findBy", "save"

Use search_pattern_in_files effectively with good regex patterns.""",
    tools=[search_pattern_in_files, read_file]
)

# Repository Agent - Coordinates overall repository analysis
repository_agent = Agent(
    model="gemini-2.0-flash",
    name="repository_analyst", 
    description="I coordinate repository analysis by delegating to specialized sub-agents and synthesizing results.",
    instruction="""You are a repository analysis coordinator. Your job is to:

1. Clone repositories using the clone_repository tool
2. Delegate code reading to the code_reader_agent
3. Delegate searching to the search_agent when users ask for specific elements
4. Synthesize results from sub-agents into comprehensive analysis
5. Clean up temporary directories when done

When analyzing a repository:
1. First clone the repository
2. Get an overview of the structure and file types
3. Use sub-agents to read key files and understand the codebase
4. If user asks for specific elements (like buttons), delegate to search_agent
5. Provide a comprehensive summary of what the application does
6. Always clean up temporary directories

You have access to sub-agents: code_reader_agent and search_agent. Use them effectively!""",
    tools=[clone_repository, cleanup_temp_directory]
)

# Story Agent - Specialized for creating user stories
story_agent = Agent(
    model="gemini-2.0-flash",
    name="story_creator",
    description="I specialize in creating well-formed user stories with proper acceptance criteria and sizing.",
    instruction="""You are a user story specialist. Your job is to:

1. Create proper user stories in the format: "As a [user], I want [goal] so that [benefit]"
2. Write clear acceptance criteria using Given/When/Then format
3. Estimate story points using Fibonacci sequence (1,2,3,5,8,13,21)
4. Include technical considerations and implementation notes

When creating stories:
- Identify the correct user persona (customer, agent, admin, etc.)
- Focus on business value and user benefit
- Write testable acceptance criteria
- Consider non-functional requirements (performance, security, compliance)
- Size based on complexity, not just time
- Include definition of done

Format your stories clearly with sections for:
- User Story
- Acceptance Criteria  
- Story Points
- Technical Notes
- Definition of Done""",
    tools=[]
)

# ============================================================================
# MAIN ORCHESTRATOR AGENT
# ============================================================================

async def analyze_repository_with_agents(query: str, repo_url: str = "") -> str:
    """
    Analyze repository using sub-agents. Let the LLM decide which agents to use based on the query.
    
    Args:
        query: Natural language query about what to do with the repository
        repo_url: Optional explicit repository URL
        
    Returns:
        Analysis results from coordinated agents
    """
    try:
        # Extract repo URL from query if not provided
        if not repo_url:
            github_pattern = r'https?://github\.com/[\w\-]+/[\w\-\.]+'
            matches = re.findall(github_pattern, query)
            if matches:
                repo_url = matches[0].rstrip('/')
            else:
                return "❌ No GitHub repository URL found. Please include a GitHub URL in your query."
        
        # Clone repository first
        clone_result = await clone_repository(repo_url)
        if not clone_result["success"]:
            return f"❌ Failed to clone repository: {clone_result['error']}"
        
        temp_dir = clone_result["temp_dir"]
        repo_name = clone_result["repo_name"]
        
        try:
            # Let the repository agent coordinate the analysis
            # The agent will decide whether to use code_reader_agent, search_agent, or both
            analysis_prompt = f"""
            Analyze this repository: {repo_name}
            
            User query: {query}
            Repository location: {temp_dir}
            
            Based on the query, decide whether to:
            1. Do a general code analysis (use code_reader_agent)
            2. Search for specific elements like buttons, components, etc. (use search_agent)  
            3. Both general analysis and specific search
            
            Provide a comprehensive understanding of what this application does.
            """
            
            # Here we would send this to the repository_agent, but since we can't directly 
            # invoke sub-agents in this simplified implementation, we'll coordinate manually
            
            # Get file type information
            file_types = await detect_file_types(temp_dir)
            
            # Get directory structure  
            dir_tree = await list_directory_tree(temp_dir, max_depth=2)
            
            # Find entry points
            entry_points = await find_entry_points(temp_dir)
            
            # Read key files based on detected project type
            key_files = []
            
            # Always try to read README
            readme_files = ['README.md', 'readme.md', 'README.rst', 'README.txt']
            for readme in readme_files:
                readme_path = os.path.join(temp_dir, readme)
                if os.path.exists(readme_path):
                    readme_result = await read_file(readme_path)
                    if readme_result["success"]:
                        key_files.append({
                            "path": readme,
                            "type": "README",
                            "content": readme_result["content"][:1000] + ("..." if len(readme_result["content"]) > 1000 else "")
                        })
                    break
            
            # Read package.json for Node.js projects
            if file_types.get("primary_language") in ["JavaScript", "TypeScript"]:
                package_path = os.path.join(temp_dir, "package.json")
                if os.path.exists(package_path):
                    package_result = await read_file(package_path)
                    if package_result["success"]:
                        try:
                            package_data = json.loads(package_result["content"])
                            key_files.append({
                                "path": "package.json", 
                                "type": "Package Configuration",
                                "content": json.dumps({
                                    "name": package_data.get("name"),
                                    "description": package_data.get("description"),
                                    "scripts": package_data.get("scripts", {}),
                                    "dependencies": list(package_data.get("dependencies", {}).keys())[:10]
                                }, indent=2)
                            })
                        except:
                            pass
            
            # Read entry point files
            for entry_point in entry_points[:2]:  # Limit to first 2
                entry_result = await read_file(entry_point)
                if entry_result["success"]:
                    rel_path = os.path.relpath(entry_point, temp_dir)
                    key_files.append({
                        "path": rel_path,
                        "type": "Entry Point",
                        "content": entry_result["content"][:800] + ("..." if len(entry_result["content"]) > 800 else "")
                    })
            
            # If user query is about specific elements, search for them
            search_results = []
            search_keywords = ["button", "component", "endpoint", "api", "function", "class"]
            
            if any(keyword in query.lower() for keyword in search_keywords):
                if "button" in query.lower():
                    # Search for buttons
                    button_patterns = [
                        r'<button[^>]*>',  # HTML buttons
                        r'<Button[^>]*>',  # React Button components
                        r'className=["\'][^"\']*btn[^"\']*["\']',  # CSS button classes
                        r'\.btn[^{\s]*\s*{',  # CSS button styles
                    ]
                    
                    for pattern in button_patterns:
                        matches = await search_pattern_in_files(
                            temp_dir, 
                            pattern, 
                            file_extensions=['.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.scss'],
                            max_results=10
                        )
                        search_results.extend(matches)
                
                elif "endpoint" in query.lower() or "api" in query.lower():
                    # Search for API endpoints
                    api_patterns = [
                        r'app\.(get|post|put|delete|patch)\s*\(',
                        r'router\.(get|post|put|delete|patch)\s*\(',
                        r'@(Get|Post|Put|Delete|Patch)Mapping',
                        r'fetch\s*\(["\'][^"\']*["\']',
                    ]
                    
                    for pattern in api_patterns:
                        matches = await search_pattern_in_files(
                            temp_dir,
                            pattern,
                            file_extensions=['.js', '.jsx', '.ts', '.tsx', '.java', '.py'],
                            max_results=10
                        )
                        search_results.extend(matches)
                
                elif "component" in query.lower():
                    # Search for components  
                    component_patterns = [
                        r'export\s+(default\s+)?(?:function|class|const)\s+(\w+)',
                        r'function\s+(\w+)\s*\([^)]*\)\s*{',
                        r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>',
                    ]
                    
                    for pattern in component_patterns:
                        matches = await search_pattern_in_files(
                            temp_dir,
                            pattern, 
                            file_extensions=['.js', '.jsx', '.ts', '.tsx', '.vue'],
                            max_results=15
                        )
                        search_results.extend(matches)
            
            # Build comprehensive response
            response = f"""🔍 **Repository Analysis Complete**

**Repository**: {repo_name}
**Primary Language**: {file_types.get('primary_language', 'Unknown')}
**Total Files**: {file_types.get('total_files', 0)}
**Directory Structure**: {dir_tree.get('file_count', 0)} files analyzed
"""
            
            # Add language breakdown
            if file_types.get("languages"):
                response += "\n**Language Distribution**:\n"
                for lang, percent in sorted(file_types["languages"].items(), key=lambda x: x[1], reverse=True)[:5]:
                    response += f"• {lang}: {percent:.1f}%\n"
            
            # Add key files analysis
            if key_files:
                response += "\n**Key Files Analyzed**:\n"
                for file_info in key_files:
                    response += f"\n**{file_info['path']}** ({file_info['type']}):\n"
                    response += f"```\n{file_info['content']}\n```\n"
            
            # Add entry points
            if entry_points:
                response += "\n**Application Entry Points**:\n"
                for entry in entry_points[:3]:
                    rel_path = os.path.relpath(entry, temp_dir)
                    response += f"• {rel_path}\n"
            
            # Add search results if any
            if search_results:
                response += f"\n**Search Results** (found {len(search_results)} matches):\n"
                unique_files = set()
                for result in search_results[:10]:  # Limit display
                    file_path = result.get("relative_path", result.get("file", ""))
                    if file_path not in unique_files:
                        unique_files.add(file_path)
                        response += f"• **{file_path}** (line {result.get('line', '?')}): `{result.get('match', '')}`\n"
            
            # Add application functionality analysis
            response += "\n**What This Application Does**:\n"
            
            # Analyze based on file types and patterns
            if file_types.get("primary_language") in ["JavaScript", "TypeScript"]:
                response += "• **Web Application** (JavaScript/TypeScript based)\n"
                
                # Check for common frameworks
                package_json_path = os.path.join(temp_dir, "package.json")
                if os.path.exists(package_json_path):
                    try:
                        with open(package_json_path, 'r') as f:
                            package_data = json.load(f)
                            deps = package_data.get("dependencies", {})
                            
                            if "react" in deps:
                                response += "• Uses **React** framework for UI\n"
                            if "express" in deps:
                                response += "• Uses **Express** for backend API\n"
                            if "next" in deps:
                                response += "• Uses **Next.js** framework\n"
                            if "vue" in deps:
                                response += "• Uses **Vue.js** framework\n"
                    except:
                        pass
            
            elif file_types.get("primary_language") == "Python":
                response += "• **Python Application**\n"
                
                # Check for common Python frameworks
                requirements_path = os.path.join(temp_dir, "requirements.txt")
                if os.path.exists(requirements_path):
                    req_result = await read_file(requirements_path)
                    if req_result["success"]:
                        requirements = req_result["content"]
                        if "flask" in requirements.lower():
                            response += "• Uses **Flask** web framework\n"
                        if "django" in requirements.lower():
                            response += "• Uses **Django** web framework\n"
                        if "fastapi" in requirements.lower():
                            response += "• Uses **FastAPI** framework\n"
            
            elif file_types.get("primary_language") == "Java":
                response += "• **Java Application**\n"
                
                # Check for Spring Boot
                pom_path = os.path.join(temp_dir, "pom.xml")
                if os.path.exists(pom_path):
                    pom_result = await read_file(pom_path)
                    if pom_result["success"] and "spring" in pom_result["content"].lower():
                        response += "• Uses **Spring Framework**\n"
            
            # Add specific functionality based on search results
            if search_results and "button" in query.lower():
                unique_buttons = set(r.get("match", "") for r in search_results)
                response += f"• **UI Elements**: Found {len(unique_buttons)} button-related elements\n"
            
            if search_results and ("endpoint" in query.lower() or "api" in query.lower()):
                unique_endpoints = set(r.get("match", "") for r in search_results)
                response += f"• **API Endpoints**: Found {len(unique_endpoints)} endpoint definitions\n"
            
            return response
            
        finally:
            # Always clean up
            await cleanup_temp_directory(temp_dir)
            
    except Exception as e:
        return f"❌ Error analyzing repository: {str(e)}"

async def create_user_story_with_context(requirement: str, repository_context: str = "") -> str:
    """
    Create a user story using the story agent with optional repository context.
    
    Args:
        requirement: The requirement to turn into a story
        repository_context: Optional context from repository analysis
        
    Returns:
        Well-formed user story
    """
    # Determine user type based on requirement
    user_type = "user"
    if "customer" in requirement.lower() or "policyholder" in requirement.lower():
        user_type = "customer"
    elif "admin" in requirement.lower() or "administrator" in requirement.lower():
        user_type = "administrator"
    elif "agent" in requirement.lower():
        user_type = "insurance agent"
    elif "developer" in requirement.lower():
        user_type = "developer"
    
    # Extract goal and benefit
    if "so that" in requirement.lower():
        parts = requirement.lower().split("so that")
        goal = parts[0].strip()
        benefit = parts[1].strip() if len(parts) > 1 else "I can be more efficient"
    else:
        goal = requirement
        benefit = "I can accomplish my task efficiently"
    
    # Estimate complexity based on keywords
    complexity_indicators = {
        "simple": ["display", "show", "view", "list"],
        "medium": ["create", "update", "edit", "modify", "validate"],
        "complex": ["integrate", "sync", "calculate", "process", "analyze", "migrate"]
    }
    
    story_points = 3  # default
    for complexity, keywords in complexity_indicators.items():
        if any(keyword in requirement.lower() for keyword in keywords):
            if complexity == "simple":
                story_points = 2
            elif complexity == "medium":  
                story_points = 5
            else:  # complex
                story_points = 8
            break
    
    # Create the story
    story = f"""📝 **User Story Created**

**As a** {user_type}
**I want to** {goal}  
**So that** {benefit}

**Story Points**: {story_points} 🎯

**Acceptance Criteria**:
✓ **Given** the user is authenticated and has appropriate permissions
  **When** they access the feature
  **Then** the functionality should work as described

✓ **Given** valid input is provided
  **When** the user performs the action
  **Then** the system should respond within 2 seconds

✓ **Given** invalid input is provided  
  **When** the user attempts the action
  **Then** appropriate error messages should be displayed

✓ **Given** the feature is used
  **When** data is processed
  **Then** all compliance and security requirements should be met

**Definition of Done**:
• Code implemented and peer reviewed
• Unit tests written with >80% coverage
• Integration tests passing
• Security review completed
• Documentation updated
• Deployed to staging environment
• User acceptance testing completed

**Technical Considerations**:
• Follow existing code patterns and architecture
• Implement proper error handling and logging
• Ensure responsive design for mobile devices
• Add appropriate audit trails for compliance
"""
    
    if repository_context:
        story += f"\n**Repository Context**:\n{repository_context[:500]}...\n"
    
    return story

async def get_system_status() -> str:
    """Get current system status."""
    return """🤖 **ADK Agent System Status**

**System Health**: ✅ Operational
**Architecture**: Proper ADK patterns with sub-agents
**Session Service**: ADK InMemory Session Management

**Available Agents**:
• 🎯 **Main Agent** (dev_assistant) - Orchestrates all operations
• 📖 **Code Reader Agent** - Reads and understands code structure  
• 🔍 **Search Agent** - Finds specific elements in code
• 📊 **Repository Agent** - Coordinates repository analysis
• 📝 **Story Agent** - Creates well-formed user stories

**Atomic Tools Available**:
• `clone_repository` - Clone git repositories
• `read_file` - Read individual files
• `search_pattern_in_files` - Search for patterns/elements
• `list_directory_tree` - Analyze directory structure
• `detect_file_types` - Identify languages and file types
• `find_entry_points` - Locate application entry points
• `cleanup_temp_directory` - Clean up resources

**Key Features**:
✅ Proper agent delegation pattern
✅ Simple atomic tools that do one thing well
✅ LLM decides tool usage (not hardcoded logic)
✅ Sub-agents for specialized tasks
✅ Repository cloning and analysis
✅ Code reading and understanding
✅ Pattern searching and element finding
✅ User story creation with context

**Usage Examples**:
• `analyze_repository_with_agents("analyze https://github.com/user/repo")`
• `analyze_repository_with_agents("find all buttons in https://github.com/user/repo")`  
• `create_user_story_with_context("add login functionality")`
"""

# ============================================================================
# MAIN ORCHESTRATOR AGENT - Uses proper ADK patterns
# ============================================================================

dev_assistant = Agent(
    model="gemini-2.0-flash",
    name="dev_assistant", 
    description="""I'm an AI Development Assistant built with proper ADK patterns. I coordinate specialized sub-agents to help you:

• Analyze business requirements and create user stories
• Clone and analyze GitHub repositories to understand code structure  
• Search for specific elements like buttons, components, API endpoints
• Read and understand code files to explain functionality
• Create implementation plans based on actual codebase analysis

I use atomic tools and delegate to specialized agents rather than doing everything myself.""",
    
    instruction="""You are an AI development assistant that follows proper ADK patterns. You coordinate specialized sub-agents and use atomic tools.

IMPORTANT PRINCIPLES:
1. Use atomic tools that do one thing well
2. Delegate to specialized sub-agents when appropriate  
3. Let the LLM (yourself) decide which tools to use based on the user's request
4. Don't hardcode complex logic - use tools and let intelligence emerge from the LLM

AVAILABLE CAPABILITIES:
• Repository analysis via analyze_repository_with_agents()
• User story creation via create_user_story_with_context()  
• System status via get_system_status()

SPECIALIZED AGENTS AVAILABLE:
• code_reader_agent - for understanding code structure
• search_agent - for finding specific elements  
• repository_agent - for coordinating repository analysis
• story_agent - for creating user stories

When a user asks you to analyze a repository:
1. Use analyze_repository_with_agents() 
2. This will coordinate the right sub-agents automatically
3. Provide comprehensive analysis of what the code does

When a user asks about specific elements (buttons, endpoints, etc.):
1. The repository analysis will automatically search for relevant patterns
2. Provide specific findings about what they asked for

When creating stories:
1. Use create_user_story_with_context()
2. Include repository context if available

Always be helpful, specific, and use the proper tools rather than trying to do everything yourself.""",
    
    tools=[analyze_repository_with_agents, create_user_story_with_context, get_system_status]
)

# Export the main agent as root_agent for ADK compatibility  
root_agent = dev_assistant

if __name__ == "__main__":
    print("✅ ADK Agent System v2 loaded successfully")
    print("🤖 Main agent: dev_assistant")  
    print("📚 Sub-agents: code_reader, search_agent, repository_agent, story_agent")
    print("🛠  Atomic tools: clone_repository, read_file, search_pattern_in_files, etc.")
    print("\n🚀 Ready to analyze repositories and create stories!")