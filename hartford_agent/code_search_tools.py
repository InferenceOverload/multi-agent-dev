"""Code Search Tools - Tools for searching and understanding code."""

from typing import Optional
from src.rag.vertex_rag_manager import VertexRAGManager, CodeRAGBot

# Global RAG manager instance
_rag_manager = None
_rag_bot = None


def _get_rag_bot():
    """Get or create RAG bot instance."""
    global _rag_manager, _rag_bot
    if _rag_manager is None:
        _rag_manager = VertexRAGManager()
        _rag_bot = CodeRAGBot(_rag_manager)
    return _rag_bot


async def search_code(repo_url: str, query: str, max_results: int = 5) -> str:
    """
    Search for specific code in a repository using RAG.
    
    Args:
        repo_url: Repository URL to search
        query: What to search for (e.g., "authentication logic", "button components", "SQL queries")
        max_results: Maximum number of results to return
        
    Returns:
        Found code snippets with file locations
    """
    rag_bot = _get_rag_bot()
    
    # Search for code
    results = await rag_bot.find_code(repo_url, query)
    
    if results.get('status') != 'success':
        return f"❌ Search failed: {results.get('message', 'Unknown error')}"
    
    if not results.get('results'):
        return f"No results found for: {query}"
    
    # Format results
    response = f"🔍 **Code Search Results for:** {query}\n\n"
    
    for i, result in enumerate(results['results'][:max_results], 1):
        file_path = result.get('file_path', 'Unknown file')
        content = result['content'][:300]  # Show first 300 chars
        score = result.get('relevance_score', 0)
        
        response += f"**{i}. `{file_path}` (relevance: {score:.2f})**\n"
        response += f"```\n{content}...\n```\n\n"
    
    response += f"Found {len(results['results'])} total matches."
    
    return response


async def find_component(repo_url: str, component_name: str) -> str:
    """
    Find a specific UI component in the repository.
    
    Args:
        repo_url: Repository URL
        component_name: Name of component to find (e.g., "button", "navbar", "login form")
        
    Returns:
        Component location and implementation details
    """
    query = f"React component Vue component Angular component {component_name}"
    return await search_code(repo_url, query, max_results=3)


async def find_api_endpoint(repo_url: str, endpoint: str) -> str:
    """
    Find API endpoint implementation.
    
    Args:
        repo_url: Repository URL
        endpoint: Endpoint to find (e.g., "/users", "authentication", "POST /api/login")
        
    Returns:
        API endpoint implementation and location
    """
    query = f"API endpoint route handler {endpoint} app.get app.post router"
    return await search_code(repo_url, query, max_results=3)


async def find_database_query(repo_url: str, table_or_operation: str) -> str:
    """
    Find database queries or operations.
    
    Args:
        repo_url: Repository URL
        table_or_operation: Table name or operation (e.g., "users table", "INSERT", "JOIN")
        
    Returns:
        SQL queries and database operations
    """
    query = f"SQL query database {table_or_operation} SELECT INSERT UPDATE DELETE FROM WHERE"
    return await search_code(repo_url, query, max_results=3)


async def find_configuration(repo_url: str, config_type: str) -> str:
    """
    Find configuration files or settings.
    
    Args:
        repo_url: Repository URL
        config_type: Type of configuration (e.g., "database", "API keys", "environment")
        
    Returns:
        Configuration files and settings
    """
    query = f"configuration config settings {config_type} .env environment variables"
    return await search_code(repo_url, query, max_results=3)


async def explain_code_structure(repo_url: str, module_or_feature: str) -> str:
    """
    Explain how a module or feature is structured.
    
    Args:
        repo_url: Repository URL
        module_or_feature: Module or feature to explain (e.g., "authentication", "payment processing")
        
    Returns:
        Explanation of code structure and architecture
    """
    rag_bot = _get_rag_bot()
    
    # Get comprehensive understanding
    question = f"How is {module_or_feature} implemented and structured in this codebase?"
    answer = await rag_bot.answer_question(repo_url, question)
    
    return f"📖 **Code Structure Explanation: {module_or_feature}**\n\n{answer}"