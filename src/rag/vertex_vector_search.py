"""Vertex AI Vector Search - Alternative to RAG Engine using Vector Search."""

import os
import hashlib
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / 'hartford_agent' / '.env'
if env_path.exists():
    load_dotenv(env_path)

try:
    from google.cloud import storage
    from google.cloud import aiplatform
    from vertexai.language_models import TextEmbeddingModel
    import vertexai
    
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "insurance-claims-poc")
    LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    VECTOR_SEARCH_AVAILABLE = True
    
except ImportError as e:
    VECTOR_SEARCH_AVAILABLE = False
    print(f"Warning: Vertex AI Vector Search not available: {e}")


class VertexVectorSearch:
    """
    Use Vertex AI's Vector Search (Matching Engine) for code similarity search.
    This is a production-ready alternative to RAG Engine.
    """
    
    def __init__(self):
        """Initialize Vector Search manager."""
        self.project_id = PROJECT_ID
        self.location = LOCATION
        self.embedding_model = None
        self.indices = {}  # repo_url -> index_endpoint
        self.embeddings_cache = {}
        
        if VECTOR_SEARCH_AVAILABLE:
            # Initialize embedding model
            self.embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
            
            # Initialize AI Platform client
            self.client = aiplatform.gapic.IndexServiceClient(
                client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
            )
            
            # Storage client for saving embeddings
            self.storage_client = storage.Client()
            self.bucket_name = f"{self.project_id}-code-embeddings"
            self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Ensure GCS bucket exists for embeddings."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            if not bucket.exists():
                bucket = self.storage_client.create_bucket(
                    self.bucket_name,
                    location=self.location
                )
                print(f"Created bucket: {self.bucket_name}")
        except Exception as e:
            print(f"Bucket check: {e}")
    
    def _generate_index_id(self, repo_url: str) -> str:
        """Generate unique index ID from repository URL."""
        url_hash = hashlib.md5(repo_url.encode()).hexdigest()[:8]
        repo_name = repo_url.split('/')[-1].replace('.git', '').replace('-', '_')
        return f"idx_{repo_name}_{url_hash}"
    
    async def create_embeddings_for_repo(self, repo_url: str, repo_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create embeddings for all code files in repository.
        
        Args:
            repo_url: Repository URL
            repo_content: Repository files and content
            
        Returns:
            Status and embedding information
        """
        if not VECTOR_SEARCH_AVAILABLE:
            return {
                "status": "unavailable",
                "message": "Vector Search not configured"
            }
        
        index_id = self._generate_index_id(repo_url)
        embeddings_data = []
        
        print(f"Creating embeddings for {len(repo_content.get('files', {}))} files...")
        
        # Process each file
        for file_path, content in repo_content.get('files', {}).items():
            if self._should_skip_file(file_path):
                continue
            
            # Prepare text for embedding
            text = self._prepare_text_for_embedding(file_path, content)
            
            # Generate embedding
            try:
                embeddings = self.embedding_model.get_embeddings([text])
                embedding_vector = embeddings[0].values
                
                # Store embedding with metadata
                embeddings_data.append({
                    "id": hashlib.md5(file_path.encode()).hexdigest(),
                    "embedding": embedding_vector,
                    "metadata": {
                        "file_path": file_path,
                        "language": self._detect_language(file_path),
                        "content_preview": content[:500],
                        "full_content": content
                    }
                })
                
            except Exception as e:
                print(f"Error embedding {file_path}: {e}")
        
        # Save embeddings to GCS
        gcs_path = f"{index_id}/embeddings.json"
        self._save_embeddings_to_gcs(gcs_path, embeddings_data)
        
        # Cache embeddings
        self.embeddings_cache[index_id] = embeddings_data
        
        return {
            "status": "success",
            "index_id": index_id,
            "embeddings_count": len(embeddings_data),
            "gcs_path": f"gs://{self.bucket_name}/{gcs_path}"
        }
    
    def _prepare_text_for_embedding(self, file_path: str, content: str) -> str:
        """Prepare code content for embedding."""
        # Add context about file
        language = self._detect_language(file_path)
        
        # Truncate long content
        max_chars = 8000  # Gecko model limit
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        
        # Format for embedding
        text = f"""
File: {file_path}
Language: {language}
---
{content}
"""
        return text
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file path."""
        ext_to_lang = {
            '.py': 'Python',
            '.js': 'JavaScript', '.jsx': 'React',
            '.ts': 'TypeScript', '.tsx': 'TypeScript React',
            '.java': 'Java',
            '.go': 'Go',
            '.sql': 'SQL',
            '.tf': 'Terraform',
            '.yaml': 'YAML', '.yml': 'YAML',
            '.json': 'JSON',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.cs': 'C#',
            '.cpp': 'C++', '.cc': 'C++',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.r': 'R',
            '.m': 'MATLAB',
            '.sh': 'Shell', '.bash': 'Bash',
            '.dockerfile': 'Docker', 'Dockerfile': 'Docker'
        }
        
        for ext, lang in ext_to_lang.items():
            if file_path.lower().endswith(ext.lower()) or file_path.endswith(ext):
                return lang
        
        return 'Unknown'
    
    def _should_skip_file(self, file_path: str) -> bool:
        """Check if file should be skipped."""
        skip_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.pdf', '.exe', '.dll', '.so']
        skip_dirs = ['node_modules', '.git', '__pycache__', 'venv', '.venv', 'dist', 'build', 'target']
        
        for skip_dir in skip_dirs:
            if skip_dir in file_path:
                return True
        
        for ext in skip_extensions:
            if file_path.endswith(ext):
                return True
        
        return False
    
    def _save_embeddings_to_gcs(self, gcs_path: str, embeddings_data: List[Dict]):
        """Save embeddings to GCS."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(gcs_path)
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = []
            for item in embeddings_data:
                serializable_item = item.copy()
                serializable_item['embedding'] = item['embedding'].tolist() if isinstance(item['embedding'], np.ndarray) else item['embedding']
                serializable_data.append(serializable_item)
            
            blob.upload_from_string(json.dumps(serializable_data))
            print(f"Saved {len(embeddings_data)} embeddings to GCS")
        except Exception as e:
            print(f"Error saving embeddings: {e}")
    
    async def search_similar_code(self, repo_url: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search for similar code using vector similarity.
        
        Args:
            repo_url: Repository URL
            query: Search query
            top_k: Number of results
            
        Returns:
            Similar code snippets
        """
        if not VECTOR_SEARCH_AVAILABLE:
            return {
                "status": "unavailable",
                "results": []
            }
        
        index_id = self._generate_index_id(repo_url)
        
        # Get embeddings for this repo
        if index_id not in self.embeddings_cache:
            # Try to load from GCS
            gcs_path = f"{index_id}/embeddings.json"
            try:
                bucket = self.storage_client.bucket(self.bucket_name)
                blob = bucket.blob(gcs_path)
                if blob.exists():
                    embeddings_data = json.loads(blob.download_as_text())
                    self.embeddings_cache[index_id] = embeddings_data
                else:
                    return {
                        "status": "not_found",
                        "message": f"No embeddings found for {repo_url}"
                    }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error loading embeddings: {e}"
                }
        
        # Generate embedding for query
        try:
            query_embeddings = self.embedding_model.get_embeddings([query])
            query_vector = np.array(query_embeddings[0].values)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error generating query embedding: {e}"
            }
        
        # Calculate similarities
        similarities = []
        for item in self.embeddings_cache[index_id]:
            embedding = np.array(item['embedding'])
            
            # Cosine similarity
            similarity = np.dot(query_vector, embedding) / (np.linalg.norm(query_vector) * np.linalg.norm(embedding))
            
            similarities.append({
                "score": float(similarity),
                "file_path": item['metadata']['file_path'],
                "content_preview": item['metadata']['content_preview'],
                "language": item['metadata']['language']
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            "status": "success",
            "query": query,
            "results": similarities[:top_k]
        }


class CodeSearchRAG:
    """
    Simple session-based code search.
    Create index when needed, use during session, let it be destroyed.
    """
    
    def __init__(self, session_id: str = None):
        """Initialize code search RAG for this session."""
        self.vector_search = VertexVectorSearch()
        self.session_id = session_id or self._generate_session_id()
        self.indexed_repos = {}  # Track what we've indexed this session
    
    def _generate_session_id(self) -> str:
        """Generate session ID."""
        return f"session_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]}"
    
    async def index_repository(self, repo_url: str, repo_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Index a repository for search.
        
        Args:
            repo_url: Repository URL
            repo_content: Repository files and content
            
        Returns:
            Indexing status
        """
        # Create embeddings
        result = await self.vector_search.create_embeddings_for_repo(repo_url, repo_content)
        
        if result['status'] == 'success':
            self.corpus_registry[repo_url] = {
                'index_id': result['index_id'],
                'indexed_at': datetime.now(),
                'file_count': result['embeddings_count']
            }
        
        return result
    
    async def search(self, repo_url: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for code in repository.
        
        Args:
            repo_url: Repository URL
            query: Search query
            top_k: Number of results
            
        Returns:
            Search results
        """
        result = await self.vector_search.search_similar_code(repo_url, query, top_k)
        
        if result['status'] == 'success':
            return result['results']
        else:
            return []
    
    async def explain_code(self, repo_url: str, query: str) -> str:
        """
        Explain code based on search results.
        
        Args:
            repo_url: Repository URL
            query: Question about code
            
        Returns:
            Explanation with code references
        """
        # Search for relevant code
        results = await self.search(repo_url, query, top_k=3)
        
        if not results:
            return f"No relevant code found for: {query}"
        
        # Format explanation
        explanation = f"Based on the codebase:\n\n"
        
        for i, result in enumerate(results, 1):
            file_path = result['file_path']
            language = result['language']
            content = result['content_preview']
            score = result['score']
            
            explanation += f"**{i}. `{file_path}` ({language}, relevance: {score:.2f})**\n"
            explanation += f"```{language.lower()}\n{content}\n```\n\n"
        
        return explanation