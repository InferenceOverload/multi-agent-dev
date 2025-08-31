"""Vertex AI RAG Engine Manager - Enterprise-scale RAG for code understanding."""

import os
import hashlib
import tempfile
import shutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import json
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / 'hartford_agent' / '.env'
if env_path.exists():
    load_dotenv(env_path)

# Google Cloud imports
try:
    from google.cloud import storage
    from google.cloud import aiplatform as vertex_ai
    import vertexai
    
    # Initialize Vertex AI with project settings
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "insurance-claims-poc")
    LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    # Try to import preview features for RAG
    try:
        from vertexai.preview import rag
        from vertexai.language_models import TextEmbeddingModel
        RAG_AVAILABLE = True
    except ImportError:
        RAG_AVAILABLE = False
        print("Warning: Vertex AI RAG preview features not available. Using fallback.")
    
    VERTEX_AI_AVAILABLE = True
    
except ImportError as e:
    VERTEX_AI_AVAILABLE = False
    RAG_AVAILABLE = False
    print(f"Warning: Vertex AI not fully configured: {e}")


class VertexRAGManager:
    """
    Manages Vertex AI RAG corpora for code repositories.
    Creates on-demand RAG corpora and handles lifecycle.
    """
    
    def __init__(self, project_id: str = None, location: str = "us-central1"):
        """
        Initialize Vertex AI RAG Manager.
        
        Args:
            project_id: GCP project ID
            location: GCP location for Vertex AI
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.active_corpora = {}  # repo_url -> corpus_info
        self.corpus_cache = {}  # Cache corpus metadata
        
        if VERTEX_AI_AVAILABLE and self.project_id:
            vertexai.init(project=self.project_id, location=self.location)
            self.storage_client = storage.Client()
            self.bucket_name = f"{self.project_id}-rag-corpora"
            self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Ensure GCS bucket exists for storing documents."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            if not bucket.exists():
                bucket = self.storage_client.create_bucket(
                    self.bucket_name,
                    location=self.location
                )
                print(f"Created bucket: {self.bucket_name}")
        except Exception as e:
            print(f"Warning: Could not create bucket: {e}")
    
    def _generate_corpus_id(self, repo_url: str) -> str:
        """Generate unique corpus ID from repository URL."""
        # Create hash of repo URL for unique ID
        url_hash = hashlib.md5(repo_url.encode()).hexdigest()[:8]
        # Clean repo name from URL
        repo_name = repo_url.split('/')[-1].replace('.git', '').replace('-', '_')
        return f"corpus_{repo_name}_{url_hash}"
    
    async def get_or_create_corpus(self, repo_url: str, repo_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get existing corpus or create new one for repository.
        
        Args:
            repo_url: Repository URL
            repo_content: Repository content and metadata
            
        Returns:
            Corpus information including ID and status
        """
        if not VERTEX_AI_AVAILABLE:
            return {
                "status": "unavailable",
                "message": "Vertex AI not configured. Using fallback search.",
                "corpus_id": None
            }
        
        corpus_id = self._generate_corpus_id(repo_url)
        
        # Check if corpus already exists
        if corpus_id in self.active_corpora:
            # Check if corpus is still fresh (less than 24 hours old)
            corpus_info = self.active_corpora[corpus_id]
            age = datetime.now() - corpus_info['created_at']
            if age < timedelta(hours=24):
                return {
                    "status": "existing",
                    "corpus_id": corpus_id,
                    "corpus_name": corpus_info['name'],
                    "created_at": corpus_info['created_at'],
                    "document_count": corpus_info.get('document_count', 0)
                }
        
        # Create new corpus
        try:
            corpus_info = await self._create_corpus(corpus_id, repo_url, repo_content)
            self.active_corpora[corpus_id] = corpus_info
            return corpus_info
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to create corpus: {str(e)}",
                "corpus_id": None
            }
    
    async def _create_corpus(self, corpus_id: str, repo_url: str, repo_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new RAG corpus for the repository.
        
        Args:
            corpus_id: Unique corpus identifier
            repo_url: Repository URL
            repo_content: Repository files and content
            
        Returns:
            Corpus creation status and information
        """
        print(f"Creating RAG corpus for {repo_url}...")
        
        # Configure embedding model
        embedding_model_config = rag.EmbeddingModelConfig(
            publisher_model="publishers/google/models/text-embedding-004"
        )
        
        # Create corpus
        corpus = rag.create_corpus(
            display_name=f"Code Corpus: {repo_url}",
            description=f"RAG corpus for code repository: {repo_url}",
            embedding_model_config=embedding_model_config,
            corpus_id=corpus_id
        )
        
        # Upload documents to GCS
        uploaded_files = await self._upload_code_files(corpus_id, repo_content)
        
        # Import documents into corpus
        import_response = await self._import_documents(corpus, uploaded_files)
        
        return {
            "status": "created",
            "corpus_id": corpus_id,
            "corpus_name": corpus.name,
            "created_at": datetime.now(),
            "document_count": len(uploaded_files),
            "import_status": import_response
        }
    
    async def _upload_code_files(self, corpus_id: str, repo_content: Dict[str, Any]) -> List[str]:
        """
        Upload code files to GCS for RAG ingestion.
        
        Args:
            corpus_id: Corpus identifier
            repo_content: Repository files to upload
            
        Returns:
            List of GCS URIs for uploaded files
        """
        uploaded_uris = []
        bucket = self.storage_client.bucket(self.bucket_name)
        
        # Process each file in the repository
        for file_path, content in repo_content.get('files', {}).items():
            # Skip non-code files
            if self._should_skip_file(file_path):
                continue
            
            # Create GCS path
            gcs_path = f"{corpus_id}/{file_path}"
            blob = bucket.blob(gcs_path)
            
            # Prepare content with metadata
            enhanced_content = self._enhance_code_content(file_path, content)
            
            # Upload to GCS
            blob.upload_from_string(enhanced_content)
            uploaded_uris.append(f"gs://{self.bucket_name}/{gcs_path}")
        
        return uploaded_uris
    
    def _should_skip_file(self, file_path: str) -> bool:
        """Check if file should be skipped for RAG indexing."""
        skip_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.pdf']
        skip_dirs = ['node_modules', '.git', '__pycache__', 'venv', '.venv', 'dist', 'build']
        
        # Check directory patterns
        for skip_dir in skip_dirs:
            if skip_dir in file_path:
                return True
        
        # Check file extensions
        for ext in skip_extensions:
            if file_path.endswith(ext):
                return True
        
        return False
    
    def _enhance_code_content(self, file_path: str, content: str) -> str:
        """
        Enhance code content with metadata for better RAG retrieval.
        
        Args:
            file_path: Path to the file
            content: File content
            
        Returns:
            Enhanced content with metadata
        """
        # Detect language from file extension
        language = self._detect_language(file_path)
        
        # Add metadata header
        enhanced = f"""# File: {file_path}
# Language: {language}
# Purpose: Code file in repository
---

{content}
"""
        return enhanced
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file path."""
        ext_to_lang = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.jsx': 'React',
            '.ts': 'TypeScript',
            '.tsx': 'TypeScript React',
            '.java': 'Java',
            '.go': 'Go',
            '.rs': 'Rust',
            '.sql': 'SQL',
            '.tf': 'Terraform',
            '.yaml': 'YAML',
            '.json': 'JSON',
            '.md': 'Markdown'
        }
        
        for ext, lang in ext_to_lang.items():
            if file_path.endswith(ext):
                return lang
        
        return 'Unknown'
    
    async def _import_documents(self, corpus: Any, file_uris: List[str]) -> Dict[str, Any]:
        """
        Import documents from GCS into RAG corpus.
        
        Args:
            corpus: Vertex AI RAG corpus object
            file_uris: List of GCS URIs to import
            
        Returns:
            Import status information
        """
        if not file_uris:
            return {"status": "no_files", "imported": 0}
        
        # Batch import for efficiency
        batch_size = 100
        total_imported = 0
        
        for i in range(0, len(file_uris), batch_size):
            batch = file_uris[i:i + batch_size]
            
            import_request = rag.ImportRagFilesRequest(
                corpus=corpus.name,
                gcs_source=rag.GcsSource(uris=batch),
                chunk_size=512,  # Optimal chunk size for code
                chunk_overlap=100  # Overlap for context
            )
            
            response = rag.import_rag_files(import_request)
            total_imported += len(batch)
        
        return {
            "status": "imported",
            "imported": total_imported,
            "chunk_size": 512,
            "chunk_overlap": 100
        }
    
    async def query_corpus(self, corpus_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query the RAG corpus for relevant code.
        
        Args:
            corpus_id: Corpus to query
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            Query results with relevant code snippets
        """
        if not VERTEX_AI_AVAILABLE:
            return {
                "status": "unavailable",
                "results": [],
                "message": "Vertex AI not configured"
            }
        
        if corpus_id not in self.active_corpora:
            return {
                "status": "not_found",
                "results": [],
                "message": f"Corpus {corpus_id} not found"
            }
        
        try:
            # Perform RAG query
            response = rag.query(
                corpus_name=self.active_corpora[corpus_id]['corpus_name'],
                query_text=query,
                similarity_top_k=top_k,
                vector_distance_threshold=0.5  # Relevance threshold
            )
            
            # Format results
            results = []
            for chunk in response.contexts:
                results.append({
                    "content": chunk.text,
                    "file_path": self._extract_file_path(chunk.text),
                    "relevance_score": chunk.score if hasattr(chunk, 'score') else 1.0,
                    "chunk_id": chunk.id if hasattr(chunk, 'id') else None
                })
            
            return {
                "status": "success",
                "query": query,
                "results": results,
                "result_count": len(results)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "results": [],
                "message": f"Query failed: {str(e)}"
            }
    
    def _extract_file_path(self, content: str) -> Optional[str]:
        """Extract file path from enhanced content."""
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# File:'):
                return line.replace('# File:', '').strip()
        return None
    
    async def cleanup_old_corpora(self, max_age_hours: int = 24):
        """
        Clean up old corpora to save resources.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
        """
        current_time = datetime.now()
        to_remove = []
        
        for corpus_id, info in self.active_corpora.items():
            age = current_time - info['created_at']
            if age > timedelta(hours=max_age_hours):
                to_remove.append(corpus_id)
        
        for corpus_id in to_remove:
            try:
                # Delete from Vertex AI
                rag.delete_corpus(name=self.active_corpora[corpus_id]['corpus_name'])
                
                # Delete from GCS
                bucket = self.storage_client.bucket(self.bucket_name)
                blobs = bucket.list_blobs(prefix=f"{corpus_id}/")
                for blob in blobs:
                    blob.delete()
                
                # Remove from active corpora
                del self.active_corpora[corpus_id]
                
                print(f"Cleaned up corpus: {corpus_id}")
                
            except Exception as e:
                print(f"Failed to cleanup corpus {corpus_id}: {e}")


class CodeRAGBot:
    """
    On-demand RAG bot for code understanding.
    Integrates with Vertex AI RAG Manager.
    """
    
    def __init__(self, rag_manager: VertexRAGManager):
        """
        Initialize Code RAG Bot.
        
        Args:
            rag_manager: Vertex AI RAG Manager instance
        """
        self.rag_manager = rag_manager
        self.query_cache = {}  # Cache recent queries
    
    async def analyze_repository(self, repo_url: str, repo_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or get RAG corpus for repository.
        
        Args:
            repo_url: Repository URL
            repo_content: Repository files and metadata
            
        Returns:
            Analysis status and corpus information
        """
        corpus_info = await self.rag_manager.get_or_create_corpus(repo_url, repo_content)
        return corpus_info
    
    async def find_code(self, repo_url: str, query: str) -> Dict[str, Any]:
        """
        Find specific code in the repository.
        
        Args:
            repo_url: Repository URL
            query: What to search for
            
        Returns:
            Found code snippets with locations
        """
        corpus_id = self.rag_manager._generate_corpus_id(repo_url)
        
        # Check cache
        cache_key = f"{corpus_id}:{query}"
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            cached_result['from_cache'] = True
            return cached_result
        
        # Query corpus
        result = await self.rag_manager.query_corpus(corpus_id, query)
        
        # Cache result
        self.query_cache[cache_key] = result
        
        return result
    
    async def answer_question(self, repo_url: str, question: str) -> str:
        """
        Answer a question about the codebase.
        
        Args:
            repo_url: Repository URL
            question: Question to answer
            
        Returns:
            Natural language answer with code references
        """
        # Find relevant code
        code_results = await self.find_code(repo_url, question)
        
        if code_results['status'] != 'success' or not code_results['results']:
            return f"I couldn't find relevant code for: {question}"
        
        # Format answer with code references
        answer = f"Based on the codebase analysis:\n\n"
        
        for i, result in enumerate(code_results['results'][:3], 1):
            file_path = result.get('file_path', 'Unknown file')
            content = result['content'][:500]  # Truncate for readability
            
            answer += f"**{i}. In `{file_path}`:**\n"
            answer += f"```\n{content}\n```\n\n"
        
        answer += f"\nFound {len(code_results['results'])} relevant code sections."
        
        return answer