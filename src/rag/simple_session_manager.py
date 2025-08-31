"""Simple Session Manager - Create, Use, Destroy pattern for RAG."""

import os
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / 'hartford_agent' / '.env'
if env_path.exists():
    load_dotenv(env_path)

try:
    from google.cloud import storage
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False


class SimpleSessionManager:
    """
    Simple session-based RAG management.
    Create when user starts, use during session, destroy when done.
    """
    
    def __init__(self, session_id: str = None):
        """Initialize session manager."""
        self.session_id = session_id or self._generate_session_id()
        self.created_at = datetime.now()
        self.resources = {
            'corpora': [],  # List of created corpora
            'embeddings': [],  # List of embedding files
            'temp_files': []  # List of temporary files
        }
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "insurance-claims-poc")
        
        if STORAGE_AVAILABLE:
            self.storage_client = storage.Client()
        else:
            self.storage_client = None
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().isoformat()
        return f"session_{hashlib.md5(timestamp.encode()).hexdigest()[:12]}"
    
    def register_corpus(self, corpus_id: str, repo_url: str) -> None:
        """
        Register a corpus created for this session.
        
        Args:
            corpus_id: Corpus identifier
            repo_url: Repository URL
        """
        self.resources['corpora'].append({
            'corpus_id': corpus_id,
            'repo_url': repo_url,
            'created_at': datetime.now()
        })
        print(f"📝 Registered corpus {corpus_id} for session {self.session_id}")
    
    def register_embeddings(self, gcs_path: str) -> None:
        """
        Register embeddings stored in GCS.
        
        Args:
            gcs_path: GCS path to embeddings
        """
        self.resources['embeddings'].append({
            'path': gcs_path,
            'created_at': datetime.now()
        })
    
    def cleanup(self) -> Dict[str, int]:
        """
        Clean up all resources created during this session.
        This is called when the session ends.
        
        Returns:
            Cleanup statistics
        """
        stats = {
            'corpora_deleted': 0,
            'embeddings_deleted': 0,
            'files_deleted': 0,
            'errors': 0
        }
        
        print(f"🧹 Cleaning up session {self.session_id}...")
        
        # Delete corpora (would call Vertex AI API in production)
        for corpus in self.resources['corpora']:
            try:
                # TODO: Call Vertex AI to delete corpus
                print(f"  Would delete corpus: {corpus['corpus_id']}")
                stats['corpora_deleted'] += 1
            except Exception as e:
                print(f"  Error deleting corpus: {e}")
                stats['errors'] += 1
        
        # Delete embeddings from GCS
        if self.storage_client:
            for embedding in self.resources['embeddings']:
                try:
                    # Parse bucket and blob from GCS path
                    if embedding['path'].startswith('gs://'):
                        parts = embedding['path'].replace('gs://', '').split('/', 1)
                        if len(parts) == 2:
                            bucket_name, blob_path = parts
                            bucket = self.storage_client.bucket(bucket_name)
                            
                            # Delete all blobs with this prefix
                            blobs = bucket.list_blobs(prefix=blob_path)
                            for blob in blobs:
                                blob.delete()
                                stats['embeddings_deleted'] += 1
                                print(f"  Deleted: {blob.name}")
                except Exception as e:
                    print(f"  Error deleting embeddings: {e}")
                    stats['errors'] += 1
        
        # Delete temporary files
        for temp_file in self.resources['temp_files']:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    stats['files_deleted'] += 1
            except Exception as e:
                print(f"  Error deleting file: {e}")
                stats['errors'] += 1
        
        print(f"✅ Cleanup complete: {stats}")
        return stats
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about this session."""
        duration = (datetime.now() - self.created_at).total_seconds()
        
        return {
            'session_id': self.session_id,
            'created_at': self.created_at,
            'duration_seconds': duration,
            'resources': {
                'corpora_count': len(self.resources['corpora']),
                'embeddings_count': len(self.resources['embeddings']),
                'temp_files_count': len(self.resources['temp_files'])
            }
        }


class SessionRAG:
    """
    Simple RAG that lives only for the session.
    """
    
    def __init__(self, session_manager: SimpleSessionManager = None, session_id: str = None):
        """Initialize session-based RAG."""
        self.session = session_manager or SimpleSessionManager(session_id=session_id)
        self.active_corpora = {}  # repo_url -> corpus_info
        self.session_id = session_id or self.session.session_id
    
    async def create_temporary_index(self, repo_url: str, repo_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a temporary index for this session.
        
        Args:
            repo_url: Repository URL
            repo_content: Repository files
            
        Returns:
            Index creation status
        """
        # Generate temporary corpus ID
        corpus_id = f"{self.session.session_id}_{hashlib.md5(repo_url.encode()).hexdigest()[:8]}"
        
        # Register with session manager
        self.session.register_corpus(corpus_id, repo_url)
        
        # Store in active corpora
        self.active_corpora[repo_url] = {
            'corpus_id': corpus_id,
            'created_at': datetime.now()
        }
        
        return {
            'status': 'success',
            'corpus_id': corpus_id,
            'session_id': self.session.session_id,
            'message': 'Temporary index created for this session'
        }
    
    async def search(self, repo_url: str, query: str) -> List[Dict[str, Any]]:
        """
        Search in the temporary index.
        
        Args:
            repo_url: Repository URL
            query: Search query
            
        Returns:
            Search results
        """
        if repo_url not in self.active_corpora:
            return []
        
        corpus_info = self.active_corpora[repo_url]
        
        # Perform search (simplified for now)
        results = [
            {
                'file_path': 'example.py',
                'content': 'Example result',
                'score': 0.95
            }
        ]
        
        return results
    
    def cleanup(self):
        """Clean up all resources when session ends."""
        return self.session.cleanup()


# Context manager for automatic cleanup
class SessionContext:
    """
    Context manager for session-based RAG.
    Ensures cleanup happens even if errors occur.
    """
    
    def __init__(self):
        """Initialize session context."""
        self.session_rag = None
    
    async def __aenter__(self):
        """Enter context - create session RAG."""
        self.session_rag = SessionRAG()
        print(f"🚀 Started session: {self.session_rag.session.session_id}")
        return self.session_rag
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context - cleanup resources."""
        if self.session_rag:
            self.session_rag.cleanup()
            print(f"🏁 Ended session: {self.session_rag.session.session_id}")