"""RAG Lifecycle Manager - Handles corpus freshness and cleanup for enterprise scale."""

import os
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / 'hartford_agent' / '.env'
if env_path.exists():
    load_dotenv(env_path)

try:
    from google.cloud import firestore
    from google.cloud import storage
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False
    print("Warning: Firestore not available for corpus tracking")


class CorpusStrategy(Enum):
    """Different strategies for corpus lifecycle."""
    SESSION = "session"  # Destroyed after user session ends
    DAILY = "daily"  # Refreshed daily at specific time
    ON_DEMAND = "on_demand"  # Created when needed, destroyed after idle
    PERSISTENT = "persistent"  # Keep for frequently accessed repos
    DEVELOPMENT = "development"  # Keep during active development


class RAGLifecycleManager:
    """
    Manages the lifecycle of RAG corpora at enterprise scale.
    Handles freshness, cleanup, and cost optimization.
    """
    
    def __init__(self):
        """Initialize lifecycle manager."""
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "insurance-claims-poc")
        self.corpus_registry = {}  # In-memory registry
        
        if FIRESTORE_AVAILABLE:
            # Use Firestore for persistent corpus tracking
            self.db = firestore.Client(project=self.project_id)
            self.corpus_collection = self.db.collection('rag_corpora')
            self.session_collection = self.db.collection('user_sessions')
        else:
            self.db = None
    
    def _generate_corpus_key(self, repo_url: str, branch: str = "main") -> str:
        """Generate unique key for corpus including branch."""
        # Include branch and date in key for daily freshness
        today = datetime.now().strftime("%Y%m%d")
        key_string = f"{repo_url}:{branch}:{today}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    async def should_create_corpus(
        self, 
        repo_url: str, 
        user_session: str,
        query_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Intelligent decision on whether to create/reuse corpus.
        
        Args:
            repo_url: Repository URL
            user_session: User session ID
            query_type: Type of query (affects strategy)
            
        Returns:
            Decision with strategy and reasoning
        """
        # Check if corpus exists and is fresh
        existing = await self._find_existing_corpus(repo_url)
        
        if existing and existing['is_fresh']:
            return {
                "action": "reuse",
                "corpus_id": existing['corpus_id'],
                "reasoning": f"Fresh corpus exists (created {existing['age_hours']}h ago)",
                "strategy": existing['strategy']
            }
        
        # Determine strategy based on usage patterns
        strategy = await self._determine_strategy(repo_url, query_type)
        
        if strategy == CorpusStrategy.SESSION:
            return {
                "action": "create_temporary",
                "reasoning": "Creating session-based corpus for this analysis",
                "strategy": "session",
                "ttl_hours": 2  # Destroy after 2 hours
            }
        
        elif strategy == CorpusStrategy.DAILY:
            return {
                "action": "create_daily",
                "reasoning": "Creating daily corpus (will refresh tomorrow)",
                "strategy": "daily",
                "ttl_hours": 24
            }
        
        elif strategy == CorpusStrategy.ON_DEMAND:
            return {
                "action": "create_on_demand",
                "reasoning": "Creating on-demand corpus for immediate use",
                "strategy": "on_demand",
                "ttl_hours": 4  # Keep for 4 hours after last use
            }
        
        elif strategy == CorpusStrategy.DEVELOPMENT:
            # Check if user is in active development session
            if await self._is_development_session(user_session, repo_url):
                return {
                    "action": "create_development",
                    "reasoning": "Active development detected - keeping corpus warm",
                    "strategy": "development",
                    "ttl_hours": 8  # Keep for full workday
                }
        
        # Default
        return {
            "action": "create_temporary",
            "reasoning": "Creating temporary corpus",
            "strategy": "session",
            "ttl_hours": 1
        }
    
    async def _find_existing_corpus(self, repo_url: str) -> Optional[Dict[str, Any]]:
        """Find existing corpus and check freshness."""
        if not self.db:
            # Check in-memory registry
            key = self._generate_corpus_key(repo_url)
            if key in self.corpus_registry:
                corpus = self.corpus_registry[key]
                age = datetime.now() - corpus['created_at']
                return {
                    'corpus_id': corpus['corpus_id'],
                    'is_fresh': age < timedelta(hours=corpus.get('ttl_hours', 24)),
                    'age_hours': age.total_seconds() / 3600,
                    'strategy': corpus.get('strategy', 'unknown')
                }
            return None
        
        # Query Firestore
        try:
            # Look for corpus created today
            today_start = datetime.now().replace(hour=0, minute=0, second=0)
            
            query = self.corpus_collection.where('repo_url', '==', repo_url)\
                                        .where('created_at', '>=', today_start)\
                                        .where('status', '==', 'active')\
                                        .limit(1)
            
            docs = query.get()
            for doc in docs:
                data = doc.to_dict()
                age = datetime.now() - data['created_at']
                
                return {
                    'corpus_id': data['corpus_id'],
                    'is_fresh': age < timedelta(hours=data.get('ttl_hours', 24)),
                    'age_hours': age.total_seconds() / 3600,
                    'strategy': data.get('strategy', 'unknown')
                }
        except Exception as e:
            print(f"Error finding corpus: {e}")
        
        return None
    
    async def _determine_strategy(self, repo_url: str, query_type: str) -> CorpusStrategy:
        """Determine best strategy based on usage patterns."""
        # Check historical usage patterns
        if self.db:
            try:
                # Count recent accesses
                week_ago = datetime.now() - timedelta(days=7)
                access_query = self.corpus_collection.where('repo_url', '==', repo_url)\
                                                    .where('accessed_at', '>=', week_ago)
                
                access_count = len(list(access_query.get()))
                
                # Frequently accessed repos get daily refresh
                if access_count > 10:
                    return CorpusStrategy.DAILY
                
                # Moderately accessed get on-demand
                if access_count > 3:
                    return CorpusStrategy.ON_DEMAND
                
            except Exception as e:
                print(f"Error checking usage: {e}")
        
        # Query type affects strategy
        if query_type in ["modification", "development", "implementation"]:
            return CorpusStrategy.DEVELOPMENT
        
        # Default to session-based
        return CorpusStrategy.SESSION
    
    async def _is_development_session(self, user_session: str, repo_url: str) -> bool:
        """Check if user is in active development session."""
        if not self.db:
            return False
        
        try:
            # Check recent activity
            hour_ago = datetime.now() - timedelta(hours=1)
            session_query = self.session_collection.where('session_id', '==', user_session)\
                                                  .where('repo_url', '==', repo_url)\
                                                  .where('last_activity', '>=', hour_ago)
            
            docs = list(session_query.get())
            
            # Multiple interactions indicate development
            return len(docs) > 2
            
        except Exception as e:
            print(f"Error checking session: {e}")
        
        return False
    
    async def register_corpus(
        self,
        corpus_id: str,
        repo_url: str,
        strategy: str,
        ttl_hours: int,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Register a newly created corpus."""
        corpus_data = {
            'corpus_id': corpus_id,
            'repo_url': repo_url,
            'strategy': strategy,
            'ttl_hours': ttl_hours,
            'created_at': datetime.now(),
            'accessed_at': datetime.now(),
            'status': 'active',
            'metadata': metadata or {}
        }
        
        # Store in memory
        key = self._generate_corpus_key(repo_url)
        self.corpus_registry[key] = corpus_data
        
        # Store in Firestore
        if self.db:
            try:
                self.corpus_collection.document(corpus_id).set(corpus_data)
            except Exception as e:
                print(f"Error registering corpus: {e}")
    
    async def mark_corpus_accessed(self, corpus_id: str) -> None:
        """Mark corpus as recently accessed to extend TTL."""
        # Update in memory
        for key, corpus in self.corpus_registry.items():
            if corpus['corpus_id'] == corpus_id:
                corpus['accessed_at'] = datetime.now()
                break
        
        # Update in Firestore
        if self.db:
            try:
                self.corpus_collection.document(corpus_id).update({
                    'accessed_at': datetime.now()
                })
            except Exception as e:
                print(f"Error updating corpus access: {e}")
    
    async def cleanup_stale_corpora(self) -> Dict[str, int]:
        """Clean up stale corpora based on TTL and strategy."""
        cleanup_stats = {
            'checked': 0,
            'deleted': 0,
            'kept': 0
        }
        
        if not self.db:
            # Clean in-memory registry
            to_delete = []
            for key, corpus in self.corpus_registry.items():
                cleanup_stats['checked'] += 1
                
                # Check if expired
                age = datetime.now() - corpus['created_at']
                idle = datetime.now() - corpus['accessed_at']
                
                should_delete = False
                
                if corpus['strategy'] == 'session':
                    # Delete if older than TTL
                    should_delete = age > timedelta(hours=corpus['ttl_hours'])
                
                elif corpus['strategy'] == 'on_demand':
                    # Delete if idle longer than TTL
                    should_delete = idle > timedelta(hours=corpus['ttl_hours'])
                
                elif corpus['strategy'] == 'daily':
                    # Delete if older than 24 hours
                    should_delete = age > timedelta(hours=24)
                
                elif corpus['strategy'] == 'development':
                    # Delete if idle more than 8 hours
                    should_delete = idle > timedelta(hours=8)
                
                if should_delete:
                    to_delete.append(key)
                    cleanup_stats['deleted'] += 1
                else:
                    cleanup_stats['kept'] += 1
            
            # Remove from registry
            for key in to_delete:
                del self.corpus_registry[key]
            
            return cleanup_stats
        
        # Clean in Firestore
        try:
            # Get all active corpora
            active_query = self.corpus_collection.where('status', '==', 'active')
            
            for doc in active_query.get():
                cleanup_stats['checked'] += 1
                data = doc.to_dict()
                
                age = datetime.now() - data['created_at']
                idle = datetime.now() - data['accessed_at']
                
                should_delete = False
                
                # Apply strategy-specific rules
                if data['strategy'] == 'session':
                    should_delete = age > timedelta(hours=data['ttl_hours'])
                elif data['strategy'] == 'on_demand':
                    should_delete = idle > timedelta(hours=data['ttl_hours'])
                elif data['strategy'] == 'daily':
                    should_delete = age > timedelta(hours=24)
                elif data['strategy'] == 'development':
                    should_delete = idle > timedelta(hours=8)
                
                if should_delete:
                    # Mark as deleted (don't actually delete, for audit)
                    doc.reference.update({
                        'status': 'deleted',
                        'deleted_at': datetime.now()
                    })
                    cleanup_stats['deleted'] += 1
                    
                    # TODO: Actually delete from Vertex AI
                    await self._delete_vertex_corpus(data['corpus_id'])
                else:
                    cleanup_stats['kept'] += 1
                    
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        return cleanup_stats
    
    async def _delete_vertex_corpus(self, corpus_id: str) -> None:
        """Delete corpus from Vertex AI."""
        # This would call Vertex AI API to delete the corpus
        print(f"Would delete corpus: {corpus_id}")
        # TODO: Implement actual Vertex AI deletion
    
    async def preserve_context_for_development(
        self,
        user_session: str,
        repo_url: str,
        corpus_id: str,
        context: Dict[str, Any]
    ) -> None:
        """
        Preserve context when user is actively developing.
        This allows continuation even if corpus is refreshed.
        """
        if self.db:
            try:
                # Store development context
                context_doc = {
                    'session_id': user_session,
                    'repo_url': repo_url,
                    'corpus_id': corpus_id,
                    'context': context,
                    'created_at': datetime.now(),
                    'expires_at': datetime.now() + timedelta(hours=8)
                }
                
                self.session_collection.document(f"{user_session}_{corpus_id}").set(context_doc)
                
            except Exception as e:
                print(f"Error preserving context: {e}")
    
    async def get_preserved_context(
        self,
        user_session: str,
        repo_url: str
    ) -> Optional[Dict[str, Any]]:
        """Get preserved context for continuing development."""
        if not self.db:
            return None
        
        try:
            # Look for recent context
            hour_ago = datetime.now() - timedelta(hours=8)
            context_query = self.session_collection.where('session_id', '==', user_session)\
                                                  .where('repo_url', '==', repo_url)\
                                                  .where('expires_at', '>', datetime.now())\
                                                  .order_by('created_at', direction=firestore.Query.DESCENDING)\
                                                  .limit(1)
            
            docs = context_query.get()
            for doc in docs:
                return doc.to_dict()['context']
                
        except Exception as e:
            print(f"Error getting context: {e}")
        
        return None


class SmartCorpusDecision:
    """
    Makes intelligent decisions about corpus creation and reuse.
    """
    
    def __init__(self, lifecycle_manager: RAGLifecycleManager):
        """Initialize decision maker."""
        self.lifecycle = lifecycle_manager
    
    async def decide_corpus_action(
        self,
        repo_url: str,
        user_intent: str,
        user_session: str
    ) -> Dict[str, Any]:
        """
        Decide whether to create, reuse, or refresh corpus.
        
        Args:
            repo_url: Repository URL
            user_intent: What user wants to do
            user_session: User session ID
            
        Returns:
            Decision with action and reasoning
        """
        # Classify user intent
        intent_type = self._classify_intent(user_intent)
        
        # Check if we're in a development flow
        preserved_context = await self.lifecycle.get_preserved_context(user_session, repo_url)
        
        if preserved_context:
            return {
                "action": "continue_with_context",
                "context": preserved_context,
                "reasoning": "Continuing from previous development session"
            }
        
        # Make corpus decision
        decision = await self.lifecycle.should_create_corpus(
            repo_url=repo_url,
            user_session=user_session,
            query_type=intent_type
        )
        
        return decision
    
    def _classify_intent(self, user_intent: str) -> str:
        """Classify user intent to determine corpus strategy."""
        intent_lower = user_intent.lower()
        
        # Development intents need longer-lived corpus
        if any(word in intent_lower for word in ['implement', 'develop', 'code', 'create', 'build']):
            return 'development'
        
        # Modification intents need accurate corpus
        if any(word in intent_lower for word in ['change', 'modify', 'update', 'fix', 'refactor']):
            return 'modification'
        
        # Analysis intents can use shorter-lived corpus
        if any(word in intent_lower for word in ['analyze', 'understand', 'explore', 'review']):
            return 'analysis'
        
        # Search intents are quick
        if any(word in intent_lower for word in ['find', 'search', 'locate', 'where']):
            return 'search'
        
        return 'general'