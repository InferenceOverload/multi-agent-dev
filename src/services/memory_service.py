"""Memory service for agent context persistence using ADK patterns."""

import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    from google.adk import memory as adk_memory
except ImportError:
    adk_memory = None

from ..models.base import MemoryEntry, AgentContext
from ..config.config import get_config
from ..utils.logger import get_logger


logger = get_logger(__name__)
config = get_config()


class MemoryService:
    """Memory service for managing agent memory and context."""
    
    def __init__(self, use_vertex_memory: bool = False):
        """Initialize memory service.
        
        Args:
            use_vertex_memory: Whether to use Vertex AI Memory Bank
        """
        self.use_vertex_memory = use_vertex_memory
        self.redis_client = None
        self.memory_bank = None
        
        # Initialize Redis for short-term memory
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=config.redis_host,
                    port=config.redis_port,
                    db=config.redis_db,
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info("Connected to Redis for memory storage")
            except Exception as e:
                logger.warning(f"Redis not available: {e}. Using in-memory storage.")
                self.redis_client = None
                self.memory_store = {}
        else:
            logger.warning("Redis module not installed. Using in-memory storage.")
            self.redis_client = None
            self.memory_store = {}
        
        # Initialize Vertex AI Memory Bank if enabled
        if use_vertex_memory:
            try:
                self.memory_bank = adk_memory.VertexMemoryBank(
                    project=config.google_cloud_project,
                    location=config.google_cloud_location
                )
                logger.info("Initialized Vertex AI Memory Bank")
            except Exception as e:
                logger.warning(f"Vertex Memory Bank not available: {e}")
    
    async def store(self, entry: MemoryEntry) -> bool:
        """Store memory entry.
        
        Args:
            entry: Memory entry to store
            
        Returns:
            Success status
        """
        try:
            # Generate unique key
            entry.id = self._generate_memory_id(entry)
            
            # Store in Redis or local memory
            if self.redis_client:
                key = f"memory:{entry.session_id}:{entry.agent_name}:{entry.key}"
                
                # Set expiration based on memory type
                ttl = self._get_ttl(entry.type)
                
                self.redis_client.setex(
                    key,
                    ttl,
                    json.dumps(entry.dict())
                )
            else:
                # Fallback to in-memory storage
                key = f"{entry.session_id}:{entry.agent_name}:{entry.key}"
                self.memory_store[key] = entry.dict()
            
            # Store in Vertex Memory Bank for long-term memory
            if self.use_vertex_memory and entry.type == "long_term":
                await self._store_in_memory_bank(entry)
            
            logger.debug(f"Stored memory entry: {entry.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False
    
    async def retrieve(
        self,
        session_id: str,
        agent_name: str,
        key: str,
        memory_type: Optional[str] = None
    ) -> Optional[Any]:
        """Retrieve memory entry.
        
        Args:
            session_id: Session ID
            agent_name: Agent name
            key: Memory key
            memory_type: Optional memory type filter
            
        Returns:
            Retrieved value or None
        """
        try:
            # Try Redis first
            if self.redis_client:
                redis_key = f"memory:{session_id}:{agent_name}:{key}"
                data = self.redis_client.get(redis_key)
                
                if data:
                    entry = json.loads(data)
                    if not memory_type or entry.get("type") == memory_type:
                        # Update access time
                        entry["last_accessed"] = datetime.utcnow().isoformat()
                        entry["usage_count"] = entry.get("usage_count", 0) + 1
                        self.redis_client.setex(
                            redis_key,
                            self._get_ttl(entry.get("type", "short_term")),
                            json.dumps(entry)
                        )
                        return entry.get("value")
            else:
                # Fallback to in-memory storage
                mem_key = f"{session_id}:{agent_name}:{key}"
                if mem_key in self.memory_store:
                    entry = self.memory_store[mem_key]
                    if not memory_type or entry.get("type") == memory_type:
                        entry["last_accessed"] = datetime.utcnow().isoformat()
                        entry["usage_count"] = entry.get("usage_count", 0) + 1
                        return entry.get("value")
            
            # Try Vertex Memory Bank for long-term memory
            if self.use_vertex_memory and memory_type == "long_term":
                return await self._retrieve_from_memory_bank(session_id, agent_name, key)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve memory: {e}")
            return None
    
    async def retrieve_relevant_memory(
        self,
        session_id: str,
        agent_name: str,
        context: Dict[str, Any],
        limit: int = 10
    ) -> Dict[str, Any]:
        """Retrieve relevant memory based on context.
        
        Args:
            session_id: Session ID
            agent_name: Agent name
            context: Current context for relevance matching
            limit: Maximum number of memories to retrieve
            
        Returns:
            Relevant memories
        """
        relevant_memories = {
            "short_term": [],
            "long_term": [],
            "patterns": [],
            "success_history": []
        }
        
        try:
            # Search Redis for relevant memories
            if self.redis_client:
                pattern = f"memory:{session_id}:{agent_name}:*"
                keys = self.redis_client.keys(pattern)
                
                # Calculate relevance scores
                scored_memories = []
                for key in keys[:100]:  # Limit search
                    data = self.redis_client.get(key)
                    if data:
                        entry = json.loads(data)
                        relevance = self._calculate_relevance(entry, context)
                        if relevance > 0.3:
                            scored_memories.append((entry, relevance))
                
                # Sort by relevance and recency
                scored_memories.sort(
                    key=lambda x: (x[1], x[0].get("last_accessed", "")),
                    reverse=True
                )
                
                # Organize by type
                for entry, score in scored_memories[:limit]:
                    memory_type = entry.get("type", "short_term")
                    if memory_type in relevant_memories:
                        relevant_memories[memory_type].append({
                            "key": entry.get("key"),
                            "value": entry.get("value"),
                            "relevance": score,
                            "created_at": entry.get("created_at"),
                            "usage_count": entry.get("usage_count", 0)
                        })
            
            # Search Vertex Memory Bank if available
            if self.use_vertex_memory:
                long_term = await self._search_memory_bank(context, limit)
                relevant_memories["long_term"].extend(long_term)
            
        except Exception as e:
            logger.error(f"Failed to retrieve relevant memory: {e}")
        
        return relevant_memories
    
    async def store_execution_result(self, context: AgentContext) -> bool:
        """Store agent execution result.
        
        Args:
            context: Agent execution context
            
        Returns:
            Success status
        """
        try:
            # Store as short-term memory
            entry = MemoryEntry(
                session_id=context.session_id,
                agent_name=context.agent_name,
                type="short_term",
                key=f"execution_{context.created_at.isoformat()}",
                value={
                    "input": context.input_data,
                    "output": context.output,
                    "status": context.status,
                    "execution_time": context.execution_time,
                    "tools_used": context.tools_used
                }
            )
            
            await self.store(entry)
            
            # Store patterns if identified
            if context.output and "patterns" in context.output:
                pattern_entry = MemoryEntry(
                    session_id=context.session_id,
                    agent_name=context.agent_name,
                    type="pattern",
                    key=f"patterns_{context.created_at.isoformat()}",
                    value=context.output["patterns"]
                )
                await self.store(pattern_entry)
            
            # Store success/failure for learning
            if context.status == "completed":
                success_entry = MemoryEntry(
                    session_id=context.session_id,
                    agent_name=context.agent_name,
                    type="success",
                    key=f"success_{context.created_at.isoformat()}",
                    value={
                        "input_type": self._categorize_input(context.input_data),
                        "approach": context.output.get("approach") if context.output else None,
                        "execution_time": context.execution_time
                    }
                )
                await self.store(success_entry)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store execution result: {e}")
            return False
    
    def _generate_memory_id(self, entry: MemoryEntry) -> str:
        """Generate unique ID for memory entry.
        
        Args:
            entry: Memory entry
            
        Returns:
            Unique ID
        """
        content = f"{entry.session_id}:{entry.agent_name}:{entry.key}:{entry.created_at}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_ttl(self, memory_type: str) -> int:
        """Get TTL for memory type.
        
        Args:
            memory_type: Type of memory
            
        Returns:
            TTL in seconds
        """
        ttl_map = {
            "short_term": 3600,  # 1 hour
            "long_term": 86400 * 30,  # 30 days
            "pattern": 86400 * 7,  # 7 days
            "success": 86400 * 14  # 14 days
        }
        return ttl_map.get(memory_type, 3600)
    
    def _calculate_relevance(
        self,
        entry: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Calculate relevance of memory to current context.
        
        Args:
            entry: Memory entry
            context: Current context
            
        Returns:
            Relevance score (0-1)
        """
        score = 0.0
        
        # Check key similarity
        entry_key = entry.get("key", "").lower()
        context_str = json.dumps(context).lower()
        
        key_words = entry_key.split("_")
        matches = sum(1 for word in key_words if word in context_str)
        if key_words:
            score += (matches / len(key_words)) * 0.3
        
        # Check value similarity (if dict)
        entry_value = entry.get("value")
        if isinstance(entry_value, dict) and isinstance(context, dict):
            # Check for common keys
            common_keys = set(entry_value.keys()) & set(context.keys())
            if entry_value and context:
                score += (len(common_keys) / max(len(entry_value), len(context))) * 0.4
        
        # Recency bonus
        created_at = entry.get("created_at")
        if created_at:
            try:
                age = (datetime.utcnow() - datetime.fromisoformat(created_at)).days
                if age < 1:
                    score += 0.2
                elif age < 7:
                    score += 0.1
            except:
                pass
        
        # Usage frequency bonus
        usage_count = entry.get("usage_count", 0)
        if usage_count > 5:
            score += 0.1
        
        return min(score, 1.0)
    
    def _categorize_input(self, input_data: Dict[str, Any]) -> str:
        """Categorize input type for learning.
        
        Args:
            input_data: Input data
            
        Returns:
            Input category
        """
        if "requirement" in input_data or "requirement_text" in input_data:
            return "requirement_analysis"
        elif "repo_url" in input_data:
            return "repository_analysis"
        elif "story" in input_data:
            return "story_creation"
        elif "code" in input_data or "implementation" in input_data:
            return "development"
        elif "pr" in input_data or "pull_request" in input_data:
            return "pull_request"
        else:
            return "general"
    
    async def _store_in_memory_bank(self, entry: MemoryEntry) -> bool:
        """Store in Vertex AI Memory Bank.
        
        Args:
            entry: Memory entry
            
        Returns:
            Success status
        """
        if not self.memory_bank:
            return False
        
        try:
            # Format for Vertex Memory Bank
            memory_content = {
                "id": entry.id,
                "content": json.dumps(entry.value),
                "metadata": {
                    "session_id": entry.session_id,
                    "agent_name": entry.agent_name,
                    "key": entry.key,
                    "type": entry.type,
                    "created_at": entry.created_at.isoformat()
                }
            }
            
            await self.memory_bank.store(memory_content)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store in Memory Bank: {e}")
            return False
    
    async def _retrieve_from_memory_bank(
        self,
        session_id: str,
        agent_name: str,
        key: str
    ) -> Optional[Any]:
        """Retrieve from Vertex AI Memory Bank.
        
        Args:
            session_id: Session ID
            agent_name: Agent name
            key: Memory key
            
        Returns:
            Retrieved value or None
        """
        if not self.memory_bank:
            return None
        
        try:
            # Search by metadata
            query = {
                "filters": {
                    "session_id": session_id,
                    "agent_name": agent_name,
                    "key": key
                }
            }
            
            results = await self.memory_bank.search(query)
            if results:
                return json.loads(results[0]["content"])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve from Memory Bank: {e}")
            return None
    
    async def _search_memory_bank(
        self,
        context: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search Vertex Memory Bank for relevant memories.
        
        Args:
            context: Search context
            limit: Maximum results
            
        Returns:
            List of relevant memories
        """
        if not self.memory_bank:
            return []
        
        try:
            # Create semantic search query
            query_text = json.dumps(context)[:1000]  # Limit query size
            
            results = await self.memory_bank.semantic_search(
                query_text,
                limit=limit
            )
            
            memories = []
            for result in results:
                content = json.loads(result["content"])
                memories.append({
                    "key": result["metadata"].get("key"),
                    "value": content,
                    "relevance": result.get("score", 0.5),
                    "created_at": result["metadata"].get("created_at")
                })
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to search Memory Bank: {e}")
            return []
    
    async def cleanup_old_memories(self, days: int = 30) -> int:
        """Clean up old memories.
        
        Args:
            days: Age threshold in days
            
        Returns:
            Number of memories cleaned
        """
        cleaned = 0
        
        try:
            if self.redis_client:
                # Get all memory keys
                keys = self.redis_client.keys("memory:*")
                
                for key in keys:
                    data = self.redis_client.get(key)
                    if data:
                        entry = json.loads(data)
                        created_at = entry.get("created_at")
                        
                        if created_at:
                            age = (datetime.utcnow() - datetime.fromisoformat(created_at)).days
                            if age > days:
                                self.redis_client.delete(key)
                                cleaned += 1
            
            logger.info(f"Cleaned {cleaned} old memories")
            
        except Exception as e:
            logger.error(f"Failed to cleanup memories: {e}")
        
        return cleaned