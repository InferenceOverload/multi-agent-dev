"""Background cleanup scheduler for RAG corpora."""

import asyncio
from datetime import datetime, timedelta
from src.rag.rag_lifecycle_manager import RAGLifecycleManager


class CorpusCleanupScheduler:
    """
    Schedules and runs periodic cleanup of stale RAG corpora.
    Designed to run as a background task in production.
    """
    
    def __init__(self, lifecycle_manager: RAGLifecycleManager = None):
        """Initialize cleanup scheduler."""
        self.lifecycle_manager = lifecycle_manager or RAGLifecycleManager()
        self.is_running = False
        self.cleanup_interval_hours = 1  # Run every hour
        self.last_cleanup = None
        self.cleanup_stats = []
    
    async def start(self):
        """Start the cleanup scheduler."""
        self.is_running = True
        print(f"🧹 Cleanup scheduler started at {datetime.now()}")
        
        while self.is_running:
            try:
                # Run cleanup
                stats = await self.lifecycle_manager.cleanup_stale_corpora()
                
                # Record stats
                self.last_cleanup = datetime.now()
                self.cleanup_stats.append({
                    'timestamp': self.last_cleanup,
                    'stats': stats
                })
                
                # Keep only last 24 hours of stats
                cutoff = datetime.now() - timedelta(hours=24)
                self.cleanup_stats = [
                    s for s in self.cleanup_stats 
                    if s['timestamp'] > cutoff
                ]
                
                print(f"✅ Cleanup completed: {stats}")
                
                # Wait for next interval
                await asyncio.sleep(self.cleanup_interval_hours * 3600)
                
            except Exception as e:
                print(f"❌ Cleanup error: {e}")
                # Wait before retrying
                await asyncio.sleep(300)  # 5 minutes
    
    def stop(self):
        """Stop the cleanup scheduler."""
        self.is_running = False
        print(f"🛑 Cleanup scheduler stopped at {datetime.now()}")
    
    def get_stats(self) -> Dict:
        """Get cleanup statistics."""
        total_deleted = sum(s['stats'].get('deleted', 0) for s in self.cleanup_stats)
        total_kept = sum(s['stats'].get('kept', 0) for s in self.cleanup_stats)
        
        return {
            'last_cleanup': self.last_cleanup,
            'cleanup_count': len(self.cleanup_stats),
            'total_deleted_24h': total_deleted,
            'total_kept_24h': total_kept,
            'next_cleanup': self.last_cleanup + timedelta(hours=self.cleanup_interval_hours) if self.last_cleanup else None
        }


# For deployment on Vertex AI or Cloud Run
async def run_cleanup_service():
    """Run cleanup service as a standalone process."""
    scheduler = CorpusCleanupScheduler()
    
    # Start cleanup in background
    cleanup_task = asyncio.create_task(scheduler.start())
    
    # Keep service running
    try:
        await cleanup_task
    except KeyboardInterrupt:
        scheduler.stop()
        print("Cleanup service stopped")


if __name__ == "__main__":
    # Run as standalone service
    asyncio.run(run_cleanup_service())