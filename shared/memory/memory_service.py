"""
Memory service implementation for code generation system.
Provides storage and retrieval of code generation history.
"""

from typing import List, Dict, Any, Optional, Protocol
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

from .schemas import (
    MemoryEntry, 
    MemorySearchResult, 
    MemoryStats,
    MemoryQuery,
    MemoryCategory
)


# Configure logging
logger = logging.getLogger(__name__)


class MemoryBackend(Protocol):
    """Protocol for memory storage backends."""
    
    def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry and return its ID."""
        ...
    
    def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        ...
    
    def search(self, query: MemoryQuery) -> List[MemorySearchResult]:
        """Search for memory entries."""
        ...
    
    def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        ...
    
    def get_stats(self) -> MemoryStats:
        """Get memory statistics."""
        ...


class InMemoryBackend:
    """Simple in-memory storage backend."""
    
    def __init__(self):
        self.memories: Dict[str, MemoryEntry] = {}
    
    def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry."""
        self.memories[entry.id] = entry
        logger.debug(f"Stored memory entry: {entry.id}")
        return entry.id
    
    def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        entry = self.memories.get(entry_id)
        if entry:
            entry.update_access()
        return entry
    
    def search(self, query: MemoryQuery) -> List[MemorySearchResult]:
        """Search for memory entries using simple text matching."""
        results = []
        
        for entry in self.memories.values():
            # Apply filters
            if query.category and entry.category != query.category:
                continue
            
            if query.agent_name and entry.agent_name != query.agent_name:
                continue
            
            if query.tags and not any(tag in entry.tags for tag in query.tags):
                continue
            
            if entry.quality_score < query.min_quality_score:
                continue
            
            # Calculate similarity (simple text matching for now)
            similarity = self._calculate_similarity(
                query.text, 
                json.dumps(entry.data)
            )
            
            if similarity > 0.1:  # Threshold
                result = MemorySearchResult(
                    entry=entry,
                    similarity_score=similarity,
                    match_reason=f"Text similarity: {similarity:.2f}"
                )
                results.append(result)
        
        # Sort by relevance score
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        
        # Limit results
        return results[:query.max_results]
    
    def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        if entry_id in self.memories:
            del self.memories[entry_id]
            logger.debug(f"Deleted memory entry: {entry_id}")
            return True
        return False
    
    def get_stats(self) -> MemoryStats:
        """Get memory statistics."""
        if not self.memories:
            return MemoryStats()
        
        # Calculate statistics
        entries_by_category = {}
        entries_by_agent = {}
        total_quality = 0.0
        
        for entry in self.memories.values():
            # Count by category
            cat = entry.category.value
            entries_by_category[cat] = entries_by_category.get(cat, 0) + 1
            
            # Count by agent
            entries_by_agent[entry.agent_name] = entries_by_agent.get(entry.agent_name, 0) + 1
            
            # Sum quality scores
            total_quality += entry.quality_score
        
        # Get most accessed entries
        sorted_by_access = sorted(
            self.memories.values(), 
            key=lambda e: e.access_count, 
            reverse=True
        )
        most_accessed = sorted_by_access[:5]
        
        # Get recent entries
        sorted_by_time = sorted(
            self.memories.values(), 
            key=lambda e: e.created_at, 
            reverse=True
        )
        recent_entries = sorted_by_time[:5]
        
        return MemoryStats(
            total_entries=len(self.memories),
            entries_by_category=entries_by_category,
            entries_by_agent=entries_by_agent,
            average_quality_score=total_quality / len(self.memories),
            most_accessed=most_accessed,
            recent_entries=recent_entries
        )
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (0.0 to 1.0)."""
        if not text1 or not text2:
            return 0.0
        
        # Convert to lowercase for comparison
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Simple word overlap similarity
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


class MemoryService:
    """
    Main memory service for the code generation system.
    Provides high-level interface for memory operations.
    """
    
    def __init__(self, backend: Optional[MemoryBackend] = None):
        """Initialize with a storage backend."""
        self.backend = backend or InMemoryBackend()
        logger.info(f"Initialized MemoryService with {type(self.backend).__name__}")
    
    def store(self, entry: MemoryEntry) -> str:
        """
        Store a memory entry.
        
        Args:
            entry: The memory entry to store
            
        Returns:
            The ID of the stored entry
        """
        # Validate entry
        if not entry.data:
            raise ValueError("Memory entry must contain data")
        
        # Update timestamp
        entry.updated_at = datetime.now()
        
        # Store in backend
        entry_id = self.backend.store(entry)
        
        logger.info(f"Stored memory: category={entry.category.value}, "
                   f"agent={entry.agent_name}, id={entry_id}")
        
        return entry_id
    
    def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a memory entry by ID.
        
        Args:
            entry_id: The ID of the entry to retrieve
            
        Returns:
            The memory entry if found, None otherwise
        """
        return self.backend.retrieve(entry_id)
    
    def search_similar(
        self,
        request: str,
        category: Optional[str] = None,
        threshold: float = 0.7,
        max_results: int = 10
    ) -> List[MemoryEntry]:
        """
        Search for similar memory entries.
        
        Args:
            request: The request text to search for
            category: Optional category filter
            threshold: Minimum similarity threshold (0.0 to 1.0)
            max_results: Maximum number of results
            
        Returns:
            List of similar memory entries
        """
        # Create query
        query = MemoryQuery(
            text=request,
            category=MemoryCategory(category) if category else None,
            min_quality_score=threshold,
            max_results=max_results
        )
        
        # Search backend
        results = self.backend.search(query)
        
        # Filter by threshold and return entries
        filtered = [r.entry for r in results if r.similarity_score >= threshold]
        
        logger.debug(f"Found {len(filtered)} similar memories for: {request[:50]}...")
        
        return filtered
    
    def search(
        self,
        query: MemoryQuery
    ) -> List[MemorySearchResult]:
        """
        Search memory with detailed query parameters.
        
        Args:
            query: Detailed search query
            
        Returns:
            List of search results with scores
        """
        return self.backend.search(query)
    
    def get_by_category(self, category: MemoryCategory) -> List[MemoryEntry]:
        """Get all memories in a category."""
        query = MemoryQuery(
            text="",  # Empty text matches all
            category=category,
            max_results=1000  # Get all
        )
        results = self.backend.search(query)
        return [r.entry for r in results]
    
    def get_by_agent(self, agent_name: str) -> List[MemoryEntry]:
        """Get all memories from a specific agent."""
        query = MemoryQuery(
            text="",
            agent_name=agent_name,
            max_results=1000
        )
        results = self.backend.search(query)
        return [r.entry for r in results]
    
    def get_high_quality(self, min_score: float = 0.8) -> List[MemoryEntry]:
        """Get high-quality memory entries."""
        query = MemoryQuery(
            text="",
            min_quality_score=min_score,
            max_results=100
        )
        results = self.backend.search(query)
        return [r.entry for r in results]
    
    def cleanup_old_entries(self, days: int = 30) -> int:
        """
        Remove entries older than specified days.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of entries removed
        """
        cutoff = datetime.now() - timedelta(days=days)
        removed = 0
        
        # Get all entries (this is inefficient for large datasets)
        all_entries = self.backend.search(MemoryQuery(text="", max_results=10000))
        
        for result in all_entries:
            if result.entry.created_at < cutoff:
                if self.backend.delete(result.entry.id):
                    removed += 1
        
        logger.info(f"Cleaned up {removed} old memory entries")
        return removed
    
    def get_stats(self) -> MemoryStats:
        """Get memory statistics."""
        return self.backend.get_stats()


# Global memory service instance
_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """Get the global memory service instance."""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service


def initialize_memory_service(backend: Optional[MemoryBackend] = None) -> MemoryService:
    """Initialize the global memory service with a specific backend."""
    global _memory_service
    _memory_service = MemoryService(backend)
    return _memory_service