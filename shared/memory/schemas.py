"""
Schemas for memory service data structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import uuid


class MemoryCategory(str, Enum):
    """Categories for memory entries."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    REFACTORING = "refactoring"
    ERROR_SOLUTION = "error_solution"
    BEST_PRACTICE = "best_practice"
    GENERAL = "general"


@dataclass
class MemoryEntry:
    """
    Represents a single memory entry.
    
    Attributes:
        id: Unique identifier for the entry
        category: Category of the memory
        agent_name: Name of the agent that created this memory
        data: The actual memory data (flexible structure)
        quality_score: Quality score (0.0 to 1.0)
        tags: List of tags for categorization
        metadata: Additional metadata
        created_at: Creation timestamp
        updated_at: Last update timestamp
        access_count: Number of times this memory was accessed
    """
    category: MemoryCategory
    agent_name: str
    data: Dict[str, Any]
    quality_score: float = 0.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    def __post_init__(self):
        """Validate data after initialization."""
        # Ensure quality score is in valid range
        self.quality_score = max(0.0, min(1.0, self.quality_score))
        
        # Convert string category to enum if needed
        if isinstance(self.category, str):
            try:
                self.category = MemoryCategory(self.category)
            except ValueError:
                self.category = MemoryCategory.GENERAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "category": self.category.value,
            "agent_name": self.agent_name,
            "data": self.data,
            "quality_score": self.quality_score,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary."""
        # Handle datetime conversion
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        
        return cls(**data)
    
    def update_access(self):
        """Update access count and timestamp."""
        self.access_count += 1
        self.updated_at = datetime.now()


@dataclass
class MemorySearchResult:
    """
    Result from a memory search operation.
    
    Attributes:
        entry: The memory entry
        similarity_score: Similarity score (0.0 to 1.0)
        relevance_score: Combined relevance score
        match_reason: Explanation of why this was matched
    """
    entry: MemoryEntry
    similarity_score: float = 0.0
    relevance_score: float = 0.0
    match_reason: str = ""
    
    def __post_init__(self):
        """Calculate relevance score if not provided."""
        if self.relevance_score == 0.0:
            # Combine similarity and quality scores
            self.relevance_score = (
                self.similarity_score * 0.7 + 
                self.entry.quality_score * 0.3
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry": self.entry.to_dict(),
            "similarity_score": self.similarity_score,
            "relevance_score": self.relevance_score,
            "match_reason": self.match_reason
        }


@dataclass
class MemoryStats:
    """
    Statistics about the memory store.
    
    Attributes:
        total_entries: Total number of memory entries
        entries_by_category: Count of entries per category
        entries_by_agent: Count of entries per agent
        average_quality_score: Average quality score across all entries
        most_accessed: List of most accessed entries
        recent_entries: List of recently added entries
    """
    total_entries: int = 0
    entries_by_category: Dict[str, int] = field(default_factory=dict)
    entries_by_agent: Dict[str, int] = field(default_factory=dict)
    average_quality_score: float = 0.0
    most_accessed: List[MemoryEntry] = field(default_factory=list)
    recent_entries: List[MemoryEntry] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_entries": self.total_entries,
            "entries_by_category": self.entries_by_category,
            "entries_by_agent": self.entries_by_agent,
            "average_quality_score": self.average_quality_score,
            "most_accessed_count": len(self.most_accessed),
            "recent_entries_count": len(self.recent_entries)
        }


@dataclass
class MemoryQuery:
    """
    Query parameters for searching memory.
    
    Attributes:
        text: Text to search for
        category: Optional category filter
        agent_name: Optional agent name filter
        tags: Optional tag filters
        min_quality_score: Minimum quality score
        max_results: Maximum number of results
        include_metadata: Whether to include metadata in results
    """
    text: str
    category: Optional[MemoryCategory] = None
    agent_name: Optional[str] = None
    tags: Optional[List[str]] = None
    min_quality_score: float = 0.0
    max_results: int = 10
    include_metadata: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "category": self.category.value if self.category else None,
            "agent_name": self.agent_name,
            "tags": self.tags,
            "min_quality_score": self.min_quality_score,
            "max_results": self.max_results,
            "include_metadata": self.include_metadata
        }