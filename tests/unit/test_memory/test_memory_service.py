"""
Unit tests for memory service.
Run with: pytest tests/unit/test_memory/test_memory_service.py -v
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.memory import (
    MemoryService,
    MemoryEntry,
    MemoryCategory,
    MemoryQuery,
    get_memory_service,
    initialize_memory_service
)


class TestMemoryEntry:
    """Test MemoryEntry schema."""
    
    def test_create_memory_entry(self):
        """Test creating a basic memory entry."""
        entry = MemoryEntry(
            category=MemoryCategory.CODE_GENERATION,
            agent_name="test_agent",
            data={"code": "def test(): pass"},
            quality_score=0.85,
            tags=["python", "function"]
        )
        
        assert entry.category == MemoryCategory.CODE_GENERATION
        assert entry.agent_name == "test_agent"
        assert entry.quality_score == 0.85
        assert "python" in entry.tags
        assert entry.id is not None
        assert entry.access_count == 0
    
    def test_quality_score_validation(self):
        """Test that quality score is clamped to valid range."""
        entry1 = MemoryEntry(
            category=MemoryCategory.CODE_GENERATION,
            agent_name="test",
            data={},
            quality_score=1.5  # Too high
        )
        assert entry1.quality_score == 1.0
        
        entry2 = MemoryEntry(
            category=MemoryCategory.CODE_GENERATION,
            agent_name="test",
            data={},
            quality_score=-0.5  # Too low
        )
        assert entry2.quality_score == 0.0
    
    def test_to_dict_from_dict(self):
        """Test serialization and deserialization."""
        original = MemoryEntry(
            category=MemoryCategory.CODE_REVIEW,
            agent_name="critic_agent",
            data={"review": "Code looks good"},
            quality_score=0.9,
            tags=["review", "positive"]
        )
        
        # Convert to dict
        data = original.to_dict()
        assert isinstance(data, dict)
        assert data["category"] == "code_review"
        
        # Convert back
        restored = MemoryEntry.from_dict(data)
        assert restored.category == original.category
        assert restored.agent_name == original.agent_name
        assert restored.quality_score == original.quality_score


class TestMemoryService:
    """Test MemoryService functionality."""
    
    @pytest.fixture
    def memory_service(self):
        """Create a fresh memory service for each test."""
        return MemoryService()
    
    def test_store_and_retrieve(self, memory_service):
        """Test storing and retrieving a memory."""
        entry = MemoryEntry(
            category=MemoryCategory.CODE_GENERATION,
            agent_name="test_agent",
            data={"code": "print('Hello')"},
            quality_score=0.8
        )
        
        # Store
        entry_id = memory_service.store(entry)
        assert entry_id is not None
        
        # Retrieve
        retrieved = memory_service.retrieve(entry_id)
        assert retrieved is not None
        assert retrieved.data["code"] == "print('Hello')"
        assert retrieved.access_count == 1  # Should increment
    
    def test_search_similar(self, memory_service):
        """Test searching for similar memories."""
        # Store some test memories
        memory_service.store(MemoryEntry(
            category=MemoryCategory.CODE_GENERATION,
            agent_name="test",
            data={"request": "create fibonacci function", "code": "def fib(n): ..."},
            quality_score=0.9
        ))
        
        memory_service.store(MemoryEntry(
            category=MemoryCategory.CODE_GENERATION,
            agent_name="test",
            data={"request": "create factorial function", "code": "def fact(n): ..."},
            quality_score=0.8
        ))
        
        memory_service.store(MemoryEntry(
            category=MemoryCategory.CODE_GENERATION,
            agent_name="test",
            data={"request": "create hello world", "code": "print('hello')"},
            quality_score=0.7
        ))
        
        # Search for similar
        results = memory_service.search_similar(
            "fibonacci function",
            category="code_generation",
            threshold=0.1
        )
        
        assert len(results) > 0
        assert "fibonacci" in str(results[0].data)
    
    def test_get_by_category(self, memory_service):
        """Test getting memories by category."""
        # Store memories in different categories
        memory_service.store(MemoryEntry(
            category=MemoryCategory.CODE_GENERATION,
            agent_name="test",
            data={"type": "generation"}
        ))
        
        memory_service.store(MemoryEntry(
            category=MemoryCategory.CODE_REVIEW,
            agent_name="test",
            data={"type": "review"}
        ))
        
        memory_service.store(MemoryEntry(
            category=MemoryCategory.CODE_GENERATION,
            agent_name="test",
            data={"type": "generation2"}
        ))
        
        # Get by category
        gen_memories = memory_service.get_by_category(MemoryCategory.CODE_GENERATION)
        assert len(gen_memories) == 2
        
        review_memories = memory_service.get_by_category(MemoryCategory.CODE_REVIEW)
        assert len(review_memories) == 1
    
    def test_get_high_quality(self, memory_service):
        """Test getting high quality memories."""
        # Store memories with different quality scores
        memory_service.store(MemoryEntry(
            category=MemoryCategory.CODE_GENERATION,
            agent_name="test",
            data={"quality": "low"},
            quality_score=0.5
        ))
        
        memory_service.store(MemoryEntry(
            category=MemoryCategory.CODE_GENERATION,
            agent_name="test",
            data={"quality": "high"},
            quality_score=0.9
        ))
        
        memory_service.store(MemoryEntry(
            category=MemoryCategory.CODE_GENERATION,
            agent_name="test",
            data={"quality": "medium"},
            quality_score=0.7
        ))
        
        # Get high quality only
        high_quality = memory_service.get_high_quality(min_score=0.8)
        assert len(high_quality) == 1
        assert high_quality[0].quality_score >= 0.8
    
    def test_get_stats(self, memory_service):
        """Test getting memory statistics."""
        # Add some test data
        for i in range(3):
            memory_service.store(MemoryEntry(
                category=MemoryCategory.CODE_GENERATION,
                agent_name="generator",
                data={"index": i},
                quality_score=0.7 + i * 0.1
            ))
        
        memory_service.store(MemoryEntry(
            category=MemoryCategory.CODE_REVIEW,
            agent_name="critic",
            data={"review": "test"},
            quality_score=0.85
        ))
        
        # Get stats
        stats = memory_service.get_stats()
        assert stats.total_entries == 4
        assert stats.entries_by_category["code_generation"] == 3
        assert stats.entries_by_category["code_review"] == 1
        assert stats.entries_by_agent["generator"] == 3
        assert stats.entries_by_agent["critic"] == 1
        assert 0.7 < stats.average_quality_score < 0.9


class TestGlobalMemoryService:
    """Test global memory service instance."""
    
    def test_get_memory_service(self):
        """Test getting global instance."""
        service1 = get_memory_service()
        service2 = get_memory_service()
        
        # Should be same instance
        assert service1 is service2
    
    def test_initialize_memory_service(self):
        """Test initializing with custom backend."""
        # Initialize with default backend
        service = initialize_memory_service()
        assert service is not None
        
        # Store something
        entry = MemoryEntry(
            category=MemoryCategory.GENERAL,
            agent_name="test",
            data={"test": "data"}
        )
        entry_id = service.store(entry)
        
        # Should be retrievable
        retrieved = service.retrieve(entry_id)
        assert retrieved is not None


# Quick test runner for development
if __name__ == "__main__":
    print("Running memory service tests...")
    
    # Create service
    service = MemoryService()
    
    # Test basic operations
    print("\n=== Testing Store and Retrieve ===")
    entry = MemoryEntry(
        category=MemoryCategory.CODE_GENERATION,
        agent_name="test_agent",
        data={
            "request": "create a hello world function",
            "code": "def hello(): print('Hello, World!')"
        },
        quality_score=0.85,
        tags=["python", "simple"]
    )
    
    entry_id = service.store(entry)
    print(f"Stored entry with ID: {entry_id}")
    
    retrieved = service.retrieve(entry_id)
    print(f"Retrieved: {retrieved.data}")
    
    print("\n=== Testing Search ===")
    # Store more entries
    service.store(MemoryEntry(
        category=MemoryCategory.CODE_GENERATION,
        agent_name="test_agent",
        data={
            "request": "create fibonacci function",
            "code": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"
        },
        quality_score=0.9
    ))
    
    results = service.search_similar("fibonacci", threshold=0.1)
    print(f"Found {len(results)} similar entries")
    for result in results:
        print(f"  - {result.data.get('request', 'No request')}")
    
    print("\n=== Testing Stats ===")
    stats = service.get_stats()
    print(f"Total entries: {stats.total_entries}")
    print(f"Average quality: {stats.average_quality_score:.2f}")
    print(f"By category: {stats.entries_by_category}")
    
    print("\nTo run full test suite: pytest tests/unit/test_memory/test_memory_service.py -v")