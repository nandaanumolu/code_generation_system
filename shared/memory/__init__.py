"""
Memory service for storing and retrieving code generation history.
Provides both in-memory and persistent storage options.
"""

from .memory_service import (
    MemoryService,
    get_memory_service,
    initialize_memory_service
)

from .schemas import (
    MemoryEntry,
    MemorySearchResult,
    MemoryStats,
    MemoryCategory,
    MemoryQuery
)

# TODO: Uncomment when backends are implemented
# from .backends import (
#     InMemoryBackend,
#     VertexAIBackend,
#     FileSystemBackend
# )

__all__ = [
    # Service
    'MemoryService',
    'get_memory_service',
    'initialize_memory_service',
    
    # Schemas
    'MemoryEntry',
    'MemorySearchResult',
    'MemoryStats',
    'MemoryCategory',
    'MemoryQuery',
    
    # TODO: Add backends when implemented
    # 'InMemoryBackend',
    # 'VertexAIBackend',
    # 'FileSystemBackend',
]