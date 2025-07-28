"""
Memory storage backends.
TODO: Implement Vertex AI and FileSystem backends.
"""

from typing import List, Optional
from .schemas import MemoryEntry, MemorySearchResult, MemoryStats, MemoryQuery
from .memory_service import MemoryBackend


class VertexAIBackend:
    """
    Vertex AI backend for memory storage.
    TODO: Implement using Vertex AI Matching Engine or similar.
    """
    
    def __init__(self, project_id: str, location: str):
        self.project_id = project_id
        self.location = location
        raise NotImplementedError("Vertex AI backend not yet implemented")
    
    def store(self, entry: MemoryEntry) -> str:
        raise NotImplementedError()
    
    def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        raise NotImplementedError()
    
    def search(self, query: MemoryQuery) -> List[MemorySearchResult]:
        raise NotImplementedError()
    
    def delete(self, entry_id: str) -> bool:
        raise NotImplementedError()
    
    def get_stats(self) -> MemoryStats:
        raise NotImplementedError()


class FileSystemBackend:
    """
    File system backend for memory storage.
    TODO: Implement JSON file-based storage.
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        raise NotImplementedError("FileSystem backend not yet implemented")
    
    def store(self, entry: MemoryEntry) -> str:
        raise NotImplementedError()
    
    def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        raise NotImplementedError()
    
    def search(self, query: MemoryQuery) -> List[MemorySearchResult]:
        raise NotImplementedError()
    
    def delete(self, entry_id: str) -> bool:
        raise NotImplementedError()
    
    def get_stats(self) -> MemoryStats:
        raise NotImplementedError()