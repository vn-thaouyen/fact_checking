"""
Configuration management for the fact-checking system
Handles environment variables and settings
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class FactCheckConfig:
    """Configuration settings for fact-checking system"""
    
    # API Configuration
    hf_token: str
    
    # Model Configuration
    embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "Qwen/Qwen2.5-32B-Instruct"
    language: str = "vi"  # Vietnamese support
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    
    # Vector Store Configuration
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "news_facts"
    
    # Query Engine Configuration
    similarity_top_k: int = 5
    citation_chunk_size: int = 512
    
    # Inference Configuration
    batch_size: int = 10
    confidence_threshold: float = 0.5
    
    # Cache Configuration
    embeddings_cache_dir: str = "./embeddings_cache"
    
    @classmethod
    def from_env(cls, env_file: str = ".env") -> "FactCheckConfig":
        """
        Load configuration from environment variables
        
        Args:
            env_file: Path to .env file (defaults to .env in current directory)
        
        Returns:
            FactCheckConfig instance
        
        Raises:
            ValueError: If HF_TOKEN is not set
        """
        # Load environment variables from .env file
        load_dotenv(env_file)
        
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN is not found in environment. "
                "Please set it in .env file or as environment variable."
            )
        
        return cls(
            hf_token=hf_token,
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            llm_model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            language=os.getenv("LANGUAGE", "vi"),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_data"),
            chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "news_facts"),
            similarity_top_k=int(os.getenv("SIMILARITY_TOP_K", "5")),
            citation_chunk_size=int(os.getenv("CITATION_CHUNK_SIZE", "512")),
            batch_size=int(os.getenv("BATCH_SIZE", "10")),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
            embeddings_cache_dir=os.getenv("EMBEDDINGS_CACHE_DIR", "./embeddings_cache"),
        )
    
    def validate(self) -> bool:
        """
        Validate configuration settings
        
        Returns:
            True if all settings are valid
        
        Raises:
            ValueError: If any setting is invalid
        """
        if not self.hf_token:
            raise ValueError("HF_TOKEN is required")
        
        if not self.embedding_model:
            raise ValueError("embedding_model is required")
        
        if not self.llm_model:
            raise ValueError("llm_model is required")
        
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError("llm_temperature must be between 0.0 and 2.0")
        
        if self.llm_max_tokens < 1:
            raise ValueError("llm_max_tokens must be at least 1")
        
        if self.similarity_top_k < 1:
            raise ValueError("similarity_top_k must be at least 1")
        
        if self.citation_chunk_size < 1:
            raise ValueError("citation_chunk_size must be at least 1")
        
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
        return True
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            "hf_token": "***" if self.hf_token else None,  # Hide API key
            "embedding_model": self.embedding_model,
            "language": self.language,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "chroma_persist_dir": self.chroma_persist_dir,
            "chroma_collection_name": self.chroma_collection_name,
            "similarity_top_k": self.similarity_top_k,
            "citation_chunk_size": self.citation_chunk_size,
            "batch_size": self.batch_size,
            "confidence_threshold": self.confidence_threshold,
        }


def get_config(env_file: str = ".env") -> FactCheckConfig:
    """
    Get and validate configuration
    
    Args:
        env_file: Path to .env file
    
    Returns:
        Validated FactCheckConfig instance
    """
    config = FactCheckConfig.from_env(env_file)
    config.validate()
    return config
