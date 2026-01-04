"""
小魔仙RAG智能体 - 基于本地模型的RAG系统
实现防止大模型幻觉的多种策略
"""

from .main import RAGAgent
from .config import Config

__all__ = ["RAGAgent", "Config"]