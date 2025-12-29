import os

from dotenv import load_dotenv
from pydantic import BaseModel

# 加载环境变量
load_dotenv()


class Config(BaseModel):
    # Obsidian配置
    OBSIDIAN_VAULT_PATH: str = os.getenv(
        "OBSIDIAN_VAULT_PATH", "/Users/xiejindong/Desktop/rag_local_km_tset"
    )
    OBSIDIAN_API_KEY: str = os.getenv("OBSIDIAN_API_KEY", "")
    OBSIDIAN_API_URL: str = os.getenv("OBSIDIAN_API_URL", "http://localhost:5136")

    # Ollama配置
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "deepseek-coder")
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # API嵌入模型配置
    OLLAMA_API_KEY: str = os.getenv("OLLAMA_API_KEY", "")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    # 向量数据库配置
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./vector_store")

    # 分块配置
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # 检索配置
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    class Config:
        env_file = ".env"
