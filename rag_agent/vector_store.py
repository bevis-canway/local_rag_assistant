import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import logging
import hashlib
import os
import httpx
from openai import OpenAI
import time

logger = logging.getLogger(__name__)


class VectorStore:
    """
    向量数据库管理器
    使用ChromaDB作为向量存储，参考aidev项目的RAG实现
    支持本地模型和API模型两种嵌入方式
    """

    def __init__(self, persist_path: str = "./vector_store", model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection("obsidian_notes")

        # 检查是否配置了OpenAI兼容的API
        openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OLLAMA_API_KEY")
        openai_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

        if openai_api_key:
            # 使用API嵌入
            logger.info("使用API嵌入模型")
            self.use_api_embeddings = True
            # 存储配置，延迟初始化OpenAI客户端
            self.api_key = openai_api_key
            self.base_url = openai_base_url
            self.embedding_model = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")  # 使用用户偏好的小魔仙模型
        else:
            # 使用本地模型
            logger.info("使用本地嵌入模型")
            self.use_api_embeddings = False
            try:
                # 尝试使用指定模型，如果无法下载则使用本地缓存或替代方案
                self.embedder = SentenceTransformer(model_name, cache_folder="./model_cache")
            except Exception as e:
                logger.warning(f"无法加载模型 {model_name}: {e}")
                logger.info("尝试使用替代模型...")
                try:
                    # 尝试使用较小的替代模型
                    self.embedder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder="./model_cache")
                except Exception as e2:
                    logger.error(f"无法加载替代模型: {e2}")
                    raise RuntimeError("无法加载嵌入模型，请检查网络连接或手动下载模型，或者配置OpenAI兼容API")

    def _get_embeddings(self, texts: List[str]):
        """获取文本嵌入向量"""
        if self.use_api_embeddings:
            # 使用API获取嵌入，逐个处理以避免Ollama服务过载
            embeddings = []
            
            for i, text in enumerate(texts):
                success = False
                max_retries = 2  # 减少重试次数以避免长时间等待
                
                for attempt in range(max_retries):
                    try:
                        # 每次请求都创建新的客户端，避免连接复用问题
                        http_client = httpx.Client(timeout=180.0)  # 增加超时时间
                        client = OpenAI(
                            api_key=self.api_key,
                            base_url=self.base_url,
                            http_client=http_client
                        )
                        
                        # 对于Ollama，需要使用embeddings端点
                        response = client.embeddings.create(
                            input=text,
                            model=self.embedding_model
                        )
                        embeddings.append(response.data[0].embedding)
                        success = True
                        
                        # 关闭HTTP客户端
                        http_client.close()
                        
                        logger.debug(f"成功处理文本 {i+1}/{len(texts)}")
                        
                        # 在请求之间添加较长延迟以避免过载Ollama服务
                        if i < len(texts) - 1:  # 不在最后一个请求后延迟
                            time.sleep(2.0)  # 增加延迟时间
                        
                        break  # 成功后跳出重试循环
                    except Exception as e:
                        logger.error(f"文本嵌入失败 (文本索引 {i}, 尝试 {attempt + 1}/{max_retries}): {e}")
                        
                        # 关闭客户端以防万一
                        if 'http_client' in locals():
                            try:
                                http_client.close()
                            except:
                                pass
                        
                        if attempt < max_retries - 1:
                            # 等待后重试
                            wait_time = 3 ** attempt  # 更长的指数退避
                            logger.info(f"等待 {wait_time} 秒后重试 (尝试 {attempt + 1})...")
                            time.sleep(wait_time)
                        else:
                            # 所有重试都失败，抛出异常
                            logger.error(f"在处理文本索引 {i} 时所有重试都失败")
                            raise
                            
            return embeddings
        else:
            # 使用本地模型获取嵌入
            try:
                return self.embedder.encode(texts).tolist()
            except Exception as e:
                logger.error(f"本地嵌入生成失败: {e}")
                raise

    def add_documents(self, documents: List[Dict[str, str]]):
        """
        添加文档到向量库
        documents: [{"id": "doc_id", "content": "content", "metadata": {...}}]
        """
        if not documents:
            return

        # 提取内容和元数据
        doc_ids = [doc["id"] for doc in documents]
        doc_contents = [doc["content"] for doc in documents]
        doc_metadatas = [doc.get("metadata", {}) for doc in documents]

        # 逐个处理以最大程度减少Ollama服务压力
        embeddings = self._get_embeddings(doc_contents)

        # 添加到集合
        self.collection.add(
            documents=doc_contents,
            metadatas=doc_metadatas,
            ids=doc_ids,
            embeddings=embeddings
        )
        logger.info(f"已添加 {len(documents)} 个文档到向量库")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        根据查询检索最相关的文档
        参考aidev项目中的知识库检索逻辑
        """
        try:
            query_embeddings = self._get_embeddings([query])
        except Exception as e:
            logger.error(f"生成查询嵌入向量失败: {e}")
            return []

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k
        )

        # 格式化结果，包含相似度分数
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "similarity": 1 - results["distances"][0][i]  # 转换为相似度分数
            })

        return formatted_results

    def clear(self):
        """
        清空向量库
        """
        self.client.delete_collection("obsidian_notes")
        self.collection = self.client.get_or_create_collection("obsidian_notes")
        logger.info("已清空向量库")