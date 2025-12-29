import logging
from typing import Dict, List

from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    检索器
    负责根据用户查询从向量库中检索相关信息
    参考aidev项目中的检索逻辑实现
    """

    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Dict]:
        """
        检索与查询最相关的文档片段
        参考aidev项目中的知识库检索实现
        """
        try:
            results = self.vector_store.search(query, self.top_k)
            logger.info(f"检索到 {len(results)} 个相关文档片段")
            return results
        except Exception as e:
            logger.error(f"检索过程出错: {e}")
            return []

    def retrieve_and_format(self, query: str) -> str:
        """
        检索并格式化返回结果
        """
        results = self.retrieve(query)
        if not results:
            return "未找到相关文档内容。"

        formatted_results = []
        for result in results:
            content = (
                result["content"][:500] + "..."
                if len(result["content"]) > 500
                else result["content"]
            )
            formatted_results.append(
                f"文档: {result['metadata'].get('title', 'Unknown')}\n"
                f"路径: {result['metadata'].get('path', 'Unknown')}\n"
                f"内容: {content}\n"
                f"相似度: {result['similarity']:.3f}\n"
            )

        return "\n" + "=" * 50 + "\n".join(formatted_results) + "\n" + "=" * 50
