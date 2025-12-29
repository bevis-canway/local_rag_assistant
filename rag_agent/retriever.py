import logging
import os
from typing import Dict, List

import ollama

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
        如果未找到相关文档，则使用本地大模型回答
        """
        results = self.retrieve(query)
        if not results:
            logger.info("未找到相关文档内容，使用本地大模型回答...")
            return self._generate_response_with_llm(query)

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

    def _generate_response_with_llm(self, query: str) -> str:
        """
        使用本地大模型生成回答
        """
        try:
            # 从环境变量获取模型名称，默认使用配置的模型
            model_name = os.getenv("OLLAMA_MODEL", "deepseek-coder")

            # 构建提示词，告知模型没有找到相关文档
            prompt = f"""你是一个智能助手。用户的问题是："{query}"

我没有在本地知识库中找到与您问题相关的内容。

请根据你的通用知识尽力回答用户的问题。如果问题涉及非常具体或专业的内容，而你无法准确回答，请诚实地告知用户你无法提供准确答案，并建议用户查阅相关资料或寻求专业帮助。"""

            # 调用 Ollama 模型
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )

            llm_answer = response["message"]["content"]
            logger.info("已使用本地大模型生成回答")

            return f"未在本地知识库中找到相关内容。\n\n智能助手回答：\n{llm_answer}"

        except Exception as e:
            logger.error(f"调用本地大模型生成回答时出错: {e}")
            return f"未找到相关文档内容。同时，本地大模型回答功能暂时不可用: {str(e)}"
