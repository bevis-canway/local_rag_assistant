import logging
import os
from typing import Dict, List, Tuple
import numpy as np

import ollama

from .prompts.prompt_templates import RAG_PROMPT_TEMPLATES
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    检索器
    负责根据用户查询从向量库中检索相关信息
    参考aidev项目中的检索逻辑实现
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        config = None,
        knowledge_base_name: str = "default",  # 添加知识库名称参数
    ):
        self.config = config
        self.vector_store = vector_store
        self.top_k = top_k
        # 相似度阈值，低于此值的文档将被忽略
        self.similarity_threshold = similarity_threshold
        self.knowledge_base_name = knowledge_base_name  # 添加知识库名称

    def retrieve(self, query: str, knowledge_base_filter: str = None) -> List[Dict]:
        """
        检索与查询最相关的文档片段
        参考aidev项目中的知识库检索实现
        """
        try:
            results = self.vector_store.search(query, self.top_k)
            logger.info(f"检索到 {len(results)} 个相关文档片段")
            
            # 如果指定了知识库过滤器，则只返回指定知识库的文档
            if knowledge_base_filter:
                results = [result for result in results 
                          if result.get('metadata', {}).get('knowledge_base') == knowledge_base_filter]
                logger.info(f"过滤后，知识库 {knowledge_base_filter} 有 {len(results)} 个相关文档片段")
            
            return results
        except Exception as e:
            logger.error(f"检索过程出错: {e}")
            return []

    def retrieve_and_filter_by_similarity(self, query: str, knowledge_base_filter: str = None) -> Tuple[List[Dict], bool]:
        """
        检索并根据相似度过滤文档
        返回过滤后的文档列表和是否找到相关文档的标志
        """
        results = self.retrieve(query, knowledge_base_filter)

        if not results:
            logger.info("未检索到任何文档")
            return [], False

        # 改进的过滤逻辑：使用动态阈值和相对相似度
        similarities = [result.get("similarity", 0) for result in results]
        max_similarity = max(similarities) if similarities else 0
        min_similarity = min(similarities) if similarities else 0
        
        logger.debug(f"检索结果相似度范围: min={min_similarity:.3f}, max={max_similarity:.3f}, avg={np.mean(similarities):.3f}")
        
        # 如果最高相似度很低，说明整体不相关
        if max_similarity < 0.1:
            logger.info(f"最高相似度 {max_similarity:.3f} 过低，认为无相关文档")
            return [], False

        # 使用自适应阈值：取top_k结果中的平均值和固定阈值的较大者
        adaptive_threshold = max(self.similarity_threshold, np.mean(similarities) * 0.5)
        
        # 过滤掉相似度低于阈值的文档
        filtered_results = [
            result
            for result in results
            if result.get("similarity", 0) >= adaptive_threshold
        ]

        # 如果没有通过自适应阈值过滤的结果，尝试使用更宽松的阈值
        if not filtered_results:
            # 尝试使用最低相似度的30%作为阈值，但不低于0.1
            fallback_threshold = max(min(similarities) * 0.3, 0.1)
            filtered_results = [
                result
                for result in results
                if result.get("similarity", 0) >= fallback_threshold
            ]
            logger.info(f"使用自适应阈值未找到结果，尝试回退阈值 {fallback_threshold:.3f}")
        
        # 检查是否有足够相关的文档
        has_relevant_docs = len(filtered_results) > 0

        logger.info(f"原始检索到 {len(results)} 个文档，过滤后剩余 {len(filtered_results)} 个相关文档")
        if has_relevant_docs:
            logger.info(f"找到 {len(filtered_results)} 个满足相似度阈值的相关文档片段")
            for i, result in enumerate(filtered_results):
                logger.debug(f"  文档 {i+1}: 相似度 {result.get('similarity', 0):.3f}, 标题: {result['metadata'].get('title', 'Unknown')}")
        else:
            logger.info(f"未找到满足相似度阈值的相关文档")

        return filtered_results, has_relevant_docs

    def format_results(self, filtered_results: List[Dict]) -> str:
        """
        格式化检索结果
        """
        if not filtered_results:
            return ""

        formatted_results = []
        for result in filtered_results:
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

    def retrieve_and_format(self, query: str, knowledge_base_filter: str = None) -> str:
        """
        检索并格式化返回结果
        如果未找到相关文档，或相关性不足，则使用本地大模型回答
        """
        filtered_results, has_relevant_docs = self.retrieve_and_filter_by_similarity(query, knowledge_base_filter)

        # 如果没有找到相关文档，使用本地大模型回答
        if not has_relevant_docs:
            logger.info(
                f"未找到满足相似度阈值的相关文档，使用本地大模型回答..."
            )
            return self._generate_response_with_llm(query)

        return self.format_results(filtered_results)

    def _generate_response_with_llm(self, query: str) -> str:
        """
        使用本地大模型生成回答
        """
        try:
            # 从环境变量获取模型名称，默认使用配置的模型
            model_name = os.getenv("OLLAMA_MODEL", "deepseek-coder")

            # 使用预定义的提示词模板
            prompt = RAG_PROMPT_TEMPLATES["no_document_found"].format(query=query)

            # 调用 Ollama 模型
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options=self.config.get_generation_options() if hasattr(self, 'config') else {"temperature": 0.2}
            )

            llm_answer = response["message"]["content"]
            logger.info("已使用本地大模型生成回答")

            return f"未在本地知识库中找到相关内容。\n\n智能助手回答：\n{llm_answer}"

        except Exception as e:
            logger.error(f"调用本地大模型生成回答时出错: {e}")
            return f"未找到相关文档内容。同时，本地大模型回答功能暂时不可用: {str(e)}"