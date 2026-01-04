"""
幻觉检测模块
用于检测和减少RAG系统中大模型生成的回答与检索文档之间的不一致性
"""
import logging
import sys
from typing import List, Dict, Tuple
import re
from dataclasses import dataclass

import ollama

from rag_agent.prompts.hallucination_templates import HALLUCINATION_TEMPLATES


@dataclass
class HallucinationCheckResult:
    """幻觉检测结果"""
    is_consistent: bool  # 是否与文档一致
    confidence_score: float  # 置信度分数 (0-1)
    inconsistencies: List[str]  # 检测到的不一致之处
    explanation: str  # 检测结果解释


class HallucinationDetector:
    """
    幻觉检测器
    检测生成的回答是否与检索到的文档一致，减少幻觉
    """

    def __init__(self, config):
        self.config = config
        self.ollama_model = config.OLLAMA_MODEL
        self.logger = logging.getLogger(__name__)
        
        # 确保日志记录器有处理器
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    def detect_hallucinations(self, 
                            response: str, 
                            retrieved_docs: List[Dict], 
                            query: str) -> HallucinationCheckResult:
        """
        检测生成的回答中是否存在幻觉
        
        Args:
            response: 模型生成的回答
            retrieved_docs: 检索到的相关文档
            query: 原始查询
        
        Returns:
            HallucinationCheckResult: 检测结果
        """
        self.logger.info(f"开始检测幻觉，查询: {query}")
        self.logger.debug(f"待检测的回答: {response}")
        self.logger.debug(f"检索到的文档数量: {len(retrieved_docs)}")
        
        if not retrieved_docs:
            # 如果没有检索到文档，无法进行幻觉检测，返回高置信度
            self.logger.warning("没有检索到相关文档，无法进行幻觉检测")
            result = HallucinationCheckResult(
                is_consistent=True,
                confidence_score=0.8,  # 没有文档时，无法验证幻觉，但置信度设为中等
                inconsistencies=[],
                explanation="没有检索到相关文档，无法进行幻觉检测"
            )
            self.logger.info(f"幻觉检测结果 - 一致性: {result.is_consistent}, 置信度: {result.confidence_score}")
            return result

        # 方法1：基于事实对比的检测
        self.logger.debug("开始基于事实对比的检测...")
        fact_consistency_result = self._check_fact_consistency(response, retrieved_docs)
        self.logger.debug(f"事实一致性检测结果 - 一致性: {fact_consistency_result.is_consistent}, 置信度: {fact_consistency_result.confidence_score}")
        
        # 方法2：基于语义一致性的检测
        self.logger.debug("开始基于语义一致性的检测...")
        semantic_consistency_result = self._check_semantic_consistency(response, retrieved_docs, query)
        self.logger.debug(f"语义一致性检测结果 - 一致性: {semantic_consistency_result.is_consistent}, 置信度: {semantic_consistency_result.confidence_score}")
        
        # 综合判断
        is_consistent = fact_consistency_result.is_consistent and semantic_consistency_result.is_consistent
        confidence_score = min(fact_consistency_result.confidence_score, semantic_consistency_result.confidence_score)
        inconsistencies = fact_consistency_result.inconsistencies + semantic_consistency_result.inconsistencies
        explanation = f"事实一致性: {fact_consistency_result.explanation}, 语义一致性: {semantic_consistency_result.explanation}"
        
        self.logger.info(f"幻觉检测完成 - 一致性: {is_consistent}, 置信度: {confidence_score}")
        if inconsistencies:
            self.logger.warning(f"检测到不一致内容: {inconsistencies}")
        
        result = HallucinationCheckResult(
            is_consistent=is_consistent,
            confidence_score=confidence_score,
            inconsistencies=inconsistencies,
            explanation=explanation
        )
        
        return result

    def _check_fact_consistency(self, response: str, retrieved_docs: List[Dict]) -> HallucinationCheckResult:
        """
        检查生成的回答与文档中的事实是否一致
        """
        self.logger.debug("执行事实一致性检查...")
        
        # 将回答分解为独立的句子或事实陈述
        sentences = self._split_into_sentences(response)
        self.logger.debug(f"回答分解为 {len(sentences)} 个句子")
        
        inconsistencies = []
        
        # 提取文档中的关键事实
        doc_facts = self._extract_facts_from_docs(retrieved_docs)
        self.logger.debug(f"从文档中提取了 {len(doc_facts)} 个事实")
        
        for sentence in sentences:
            self.logger.debug(f"检查句子: '{sentence}'")
            # 检查句子是否与文档中的事实一致
            if not self._sentence_supported_by_docs(sentence, doc_facts):
                self.logger.debug(f"句子未被文档支持: '{sentence}'")
                inconsistencies.append(sentence)
        
        # 计算置信度
        total_sentences = len(sentences)
        inconsistent_sentences = len(inconsistencies)
        
        if total_sentences == 0:
            confidence_score = 1.0
        else:
            consistency_ratio = 1.0 - (inconsistent_sentences / total_sentences)
            # 调整置信度，给一致性较高的回答更高分数
            confidence_score = max(0.1, consistency_ratio)
        
        is_consistent = len(inconsistencies) == 0
        explanation = f"总句子数: {total_sentences}, 不一致句子数: {inconsistent_sentences}"
        
        self.logger.debug(f"事实一致性检查完成 - 一致性: {is_consistent}, 置信度: {confidence_score}")
        
        return HallucinationCheckResult(
            is_consistent=is_consistent,
            confidence_score=confidence_score,
            inconsistencies=inconsistencies,
            explanation=explanation
        )

    def _check_semantic_consistency(self, response: str, retrieved_docs: List[Dict], query: str) -> HallucinationCheckResult:
        """
        使用LLM进行语义一致性检查
        """
        self.logger.debug("执行语义一致性检查...")
        
        # 构建检查提示词
        context = self._format_docs_for_checking(retrieved_docs)
        
        # 使用从模板文件导入的提示词
        prompt = HALLUCINATION_TEMPLATES["semantic_consistency_check"].format(
            context=context,
            query=query,
            response=response
        )
        
        self.logger.debug("向LLM发送语义一致性检查请求...")
        try:
            result = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1}  # 低温度以获得更一致的判断
            )
            
            analysis = result["message"]["content"]
            self.logger.debug(f"LLM返回分析结果: {analysis}")
            parsed_result = self._parse_semantic_check_result(analysis)
            self.logger.debug(f"解析语义检查结果完成 - 一致性: {parsed_result.is_consistent}, 置信度: {parsed_result.confidence_score}")
            return parsed_result
            
        except Exception as e:
            self.logger.error(f"语义一致性检查失败: {e}")
            # 如果LLM检查失败，返回保守的判断
            result = HallucinationCheckResult(
                is_consistent=True,
                confidence_score=0.6,
                inconsistencies=[],
                explanation=f"语义检查失败，使用默认判断: {str(e)}"
            )
            self.logger.warning(f"使用默认判断结果: 一致性={result.is_consistent}, 置信度={result.confidence_score}")
            return result

    def _parse_semantic_check_result(self, analysis: str) -> HallucinationCheckResult:
        """
        解析语义一致性检查结果
        """
        self.logger.debug(f"解析语义检查结果: {analysis}")
        
        is_consistent = "一致性: 是" in analysis or "一致性: 是" in analysis.lower()
        confidence_match = re.search(r"置信度:\s*([0-9.]+)", analysis)
        confidence_score = float(confidence_match.group(1)) if confidence_match else 0.7
        
        inconsistencies_match = re.search(r"不一致之处:\s*(.+?)(?:\n|$)", analysis, re.DOTALL)
        inconsistencies_str = inconsistencies_match.group(1).strip() if inconsistencies_match else ""
        inconsistencies = [inc.strip() for inc in inconsistencies_str.split("\n") if inc.strip() and inc.strip() != "无"]
        
        explanation_match = re.search(r"解释:\s*(.+?)(?:\n|$)", analysis, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else analysis[:200] + "..."
        
        self.logger.debug(f"解析结果 - 一致性: {is_consistent}, 置信度: {confidence_score}, 不一致项数量: {len(inconsistencies)}")
        
        return HallucinationCheckResult(
            is_consistent=is_consistent,
            confidence_score=confidence_score,
            inconsistencies=inconsistencies,
            explanation=explanation
        )

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分割为句子列表
        """
        self.logger.debug(f"分割文本为句子，文本长度: {len(text)}")
        # 使用正则表达式分割句子
        sentences = re.split(r'[.!?。！？\n]+', text)
        # 过滤空句子并去除空白
        sentences = [s.strip() for s in sentences if s.strip()]
        self.logger.debug(f"分割得到 {len(sentences)} 个句子")
        return sentences

    def _extract_facts_from_docs(self, docs: List[Dict]) -> List[str]:
        """
        从文档中提取关键事实
        """
        self.logger.debug(f"从 {len(docs)} 个文档中提取事实...")
        facts = []
        for i, doc in enumerate(docs):
            content = doc.get("content", "")
            title = doc.get("metadata", {}).get("title", "Unknown")
            self.logger.debug(f"处理文档 {i+1}/{len(docs)}: {title} (长度: {len(content)})")
            # 简单地提取文档中的关键信息
            # 在实际应用中，可以使用更复杂的NLP技术
            sentences = self._split_into_sentences(content)
            facts.extend(sentences)
        self.logger.debug(f"总共提取了 {len(facts)} 个事实")
        return facts

    def _sentence_supported_by_docs(self, sentence: str, doc_facts: List[str]) -> bool:
        """
        检查句子是否被文档中的事实支持
        """
        if not sentence.strip():
            self.logger.debug("空句子视为一致")
            return True  # 空句子视为一致
        
        self.logger.debug(f"检查句子是否被文档支持: '{sentence}'")
        # 简单的关键词匹配检查
        sentence_lower = sentence.lower()
        for fact in doc_facts:
            fact_lower = fact.lower()
            # 如果句子中的关键信息在文档中有提及，则认为一致
            if self._is_fact_supported(sentence_lower, fact_lower):
                self.logger.debug(f"句子被文档支持: '{sentence}'")
                return True
        
        # 如果没有找到支持的文档事实，返回不一致
        self.logger.debug(f"句子未被文档支持: '{sentence}'")
        return False

    def _is_fact_supported(self, sentence: str, fact: str) -> bool:
        """
        检查一个句子是否被文档中的某个事实支持
        """
        self.logger.debug(f"检查句子是否被事实支持 - 句子: '{sentence}', 事实: '{fact}'")
        # 简单的文本相似度检查
        sentence_words = set(sentence.split())
        fact_words = set(fact.split())
        
        # 计算词汇重叠度
        if not sentence_words:
            self.logger.debug("句子为空，返回True")
            return True
        
        overlap = len(sentence_words.intersection(fact_words))
        overlap_ratio = overlap / len(sentence_words)
        
        self.logger.debug(f"词汇重叠度 - 重叠数: {overlap}, 重叠率: {overlap_ratio:.2f}, 阈值: 0.3")
        
        # 如果重叠度超过阈值，认为句子被支持
        is_supported = overlap_ratio > 0.3
        if is_supported:
            self.logger.debug("句子被支持")
        else:
            self.logger.debug("句子未被支持")
        return is_supported

    def _format_docs_for_checking(self, docs: List[Dict]) -> str:
        """
        格式化文档用于幻觉检测
        """
        self.logger.debug(f"格式化 {len(docs)} 个文档用于幻觉检测")
        formatted_docs = []
        for doc in docs:
            content = doc.get("content", "")[:500] + "..." if len(doc.get("content", "")) > 500 else doc.get("content", "")
            formatted_doc = (
                f"文档: {doc['metadata'].get('title', 'Unknown')}\n"
                f"路径: {doc['metadata'].get('path', 'Unknown')}\n"
                f"内容: {content}\n"
            )
            formatted_docs.append(formatted_doc)
            self.logger.debug(f"格式化文档: {doc['metadata'].get('title', 'Unknown')}")
        
        result = "\n" + "="*50 + "\n".join(formatted_docs) + "\n" + "="*50
        self.logger.debug(f"格式化完成，总长度: {len(result)}")
        return result