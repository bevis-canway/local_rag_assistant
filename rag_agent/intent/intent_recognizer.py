"""
高级意图识别系统
为小魔仙RAG智能体提供基于业界最佳实践的意图识别功能
实现查询分类、历史对话感知的查询重写、结构化意图解析等功能
"""
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import ollama

from ..config import Config
from ..prompts.prompt_templates import RAG_PROMPT_TEMPLATES
from ..vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    """意图识别结果"""
    intent_type: str  # 意图类型
    confidence: float  # 置信度
    entity: str = ""  # 实体
    aspect: str = ""  # 方面
    extracted_info: Dict = None  # 提取的额外信息


class IntentRecognizer:
    """
    高级意图识别器
    实现查询分类、历史对话感知的查询重写、结构化意图解析等功能
    """
    
    def __init__(self, config: Config, vector_store: VectorStore = None):
        self.config = config
        self.vector_store = vector_store
        self.ollama_model = config.OLLAMA_MODEL
        
        # 定义意图类型
        self.intent_types = {
            "chitchat": "闲聊类查询",
            "knowledge_query": "知识库查询",
            "tool_request": "工具请求",
            "ambiguous": "模糊需澄清",
            "history_query": "历史对话查询"
        }

    def recognize_intent(self, query: str, chat_history: List[Dict] = None) -> IntentResult:
        """
        识别用户查询的意图
        
        Args:
            query: 用户查询
            chat_history: 对话历史
            
        Returns:
            IntentResult: 意图识别结果
        """
        # 1. 检查是否为历史对话查询
        if self._is_history_query(query):
            return IntentResult("history_query", 1.0, extracted_info={"query_text": query})
        
        # 2. 进行查询分类
        classification_result = self._classify_query(query, chat_history)
        
        # 3. 如果是知识查询，进行历史感知的查询重写
        if classification_result["intent"] == "knowledge_query":
            rewritten_query = self._rewrite_query_with_history(query, chat_history)
            classification_result["original_query"] = query
            classification_result["rewritten_query"] = rewritten_query
        else:
            classification_result["original_query"] = query
            classification_result["rewritten_query"] = query
        
        # 4. 进行结构化意图解析
        structured_result = self._parse_structured_intent(
            classification_result["rewritten_query"], 
            classification_result["intent"]
        )
        
        # 5. 检查置信度并决定是否需要澄清
        if structured_result["confidence"] < 0.7:
            return IntentResult(
                "ambiguous", 
                structured_result["confidence"],
                entity=structured_result.get("entity", ""),
                aspect=structured_result.get("aspect", ""),
                extracted_info=structured_result
            )
        
        return IntentResult(
            intent_type=structured_result["intent"],
            confidence=structured_result["confidence"],
            entity=structured_result.get("entity", ""),
            aspect=structured_result.get("aspect", ""),
            extracted_info=structured_result
        )

    def _is_history_query(self, query: str) -> bool:
        """
        检查是否为历史对话查询
        """
        history_keywords = [
            "前面", "之前", "刚才", "上一个", "第一个", "历史", "之前问", "前面问", 
            "刚才问", "上个", "之前的", "前面的", "刚才的", "我问", "我的问题",
            "前面说", "刚才说", "之前说", "对话历史", "我们刚才", "我们之前", 
            "刚才说了什么", "之前说了什么", "前面说了什么"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in history_keywords)

    def _classify_query(self, query: str, chat_history: List[Dict] = None) -> Dict:
        """
        查询分类：将用户输入分为四类
        """
        # 构建分类提示词
        prompt = self._build_classification_prompt(query, chat_history)
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            
            result = response["message"]["content"].strip()
            
            # 解析分类结果
            return self._parse_classification_result(result, query)
            
        except Exception as e:
            logger.error(f"查询分类失败: {e}")
            # 默认返回知识查询
            return {
                "intent": "knowledge_query",
                "confidence": 0.5,
                "reason": "分类失败，使用默认分类"
            }

    def _build_classification_prompt(self, query: str, chat_history: List[Dict] = None) -> str:
        """
        构建查询分类的提示词
        """
        if chat_history:
            chat_history_str = "\n".join([
                f"用户: {item['query']}\n助手: {item['response']}" 
                for item in chat_history[-3:]  # 只取最近3轮对话
            ])
        else:
            chat_history_str = "无历史对话"
        
        prompt = f"""
请对以下用户输入进行分类：

分类类型：
1. chitchat（闲聊）：如问候、感谢、告别等
2. knowledge_query（知识查询）：询问知识库中的信息
3. tool_request（工具请求）：请求执行某种操作或使用工具
4. ambiguous（模糊需澄清）：意图不明确，需要澄清

用户输入：{query}

历史对话：
{chat_history_str}

请按照以下JSON格式返回分类结果：
{{
  "intent": "chitchat|knowledge_query|tool_request|ambiguous",
  "confidence": 0.0-1.0,
  "reason": "分类理由"
}}

只返回JSON格式的内容，不要返回其他内容。
"""
        
        return prompt

    def _parse_classification_result(self, response: str, original_query: str) -> Dict:
        """
        解析分类结果
        """
        try:
            # 尝试解析JSON响应
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            # 如果JSON解析失败，尝试从文本中提取信息
            response_lower = response.lower()
            
            if "chitchat" in response_lower or any(greeting in original_query for greeting in ["你好", "谢谢", "再见", "hello", "hi"]):
                intent = "chitchat"
            elif "tool_request" in response_lower:
                intent = "tool_request"
            elif "ambiguous" in response_lower:
                intent = "ambiguous"
            else:
                intent = "knowledge_query"  # 默认为知识查询
            
            return {
                "intent": intent,
                "confidence": 0.6,  # 默认置信度
                "reason": "解析失败，使用规则匹配"
            }

    def _rewrite_query_with_history(self, query: str, chat_history: List[Dict] = None) -> str:
        """
        历史对话感知的查询重写
        将用户查询重写为独立、无指代、语义完整的问题
        """
        if not chat_history:
            return query
        
        # 构建重写提示词
        prompt = self._build_query_rewrite_prompt(query, chat_history)
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            
            rewritten_query = response["message"]["content"].strip()
            
            # 如果返回的结果为空或与原查询相同，返回原查询
            if not rewritten_query or rewritten_query.strip() == query or "无法重写" in rewritten_query:
                return query
            
            return rewritten_query
        except Exception as e:
            logger.error(f"查询重写失败: {e}")
            return query

    def _build_query_rewrite_prompt(self, query: str, chat_history: List[Dict] = None) -> str:
        """
        构建查询重写的提示词
        """
        if chat_history:
            chat_history_str = "\n".join([
                f"用户: {item['query']}\n助手: {item['response']}" 
                for item in chat_history[-3:]  # 只取最近3轮对话
            ])
        else:
            chat_history_str = ""
        
        prompt = f"""
你是一个智能对话系统，负责将用户的最新输入重写成一个完全独立的查询。

请参考以下历史对话：
{chat_history_str}

用户最新输入：{query}

请将用户的最新输入重写成一个完全独立的查询，要求：
1. 信息全面，包含所有必要信息
2. 不依赖历史对话信息
3. 无指代，明确提及具体对象
4. 语义完整，可以独立理解

例如：
- 历史：用户问"iPhone 15 有哪些新功能？"，助手回答了A、B、C...
- 新输入："它的电池续航呢？"
- 重写："iPhone 15 的电池续航怎么样？"

请只返回重写后的查询，不要返回其他内容。
"""
        
        return prompt

    def _parse_structured_intent(self, query: str, intent_type: str) -> Dict:
        """
        结构化意图解析
        """
        if intent_type == "chitchat":
            # 闲聊类不需要结构化解析
            return {
                "intent": "chitchat",
                "entity": "",
                "aspect": "",
                "confidence": 0.9,
                "query": query
            }
        
        # 构建结构化解析提示词
        prompt = self._build_structured_parsing_prompt(query)
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            
            result = response["message"]["content"].strip()
            
            # 解析结构化结果
            return self._parse_structured_result(result)
            
        except Exception as e:
            logger.error(f"结构化意图解析失败: {e}")
            return {
                "intent": intent_type,
                "entity": "",
                "aspect": "",
                "confidence": 0.5,
                "query": query,
                "error": str(e)
            }

    def _build_structured_parsing_prompt(self, query: str) -> str:
        """
        构建结构化意图解析的提示词
        """
        prompt = f"""
请解析以下查询的意图结构：

用户查询：{query}

请按照以下JSON格式返回解析结果：
{{
  "intent": "knowledge_query|tool_request|...",
  "entity": "主要实体对象（如产品名称、概念等）",
  "aspect": "查询的方面（如功能、价格、使用方法等）",
  "confidence": 0.0-1.0
}}

例如：
- 查询："iPhone 15 的电池续航怎么样？"
- 结果：{{
  "intent": "knowledge_query",
  "entity": "iPhone 15",
  "aspect": "电池续航",
  "confidence": 0.95
}}

只返回JSON格式的内容，不要返回其他内容。
"""
        
        return prompt

    def _parse_structured_result(self, response: str) -> Dict:
        """
        解析结构化结果
        """
        try:
            # 尝试解析JSON响应
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            # 如果JSON解析失败，返回默认结构
            return {
                "intent": "knowledge_query",
                "entity": "",
                "aspect": "",
                "confidence": 0.5,
                "raw_response": response
            }

    def get_clarification_response(self, intent_result: IntentResult) -> str:
        """
        获取澄清响应
        当置信度较低时，返回澄清消息
        """
        if intent_result.confidence >= 0.7:
            return None  # 不需要澄清
        
        entity = intent_result.entity or "相关信息"
        aspect = intent_result.aspect or "方面"
        
        return f"抱歉，您是指关于 {entity} 的 {aspect} 吗？请提供更具体的信息，这样我可以更好地帮助您。"