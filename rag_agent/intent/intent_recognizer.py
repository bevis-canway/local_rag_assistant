"""
高级意图识别系统
为小魔仙RAG智能体提供基于业界最佳实践的意图识别功能
实现查询分类、历史对话感知的查询重写、结构化意图解析等功能
"""
import json
import logging
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import ollama

from ..config import Config
from ..prompts.prompt_templates import RAG_PROMPT_TEMPLATES
from ..vector_store import VectorStore

# 配置日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 如果没有处理器，添加控制台处理器
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 为了确保日志输出，同时提供一个直接打印的备用方式
def debug_print(message):
    """备用调试打印函数"""
    import datetime
    print(f"[DEBUG] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


@dataclass
class IntentResult:
    """意图识别结果"""
    intent_type: str  # 意图类型
    confidence: float  # 置信度
    entity: str = ""  # 实体
    aspect: str = ""  # 方面
    original_query: str = ""  # 原始查询
    rewritten_query: str = ""  # 重写后的查询
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
        logger.info(f"开始识别意图，查询: {query}")
        debug_print(f"开始识别意图，查询: {query}")
        
        # 1. 检查是否为历史对话查询
        is_history = self._is_history_query(query)
        logger.debug(f"历史对话查询检查结果: {is_history}")
        debug_print(f"历史对话查询检查结果: {is_history}")
        
        if is_history:
            result = IntentResult(
                intent_type="history_query",
                confidence=1.0,
                original_query=query,
                rewritten_query=query,
                extracted_info={"query_text": query}
            )
            logger.info(f"识别为历史查询，结果: {result.intent_type}")
            debug_print(f"识别为历史查询，结果: {result.intent_type}")
            return result

        # 2. 进行查询分类
        logger.debug("开始查询分类")
        debug_print("开始查询分类")
        classification_result = self._classify_query(query, chat_history)
        logger.debug(f"查询分类结果: {classification_result}")
        debug_print(f"查询分类结果: {classification_result}")

        # 3. 如果是知识查询，进行历史感知的查询重写
        if classification_result["intent"] == "knowledge_query":
            logger.debug("进行查询重写")
            debug_print("进行查询重写")
            rewritten_query = self._rewrite_query_with_history(query, chat_history)
            classification_result["original_query"] = query
            classification_result["rewritten_query"] = rewritten_query
            logger.debug(f"查询重写完成，原查询: {query}, 重写后: {rewritten_query}")
            debug_print(f"查询重写完成，原查询: {query}, 重写后: {rewritten_query}")
        else:
            classification_result["original_query"] = query
            classification_result["rewritten_query"] = query
            logger.debug(f"非知识查询，无需重写，意图: {classification_result['intent']}")
            debug_print(f"非知识查询，无需重写，意图: {classification_result['intent']}")

        # 4. 进行结构化意图解析
        logger.debug("开始结构化意图解析")
        debug_print("开始结构化意图解析")
        structured_result = self._parse_structured_intent(
            classification_result["rewritten_query"],
            classification_result["intent"]
        )
        logger.debug(f"结构化意图解析结果: {structured_result}")
        debug_print(f"结构化意图解析结果: {structured_result}")

        # 5. 检查置信度并决定是否需要澄清
        # 对于知识查询，降低澄清阈值，因为这类查询通常不需要澄清
        base_confidence = structured_result.get("confidence", classification_result.get("confidence", 0.5))
        
        # 如果是知识查询，置信度阈值降低到0.6，因为这类查询通常比较明确
        if classification_result["intent"] == "knowledge_query":
            clarification_threshold = 0.3
        else:
            clarification_threshold = 0.4
        
        logger.debug(f"置信度: {base_confidence}, 澄清阈值: {clarification_threshold}")
        debug_print(f"置信度: {base_confidence}, 澄清阈值: {clarification_threshold}")

        if base_confidence < clarification_threshold and classification_result["intent"] != "chitchat":
            result = IntentResult(
                intent_type="ambiguous",
                confidence=base_confidence,
                entity=structured_result.get("entity", ""),
                aspect=structured_result.get("aspect", ""),
                original_query=classification_result["original_query"],
                rewritten_query=classification_result["rewritten_query"],
                extracted_info=structured_result
            )
            logger.info(f"识别为模糊意图，需要澄清，置信度: {base_confidence}")
            debug_print(f"识别为模糊意图，需要澄清，置信度: {base_confidence}")
            return result

        result = IntentResult(
            intent_type=classification_result["intent"],
            confidence=base_confidence,
            entity=structured_result.get("entity", ""),
            aspect=structured_result.get("aspect", ""),
            original_query=classification_result["original_query"],
            rewritten_query=classification_result["rewritten_query"],
            extracted_info=structured_result
        )
        logger.info(f"意图识别完成，类型: {result.intent_type}, 置信度: {result.confidence}")
        debug_print(f"意图识别完成，类型: {result.intent_type}, 置信度: {result.confidence}")
        return result

    def _is_history_query(self, query: str) -> bool:
        """
        检查是否为历史对话查询
        """
        logger.debug(f"检查是否为历史对话查询: {query}")
        debug_print(f"检查是否为历史对话查询: {query}")
        
        history_keywords = [
            "前面", "之前", "刚才", "上一个", "第一个", "历史", "之前问", "前面问",
            "刚才问", "上个", "之前的", "前面的", "刚才的", "我问", "我的问题",
            "前面说", "刚才说", "之前说", "对话历史", "我们刚才", "我们之前",
            "刚才说了什么", "之前说了什么", "前面说了什么"
        ]

        query_lower = query.lower()
        is_history = any(keyword in query_lower for keyword in history_keywords)
        logger.debug(f"历史查询检查结果: {is_history}")
        debug_print(f"历史查询检查结果: {is_history}")
        return is_history

    def _classify_query(self, query: str, chat_history: List[Dict] = None) -> Dict:
        """
        查询分类：将用户输入分为四类
        """
        # 先进行快速关键词匹配，提高准确性
        quick_classification = self._quick_classify_by_keywords(query)
        if quick_classification:
            return quick_classification

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
            debug_print(f"查询分类失败: {e}")
            # 默认返回知识查询
            return {
                "intent": "knowledge_query",
                "confidence": 0.6,
                "reason": "分类失败，使用默认分类"
            }

    def _quick_classify_by_keywords(self, query: str) -> Optional[Dict]:
        """
        快速关键词匹配分类
        """
        logger.debug(f"快速关键词匹配分类，查询: {query}")
        debug_print(f"快速关键词匹配分类，查询: {query}")
        
        query_lower = query.lower()

        # 闲聊类关键词
        chitchat_keywords = ["你好", "您好", "谢谢", "再见", "拜拜", "hi", "hello", "早上好", "晚上好", "中午好"]
        if any(keyword in query_lower for keyword in chitchat_keywords):
            result = {
                "intent": "chitchat",
                "confidence": 0.9,
                "reason": "匹配到闲聊关键词"
            }
            logger.debug(f"匹配到闲聊关键词，结果: {result}")
            debug_print(f"匹配到闲聊关键词，结果: {result}")
            return result

        # 工具请求关键词
        tool_keywords = ["帮我", "计算", "转换", "生成", "翻译", "总结", "分析", "提取"]
        if any(keyword in query_lower for keyword in tool_keywords):
            result = {
                "intent": "tool_request",
                "confidence": 0.8,
                "reason": "匹配到工具请求关键词"
            }
            logger.debug(f"匹配到工具请求关键词，结果: {result}")
            debug_print(f"匹配到工具请求关键词，结果: {result}")
            return result

        # 知识查询关键词
        knowledge_keywords = ["什么是", "怎么", "如何", "为什么", "什么", "哪个", "哪个是", "介绍", "解释", "说明",
                              "定义", "概念", "了解", "查询", "搜索", "查找", "人才偏好", "公司", "企业"]
        if any(keyword in query_lower for keyword in knowledge_keywords):
            result = {
                "intent": "knowledge_query",
                "confidence": 0.85,
                "reason": "匹配到知识查询关键词"
            }
            logger.debug(f"匹配到知识查询关键词，结果: {result}")
            debug_print(f"匹配到知识查询关键词，结果: {result}")
            return result

        logger.debug("未匹配到任何关键词")
        debug_print("未匹配到任何关键词")
        return None

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
2. knowledge_query（知识查询）：询问知识库中的信息，如"什么是XXX？"、"如何XXX？"、"XXX是什么？"
3. tool_request（工具请求）：请求执行某种操作或使用工具，如"帮我XXX"、"计算XXX"、"生成XXX"
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
            logger.info(f"分类结果: {result}")
            debug_print(f"分类结果: {result}")
            return result
        except json.JSONDecodeError:
            # 如果JSON解析失败，尝试从文本中提取信息
            logger.error(f"JSON解析失败，尝试从文本中提取信息: {response}")
            debug_print(f"JSON解析失败，尝试从文本中提取信息: {response}")
            response_lower = response.lower()

            if "chitchat" in response_lower or any(
                    greeting in original_query for greeting in ["你好", "您好", "谢谢", "再见", "拜拜", "hello", "hi"]):
                intent = "chitchat"
                confidence = 0.8
            elif "tool_request" in response_lower:
                intent = "tool_request"
                confidence = 0.75
            elif "ambiguous" in response_lower:
                intent = "ambiguous"
                confidence = 0.4
            else:
                intent = "knowledge_query"  # 默认为知识查询
                confidence = 0.7

            return {
                "intent": intent,
                "confidence": confidence,
                "reason": "解析失败，使用规则匹配"
            }

    def _rewrite_query_with_history(self, query: str, chat_history: List[Dict] = None) -> str:
        """
        历史对话感知的查询重写
        将用户查询重写为独立、无指代、语义完整的问题
        """
        if not chat_history:
            return query

        # 检查查询是否包含指代词
        reference_words = ["它", "这个", "那个", "这些", "那些", "其", "该"]
        contains_reference = any(ref in query for ref in reference_words)

        if not contains_reference:
            # 如果查询不包含指代词，则不需要重写
            return query

        # 构建重写提示词
        prompt = self._build_query_rewrite_prompt(query, chat_history)

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )

            rewritten_query = response["message"]["content"].strip()
            logger.info(f"重写结果: {rewritten_query}")
            debug_print(f"重写结果: {rewritten_query}")

            # 如果返回的结果为空或与原查询相同，返回原查询
            if not rewritten_query or rewritten_query.strip() == query or "无法重写" in rewritten_query or rewritten_query.strip() == "无":
                return query

            return rewritten_query
        except Exception as e:
            logger.error(f"查询重写失败: {e}")
            debug_print(f"查询重写失败: {e}")
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
            debug_print(f"结构化意图解析失败: {e}")
            # 对于知识查询，即使解析失败也给予较高置信度
            if intent_type == "knowledge_query":
                return {
                    "intent": intent_type,
                    "entity": self._extract_entity_simple(query),
                    "aspect": self._extract_aspect_simple(query),
                    "confidence": 0.75,
                    "query": query,
                    "error": str(e)
                }
            else:
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
            logger.info(f"结构化结果: {result}")
            debug_print(f"结构化结果: {result}")
            return result
        except json.JSONDecodeError:
            # 如果JSON解析失败，返回基于规则的解析结果
            # 避免将原始响应内容作为结构化结果的一部分，以防止显示JSON内容
            return {
                "intent": "knowledge_query",
                "entity": self._extract_entity_simple(response),
                "aspect": self._extract_aspect_simple(response),
                "confidence": 0.6,
                "error": "解析失败"
            }

    def _extract_entity_simple(self, query: str) -> str:
        """
        简单的实体提取
        """
        logger.debug(f"简单实体提取，查询: {query}")
        debug_print(f"简单实体提取，查询: {query}")
        # 这里可以实现简单的实体提取逻辑
        # 暂时返回空字符串，实际应用中可以使用NLP技术进行实体识别
        return ""

    def _extract_aspect_simple(self, query: str) -> str:
        """
        简单的方面提取
        """
        logger.debug(f"简单方面提取，查询: {query}")
        debug_print(f"简单方面提取，查询: {query}")
        # 简单地从查询中提取可能的方面
        aspect_keywords = ["什么", "如何", "怎么", "为什么", "哪个", "哪个是", "怎样", "多大", "多少", "哪里", "何时"]
        for keyword in aspect_keywords:
            if keyword in query:
                # 返回关键词后面的部分
                parts = query.split(keyword)
                if len(parts) > 1:
                    aspect = f"{keyword}{parts[1]}"
                    logger.debug(f"提取到方面: {aspect}")
                    debug_print(f"提取到方面: {aspect}")
                    return aspect
        logger.debug(f"未提取到特定方面，返回原查询: {query}")
        debug_print(f"未提取到特定方面，返回原查询: {query}")
        return query

    def get_clarification_response(self, intent_result: IntentResult) -> str:
        """
        获取澄清响应
        当置信度较低时，返回澄清消息
        """
        logger.debug(f"获取澄清响应，置信度: {intent_result.confidence}, 意图类型: {intent_result.intent_type}")
        debug_print(f"获取澄清响应，置信度: {intent_result.confidence}, 意图类型: {intent_result.intent_type}")
        
        if intent_result.confidence >= 0.5 or intent_result.intent_type == "chitchat":
            logger.debug("置信度足够高或为闲聊类型，无需澄清")
            debug_print("置信度足够高或为闲聊类型，无需澄清")
            return None  # 不需要澄清

        entity = intent_result.entity or "相关信息"
        aspect = intent_result.aspect or "方面"
        
        clarification = f"抱歉，您是指关于 {entity} 的 {aspect} 吗？请提供更具体的信息，这样我可以更好地帮助您。"
        logger.debug(f"生成澄清消息: {clarification}")
        debug_print(f"生成澄清消息: {clarification}")
        
        return clarification
