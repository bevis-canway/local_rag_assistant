"""
意图识别系统
为小魔仙RAG智能体提供用户意图识别功能
参考bk_ai_dev项目实现
"""
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
    extracted_info: Dict  # 提取的额外信息


class IntentRecognizer:
    """
    意图识别器
    负责识别用户查询的意图类型
    """
    
    def __init__(self, config: Config, vector_store: VectorStore = None):
        self.config = config
        self.vector_store = vector_store
        self.ollama_model = config.OLLAMA_MODEL
        
        # 预定义意图类型
        self.intent_types = {
            "general_query": {
                "keywords": ["什么是", "怎么", "如何", "介绍", "解释", "说明", "定义", "概念"],
                "description": "一般性知识查询"
            },
            "specific_info": {
                "keywords": ["查找", "搜索", "获取", "告诉我", "给我", "具体", "详细"],
                "description": "特定信息查询"
            },
            "clarification": {
                "keywords": ["还是", "或者", "哪个", "哪种", "哪个更好", "推荐"],
                "description": "需要澄清或选择的查询"
            },
            "follow_up": {
                "keywords": ["然后", "接着", "另外", "还有", "进一步", "补充"],
                "description": "跟进或补充性问题"
            },
            "chit_chat": {
                "keywords": ["你好", "谢谢", "再见", "好的", "明白", "不错"],
                "description": "闲聊或礼貌性回复"
            }
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
        # 1. 基于关键词的初步识别
        keyword_intent = self._recognize_by_keywords(query)
        
        # 2. 基于LLM的意图识别（更精确）
        llm_intent = self._recognize_by_llm(query, chat_history)
        
        # 3. 综合判断
        final_intent = self._combine_intent_results(keyword_intent, llm_intent, query)
        
        logger.info(f"意图识别结果 - 查询: {query}, 意图类型: {final_intent.intent_type}, 置信度: {final_intent.confidence}")
        
        return final_intent

    def _recognize_by_keywords(self, query: str) -> Optional[IntentResult]:
        """
        基于关键词匹配的意图识别
        """
        query_lower = query.lower()
        
        best_match = None
        best_score = 0
        
        for intent_type, intent_info in self.intent_types.items():
            score = 0
            for keyword in intent_info["keywords"]:
                if keyword.lower() in query_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = intent_type
        
        if best_match:
            # 根据匹配关键词数量计算置信度
            confidence = min(best_score / len(self.intent_types[best_match]["keywords"]), 1.0)
            return IntentResult(best_match, confidence, {})
        
        return None

    def _recognize_by_llm(self, query: str, chat_history: List[Dict] = None) -> IntentResult:
        """
        基于LLM的意图识别
        """
        try:
            # 构建提示词
            prompt = self._build_intent_recognition_prompt(query, chat_history)
            
            # 调用Ollama模型
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            
            llm_response = response["message"]["content"]
            
            # 解析LLM的响应
            return self._parse_llm_response(llm_response, query)
            
        except Exception as e:
            logger.error(f"LLM意图识别失败: {e}")
            # 如果LLM识别失败，返回默认意图
            return IntentResult("general_query", 0.5, {})

    def _build_intent_recognition_prompt(self, query: str, chat_history: List[Dict] = None) -> str:
        """
        构建意图识别的提示词
        """
        # 定义意图类型描述
        intent_descriptions = "\n".join([
            f"{intent_type}: {info['description']}" 
            for intent_type, info in self.intent_types.items()
        ])
        
        if chat_history:
            chat_history_str = "\n".join([
                f"用户: {item['query']}\n助手: {item['response']}" 
                for item in chat_history[-3:]  # 只取最近3轮对话
            ])
            prompt = RAG_PROMPT_TEMPLATES["intent_recognition_with_history"].format(
                query=query,
                chat_history=chat_history_str,
                intent_descriptions=intent_descriptions
            )
        else:
            prompt = RAG_PROMPT_TEMPLATES["intent_recognition"].format(
                query=query,
                intent_descriptions=intent_descriptions
            )
        
        return prompt

    def _parse_llm_response(self, response: str, query: str) -> IntentResult:
        """
        解析LLM的意图识别响应
        """
        # 尝试从响应中提取意图类型
        response_lower = response.lower()
        
        # 寻找意图类型标识
        for intent_type in self.intent_types.keys():
            if intent_type in response_lower:
                # 提取置信度（如果有的话）
                confidence = 0.8  # 默认高置信度
                return IntentResult(intent_type, confidence, {})
        
        # 如果没有明确匹配，使用默认意图
        return IntentResult("general_query", 0.6, {})

    def _combine_intent_results(self, keyword_result: Optional[IntentResult], 
                              llm_result: IntentResult, query: str) -> IntentResult:
        """
        综合关键词识别和LLM识别的结果
        """
        if keyword_result is None:
            return llm_result
        
        # 如果关键词识别置信度较高，优先使用关键词结果
        if keyword_result.confidence > 0.7:
            return keyword_result
        
        # 否则使用LLM结果（通常更准确）
        return IntentResult(
            intent_type=llm_result.intent_type,
            confidence=max(keyword_result.confidence * 0.3 + llm_result.confidence * 0.7, llm_result.confidence),
            extracted_info={**keyword_result.extracted_info, **llm_result.extracted_info}
        )

    def classify_query_type(self, query: str) -> str:
        """
        分类查询类型（新查询、继续对话、结束对话）
        """
        prompt = f"""
        请对以下用户输入进行分类：
        
        1. 如果用户输入与历史对话完全无关，且理解该输入无需依赖历史对话信息，请返回"new"
        2. 如果用户输入是在继续历史对话（如追问、补充信息等），请返回"continue"  
        3. 如果用户表示结束对话（如谢谢、再见等），请返回"finish"
        
        用户输入：{query}
        
        请只返回分类结果（new/continue/finish），不要返回其他内容。
        """
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            
            result = response["message"]["content"].strip().lower()
            if result in ["new", "continue", "finish"]:
                return result
            else:
                return "new"  # 默认返回新查询
        except Exception as e:
            logger.error(f"查询类型分类失败: {e}")
            return "new"  # 默认返回新查询