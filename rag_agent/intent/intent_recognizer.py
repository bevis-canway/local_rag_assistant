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
            },
            "history_query": {
                "keywords": ["前面", "之前", "刚才", "上一个", "第一个", "历史", "之前问", "前面问", 
                           "刚才问", "上个", "之前的", "前面的", "刚才的", "我问", "我的问题",
                           "前面说", "刚才说", "之前说", "对话历史", "我们刚才", "我们之前"],
                "description": "关于历史对话的查询"
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
        # 1. 检查是否为历史对话查询
        history_intent = self._recognize_history_query(query, chat_history)
        if history_intent:
            return history_intent
        
        # 2. 基于关键词的初步识别
        keyword_intent = self._recognize_by_keywords(query)
        
        # 3. 基于LLM的意图识别（更精确）
        llm_intent = self._recognize_by_llm(query, chat_history)
        
        # 4. 综合判断
        final_intent = self._combine_intent_results(keyword_intent, llm_intent, query)
        
        logger.info(f"意图识别结果 - 查询: {query}, 意图类型: {final_intent.intent_type}, 置信度: {final_intent.confidence}")
        
        return final_intent

    def _recognize_history_query(self, query: str, chat_history: List[Dict] = None) -> Optional[IntentResult]:
        """
        专门识别历史对话查询
        """
        if not chat_history:
            return None
            
        # 检查是否为历史查询
        history_keywords = self.intent_types["history_query"]["keywords"]
        query_lower = query.lower()
        
        for keyword in history_keywords:
            if keyword in query_lower:
                # 确认这是历史查询
                return IntentResult("history_query", 0.9, {"query_text": query})
        
        return None

    def _recognize_by_keywords(self, query: str) -> Optional[IntentResult]:
        """
        基于关键词匹配的意图识别
        """
        query_lower = query.lower()
        
        best_match = None
        best_score = 0
        
        for intent_type, intent_info in self.intent_types.items():
            # 跳过历史查询，因为它已经在单独的方法中处理
            if intent_type == "history_query":
                continue
                
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

    def classify_query_type(self, query: str, chat_history: List[Dict] = None) -> str:
        """
        分类查询类型（新查询、继续对话、结束对话）
        参考aidev项目实现
        """
        # 构建提示词，参考aidev项目中的latest_query_classification逻辑
        prompt = self._build_query_classification_prompt(query, chat_history)
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            
            result = response["message"]["content"].strip().lower()
            
            # 检查返回结果，参考aidev项目格式
            if "<<<<<new>>>>>" in result:
                return "new"
            elif "<<<<<continue>>>>>" in result:
                return "continue" 
            elif "<<<<<finish>>>>>" in result:
                return "finish"
            else:
                # 如果LLM没有按格式返回，使用规则判断
                return self._rule_based_query_classification(query, chat_history)
                
        except Exception as e:
            logger.error(f"查询类型分类失败: {e}")
            return self._rule_based_query_classification(query, chat_history)

    def _build_query_classification_prompt(self, query: str, chat_history: List[Dict] = None) -> str:
        """
        构建查询分类的提示词，参考aidev项目
        """
        if chat_history:
            chat_history_str = "\n".join([
                f"HumanMessage(content='{item['query']}'), AIMessage(content='{item['response']}')" 
                for item in chat_history[-3:]  # 只取最近3轮对话
            ])
        else:
            chat_history_str = "[]"
        
        # 参考aidev项目中的latest_query_classification_sys_prompt_template
        prompt = f"""
现有一个智能对话系统。
我会给你一段用户和该智能对话系统的历史对话，以及当前用户的最新输入。
用户和该智能对话系统的历史对话的格式样例为：
{chat_history_str}
其中"HumanMessage"表示用户，"AIMessage"表示该智能对话系统。

你负责对当前用户的最新输入进行分类：
1. 如果你认为用户的这个最新输入跟历史对话信息已经完全无关，且理解该最新输入已经无需依赖历史对话信息，请只返回`<<<<<new>>>>>`

2. 如果你认为用户的这个最新输入是对历史对话的正面评价、正面反馈、正面确认等，且会话到此已经可以结束了，
   例如用户最新输入了"谢谢"、"你说得真好"等，请只返回`<<<<<finish>>>>>`

3. 其余所有情况，例如用户的这个最新输入是在接着历史对话继续进行提问或答复，或者例如完整理解这个最新输入需要依赖历史对话，
   请只返回`<<<<<continue>>>>>`

注意：
1. 举个例子，假设对话历史为：[HumanMessage(content='我的手机号xxx存在经常被无故停机的问题'), AIMessage(content='收到')]，
   假设用户当前的最新输入为"手机号yyy也是"，
   则需要依赖历史对话信息才能知道用户当前的最新输入是想询问"手机号yyy也存在经常被无故停机的问题"，因此需要返回`<<<<<continue>>>>>`
2. 再举个例子，假设对话历史为：[HumanMessage(content='广东省的省会是哪个城市'), AIMessage(content='广州')]，
   假设用户当前的最新输入为"福建呢"，则需要依赖历史对话信息才能知道用户当前的最新输入是想询问"福建省的省会是哪个城市"，
   因此需要返回`<<<<<continue>>>>>`
3. 务必确认会话到此已经可以结束了，才可以返回`<<<<<finish>>>>>`
4. 只返回`<<<<<new>>>>>`或者`<<<<<continue>>>>>`或者`<<<<<finish>>>>>`即可！永远不要返回其他任何内容！永远不要返回你的推理过程！

用户和该智能对话系统的对话历史如下：```{chat_history_str}```


用户当前的最新输入如下：```{query}```
"""
        
        return prompt

    def _rule_based_query_classification(self, query: str, chat_history: List[Dict] = None) -> str:
        """
        基于规则的查询分类，作为LLM失败时的备选
        """
        if not chat_history:
            return "new"
        
        # 检查是否为结束语
        finish_keywords = ["谢谢", "再见", "好的", "明白了", "结束", "拜拜", "ok", "bye"]
        if any(keyword in query for keyword in finish_keywords):
            return "finish"
        
        # 检查是否为继续对话
        continue_keywords = ["然后", "接着", "还有", "另外", "对了", "是的", "嗯", "还有吗", "继续"]
        if any(keyword in query for keyword in continue_keywords):
            return "continue"
        
        # 默认为新查询
        return "new"

    def query_rewrite_for_independence(self, query: str, chat_history: List[Dict] = None) -> str:
        """
        将用户查询重写为独立的查询，参考aidev项目的query_rewrite_for_independence
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
            if not rewritten_query or rewritten_query == query:
                return query
            
            return rewritten_query
        except Exception as e:
            logger.error(f"查询重写失败: {e}")
            return query

    def _build_query_rewrite_prompt(self, query: str, chat_history: List[Dict] = None) -> str:
        """
        构建查询重写的提示词，参考aidev项目
        """
        if chat_history:
            chat_history_str = "\n".join([
                f"用户: {item['query']}\n助手: {item['response']}" 
                for item in chat_history[-3:]  # 只取最近3轮对话
            ])
        else:
            chat_history_str = ""
        
        # 参考aidev项目中的query_rewrite_for_independence_sys_prompt_template
        prompt = f"""
现有一个智能对话系统。
我会给你一段用户和该智能对话系统的历史对话，以及当前用户的最新输入。
用户和该智能对话系统的历史对话的格式样例为：
{chat_history_str}
其中"HumanMessage"表示用户，"AIMessage"表示该智能对话系统。

你负责根据这些信息，将用户的最新输入重写成一个完全独立的query。
我会仅仅使用你重写后的query去私域知识库中检索相关文档，而不再使用历史对话！
因此，你重写后的query信息要全面、要包含所有必要的信息、完全不再依赖历史对话信息！

注意：
1. 举个例子，假设对话历史为：[HumanMessage(content='我的手机号xxx存在经常被无故停机的问题'), AIMessage(content='收到')]，
   假设用户当前的最新输入为"手机号yyy也是"，你可以返回"手机号yyy也存在经常被无故停机的问题"
2. 再举个例子，假设对话历史为：[HumanMessage(content='广东省的省会是哪个城市'), AIMessage(content='广州')]，
   假设用户当前的最新输入为"福建呢"，你可以返回"福建省的省会是哪个城市"
3. 只返回重写后的query即可！不要返回其他任何内容！返回中不要出现"用户query重写："等表述！

用户和该智能对话系统的对话历史如下：```{chat_history_str}```


用户当前的最新输入如下：```{query}```

注意：只返回重写后的query即可！不要返回其他任何内容！返回中不要出现"用户提问重写："等表述！
"""
        
        return prompt