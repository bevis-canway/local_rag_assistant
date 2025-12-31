"""
意图识别系统提示词模板
为小魔仙RAG智能体提供意图识别相关的提示词模板
"""

# 查询分类提示词模板
CLASSIFICATION_PROMPT_TEMPLATE = """
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

# 查询重写提示词模板
QUERY_REWRITE_PROMPT_TEMPLATE = """
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

# 结构化意图解析提示词模板
STRUCTURED_PARSING_PROMPT_TEMPLATE = """
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