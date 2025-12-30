"""
Prompt template management
Unified management of all prompt templates, categorized by function
"""

# RAG-related prompts
RAG_PROMPT_TEMPLATES = {
    "no_document_found": """You are an intelligent assistant. The user's question is: "{query}"

I could not find content related to your question in the local knowledge base.

Please try to answer the user's question based on your general knowledge. If the question involves very specific or professional content that you cannot answer accurately, please honestly inform the user that you cannot provide an accurate answer and suggest that they consult relevant materials or seek professional help.""",
    "rag_answer": """You are a professional assistant, responding to humans in a useful, accurate, and concise manner.

The user has uploaded some documents.
The list of uploaded documents is as follows:
```
{context}
```

These documents have been split and stored in the knowledge base.
Document content can be queried using natural language through the knowledge base retriever.

Please follow these principles:

- If the retrieved documents contain information relevant to the user's question, use that information to provide an accurate answer in Chinese.
- If the retrieved documents contain partial information, combine it with your general knowledge to provide the best possible answer.
- If the retrieved documents are completely unrelated to the question, acknowledge this and answer based on your general knowledge if possible.
- Ensure the final answer is in Chinese.
- The information you obtain may be unrelated to the need, please eliminate it in the final answer, or directly answer "I don't know". Absolutely do not return irrelevant information in the final answer.
- If the final result contains incorrect information, be sure to return the result in natural language!

User question:
{query}

Based on the above document content and your general knowledge, please provide a comprehensive and accurate answer in Chinese. If the documents provide relevant information, prioritize that information in your response.
""",
    "intent_recognition": """请分析用户查询的意图类型。

意图类型定义：
{intent_descriptions}

用户查询：{query}

请分析该查询属于哪种意图类型，并简要说明理由。返回格式：
意图类型: [类型名称]
置信度: [0-1之间的数值]
理由: [简要说明]
""",
    "intent_recognition_with_history": """请分析用户查询的意图类型，考虑历史对话上下文。

意图类型定义：
{intent_descriptions}

历史对话：
{chat_history}

用户查询：{query}

请分析该查询属于哪种意图类型，并简要说明理由。返回格式：
意图类型: [类型名称]
置信度: [0-1之间的数值]
理由: [简要说明]
""",
}

# System prompts
SYSTEM_PROMPTS = {
    "default_assistant": "You are an intelligent assistant providing accurate and useful information.",
    "rag_assistant": "You are a RAG intelligent assistant, answering user questions based on the provided context information.",
}

# Formatting prompts
FORMATTING_PROMPTS = {
    "response_format": """Please answer in the following format:

1. First summarize the core answer
2. Then provide detailed explanation
3. If needed, give relevant suggestions

Keep the answer concise and to the point, highlighting key points.""",
}