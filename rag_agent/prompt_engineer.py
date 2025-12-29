from .prompts.prompt_templates import RAG_PROMPT_TEMPLATES


class PromptEngineer:
    """
    提示工程模块
    参考aidev项目中的提示词设计，构建用于大模型的提示词
    """

    @staticmethod
    def build_rag_prompt(
        query: str, context: str, max_context_length: int = 3000
    ) -> str:
        """
        构建RAG提示词
        参考aidev项目中的多模态提示词模板
        """
        # 如果上下文太长，进行截断
        if len(context) > max_context_length:
            context = context[:max_context_length] + "...(内容已截断)"

        # 使用预定义的RAG提示词模板
        prompt = RAG_PROMPT_TEMPLATES["rag_answer"].format(context=context, query=query)

        return prompt

    @staticmethod
    def build_summarization_prompt(content: str) -> str:
        """
        构建内容摘要提示词
        """
        prompt = f"""
请对以下内容进行简洁的摘要，提取关键信息：
{content}

摘要：
        """.strip()

        return prompt
