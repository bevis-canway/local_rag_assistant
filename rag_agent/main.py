import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import ollama
import tiktoken

# 使用绝对导入替代相对导入
from rag_agent.config import Config
from rag_agent.intent.intent_recognizer import IntentRecognizer, IntentResult  # 添加意图识别导入
from rag_agent.obsidian_connector import ObsidianConnector
from rag_agent.prompt_engineer import PromptEngineer
from rag_agent.prompts.prompt_templates import RAG_PROMPT_TEMPLATES  # 添加此行以导入提示词模板
from rag_agent.retriever import Retriever
from rag_agent.vector_store import VectorStore

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGAgent:
    """
    RAG智能体主类
    整合所有模块，提供问答功能
    参考aidev项目中的Agent实现
    """

    def __init__(self, config: Config):
        self.config = config

        # 初始化各模块
        self.obsidian_connector = ObsidianConnector(
            vault_path=config.OBSIDIAN_VAULT_PATH,
            api_url=config.OBSIDIAN_API_URL,
            api_key=config.OBSIDIAN_API_KEY,
        )

        # 确保环境变量已设置，以便VectorStore可以使用API嵌入

        if config.OLLAMA_API_KEY:
            os.environ.setdefault("OLLAMA_API_KEY", config.OLLAMA_API_KEY)
        if config.OLLAMA_BASE_URL:
            os.environ.setdefault("OLLAMA_BASE_URL", config.OLLAMA_BASE_URL)
        if config.EMBEDDING_MODEL:
            os.environ.setdefault("EMBEDDING_MODEL", config.EMBEDDING_MODEL)

        self.vector_store = VectorStore(persist_path=config.VECTOR_DB_PATH)
        # 使用可配置的相似度阈值，默认为0.3
        similarity_threshold = getattr(config, "SIMILARITY_THRESHOLD", 0.3)
        self.retriever = Retriever(
            self.vector_store,
            top_k=config.TOP_K,
            similarity_threshold=similarity_threshold,
        )
        
        # 初始化意图识别器
        self.intent_recognizer = IntentRecognizer(config, self.vector_store)
        
        self.prompt_engineer = PromptEngineer()

        # 初始化分词器用于计算token
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # 初始化对话历史
        self.chat_history = []

    def index_obsidian_notes(self):
        """
        索引Obsidian笔记到向量库
        """
        logger.info("开始索引Obsidian笔记...")

        # 获取所有笔记列表
        notes = self.obsidian_connector.list_notes()
        logger.info(self.config.OBSIDIAN_VAULT_PATH)
        logger.info(f"找到 {len(notes)} 个笔记")

        documents = []
        for note in notes:
            content = self.obsidian_connector.get_note_content(note["id"])
            if content.strip():
                # 分块处理长文档
                chunks = self._split_document(content, note["title"], note["id"])
                for i, chunk in enumerate(chunks):
                    doc_id = f"{note['id']}_chunk_{i}"
                    documents.append(
                        {
                            "id": doc_id,
                            "content": chunk,
                            "metadata": {
                                "title": note["title"],
                                "path": note["id"],
                                "chunk_index": i,
                            },
                        }
                    )

        # 批量添加到向量库
        if documents:
            self.vector_store.add_documents(documents)
            logger.info(f"完成索引，共添加 {len(documents)} 个文档块")
        else:
            logger.warning("没有找到可索引的文档内容")

    def _split_document(self, content: str, title: str, note_id: str) -> List[str]:
        """
        分割文档为小块
        参考aidev项目中的文档处理逻辑
        """
        # 简单的按长度分割，保留段落完整性
        chunks = []
        paragraphs = content.split("\n\n")

        current_chunk = ""
        for para in paragraphs:
            # 估算token数
            para_token_count = len(self.tokenizer.encode(para))
            current_token_count = len(self.tokenizer.encode(current_chunk))

            if (
                current_token_count + para_token_count > self.config.CHUNK_SIZE
                and current_chunk
            ):
                # 当前块已满，保存并开始新块
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                # 添加到当前块
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # 添加最后一块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # 如果单个段落太长，强制分割
        final_chunks = []
        for chunk in chunks:
            if len(self.tokenizer.encode(chunk)) > self.config.CHUNK_SIZE:
                # 按字符强制分割
                sub_chunks = self._force_split_chunk(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _force_split_chunk(self, text: str) -> List[str]:
        """
        强制分割过长的文本块
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.config.CHUNK_SIZE

            if end >= text_length:
                # 如果end超出文本长度，直接添加剩余文本
                chunks.append(text[start:text_length].strip())
                break

            # 尝试在句子边界分割
            found_boundary = False

            # 向前搜索句子边界
            while end > start:
                if end < text_length and text[end] in ".!?。！？\n":
                    end += 1  # 包含边界字符
                    found_boundary = True
                    break
                end -= 1

            # 如果没有找到合适的句子边界，按固定长度分割
            if not found_boundary:
                end = min(start + self.config.CHUNK_SIZE, text_length)

            chunks.append(text[start:end].strip())
            start = end

            # 添加重叠部分
            if self.config.CHUNK_OVERLAP > 0:
                start = max(
                    start - self.config.CHUNK_OVERLAP, start
                )  # 确保不会倒退到start位置之前

        return chunks

    def query(self, user_question: str, chat_history: List[Dict] = None) -> str:
        """
        处理用户查询
        """
        logger.info(f"处理查询: {user_question}")

        # 如果没有提供对话历史，则使用实例的对话历史
        if chat_history is None:
            chat_history = self.chat_history

        # 检查是否是关于历史对话的查询
        if self._is_history_query(user_question):
            history_response = self._handle_history_query(user_question, chat_history)
            if history_response:
                # 更新对话历史
                self.chat_history.append({
                    "query": user_question,
                    "response": history_response,
                    "intent": "history_query"
                })
                
                # 限制对话历史长度，避免过长
                if len(self.chat_history) > 10:  # 保留最近10轮对话
                    self.chat_history = self.chat_history[-10:]
                
                return history_response

        # 1. 进行意图识别
        intent_result: IntentResult = self.intent_recognizer.recognize_intent(user_question, chat_history)
        logger.info(f"意图识别结果: {intent_result.intent_type}, 置信度: {intent_result.confidence}")
        
        # 2. 根据意图类型决定处理方式
        if intent_result.intent_type == "chit_chat":
            # 对于闲聊类意图，直接使用通用模型回答
            prompt = RAG_PROMPT_TEMPLATES["no_document_found"].format(query=user_question)
        else:
            # 对于其他意图类型，进行文档检索
            filtered_results, has_relevant_docs = self.retriever.retrieve_and_filter_by_similarity(user_question)

            if has_relevant_docs:
                # 如果有相关文档，格式化并使用RAG提示词
                context = self.retriever.format_results(filtered_results)
                prompt = self.prompt_engineer.build_rag_prompt(user_question, context)
            else:
                # 如果没有相关文档，使用无文档提示词
                logger.info("未找到相关文档，使用通用模型回答")
                prompt = RAG_PROMPT_TEMPLATES["no_document_found"].format(query=user_question)

        # 3. 调用Ollama模型
        try:
            response = ollama.chat(
                model=self.config.OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response["message"]["content"]
            logger.info("成功获取模型回答")
            
            # 更新对话历史
            self.chat_history.append({
                "query": user_question,
                "response": answer,
                "intent": intent_result.intent_type
            })
            
            # 限制对话历史长度，避免过长
            if len(self.chat_history) > 10:  # 保留最近10轮对话
                self.chat_history = self.chat_history[-10:]
            
            return answer
        except Exception as e:
            logger.error(f"调用Ollama模型失败: {e}")
            return f"抱歉，处理您的问题时出现错误: {str(e)}"
    
    def _is_history_query(self, query: str) -> bool:
        """
        检查查询是否涉及历史对话
        """
        history_keywords = [
            "前面", "之前", "刚才", "上一个", "第一个", "历史", "之前问", "前面问", 
            "刚才问", "上个", "之前的", "前面的", "刚才的", "我问", "我的问题",
            "前面说", "刚才说", "之前说", "对话历史", "我们刚才", "我们之前"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in history_keywords)
    
    def _handle_history_query(self, query: str, chat_history: List[Dict]) -> str:
        """
        处理涉及历史对话的查询
        """
        if not chat_history:
            return "我们还没有开始对话，您还没有问过任何问题。"
        
        # 根据查询类型提供不同的历史信息
        if "第一个" in query or "第一个问题" in query:
            first_turn = chat_history[0]
            return f"您的第一个问题是：\"{first_turn['query']}\""
        
        elif "前面" in query or "之前" in query or "刚才" in query or "上一个" in query:
            # 返回最近的几个对话
            recent_history = chat_history[-3:]  # 返回最近3轮对话
            history_str = "最近的对话记录：\n"
            for i, turn in enumerate(recent_history, 1):
                history_str += f"{i}. 问题: {turn['query']}\n   回答: {turn['response']}\n"
            return history_str.strip()
        
        elif "什么问题" in query or "问了什么" in query:
            # 返回最近的问题
            recent_questions = [turn['query'] for turn in chat_history[-5:]]  # 最近5个问题
            if recent_questions:
                questions_str = "\n".join([f"- {q}" for q in recent_questions])
                return f"您最近问过的问题包括：\n{questions_str}"
            else:
                return "我无法找到您之前问过的问题。"
        
        else:
            # 默认返回最近的对话
            recent_history = chat_history[-2:]  # 返回最近2轮对话
            history_str = "最近的对话记录：\n"
            for i, turn in enumerate(recent_history, 1):
                history_str += f"{i}. 问题: {turn['query']}\n   回答: {turn['response']}\n"
            return history_str.strip()

    def run_cli(self):
        """
        运行命令行交互界面
        """
        print(
            "小魔仙RAG智能体已启动！输入 'quit' 或 'exit' 退出，输入 'reindex' 重新索引笔记。"
        )
        print("输入 'status' 查看向量库状态。")
        print("输入 'clear' 清空对话历史。")

        while True:
            try:
                user_input = input("\n您的问题: ").strip()

                if user_input.lower() in ["quit", "exit"]:
                    print("再见！")
                    break
                elif user_input.lower() == "reindex":
                    print("正在重新索引笔记...")
                    self.index_obsidian_notes()
                    print("索引完成！")
                    continue
                elif user_input.lower() == "status":
                    stats = self.vector_store.collection.count()
                    print(f"向量库状态: 当前有 {stats} 个文档块")
                    continue
                elif user_input.lower() == "clear":
                    self.chat_history = []
                    print("对话历史已清空。")
                    continue
                elif not user_input:
                    continue

                # 处理查询，传递当前对话历史
                answer = self.query(user_input, self.chat_history)
                print(f"\n回答: {answer}")

            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                logger.error(f"处理输入时出错: {e}")
                print(f"出现错误: {str(e)}")


def main():
    """
    主函数
    """
    # 加载配置
    config = Config()

    # 检查必要的配置
    if not config.OBSIDIAN_VAULT_PATH:
        print("错误: 请设置 OBSIDIAN_VAULT_PATH 环境变量指向您的Obsidian知识库路径")
        return

    # 创建RAG智能体
    agent = RAGAgent(config)

    # 检查向量库是否为空，如果为空则索引笔记
    try:
        # 尝获取一个简单的统计来判断是否已有数据
        stats = agent.vector_store.collection.count()
        print(f"当前向量库状态: {stats} 个文档块")
        if stats == 0:
            print("检测到向量库为空，正在索引Obsidian笔记...")
            agent.index_obsidian_notes()
            print("索引完成！")
    except Exception as e:
        print(f"检查向量库状态时出错: {e}")
        print("正在索引Obsidian笔记...")
        agent.index_obsidian_notes()
        print("索引完成！")

    # 运行命令行界面
    agent.run_cli()


if __name__ == "__main__":
    main()
