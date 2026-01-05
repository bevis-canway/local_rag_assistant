import logging
import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, AsyncGenerator

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import ollama
import tiktoken

# 使用绝对导入替代相对导入
from rag_agent.config import Config
from rag_agent.hallucination_detector import HallucinationDetector, HallucinationCheckResult
from rag_agent.intent.intent_recognizer import IntentRecognizer, IntentResult  # 添加
from rag_agent.obsidian_connector import ObsidianConnector
from rag_agent.prompt_engineer import PromptEngineer
from rag_agent.prompts.prompt_templates import RAG_PROMPT_TEMPLATES  # 添加此行以导入提示词模板
from rag_agent.retriever import Retriever
from rag_agent.vector_store import VectorStore
from rag_agent.streaming_handler import StreamingHandler, StreamEvent, EventType  # 导入流式响应处理模块
from rag_agent.knowledge_base_manager import KnowledgeBaseManager, KnowledgeBaseConfig  # 导入多知识库管理模块

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

        # 初始化知识库管理器
        self.knowledge_base_manager = KnowledgeBaseManager(config)
        
        # 初始化各模块
        self.obsidian_connector = ObsidianConnector(
            vault_path=config.OBSIDIAN_VAULT_PATH,
            api_url=config.OBSIDIAN_API_URL,
            api_key=config.OBSIDIAN_API_KEY,
        )
        
        # 初始化幻觉检测器
        self.hallucination_detector = HallucinationDetector(config)

        # 确保环境变量已设置，以便VectorStore可以使用API嵌入

        if config.OLLAMA_API_KEY:
            os.environ.setdefault("OLLAMA_API_KEY", config.OLLAMA_API_KEY)
        if config.OLLAMA_BASE_URL:
            os.environ.setdefault("OLLAMA_BASE_URL", config.OLLAMA_BASE_URL)
        if config.EMBEDDING_MODEL:
            os.environ.setdefault("EMBEDDING_MODEL", config.EMBEDDING_MODEL)

        # 获取第一个可用的知识库作为默认知识库（保持向后兼容性）
        kb_infos = self.knowledge_base_manager.list_knowledge_bases()
        if kb_infos:
            default_kb_name = kb_infos[0].name
        else:
            # 如果没有自动发现的知识库，添加默认知识库
            from rag_agent.knowledge_base_manager import KnowledgeBaseConfig
            default_kb = KnowledgeBaseConfig(
                name="default",
                type="obsidian",
                path=config.OBSIDIAN_VAULT_PATH,
                description="默认Obsidian知识库",
                enabled=True,
                vector_store_path=config.VECTOR_DB_PATH
            )
            self.knowledge_base_manager.add_knowledge_base(default_kb)
            default_kb_name = "default"
        
        self.vector_store = self.knowledge_base_manager.get_vector_store(default_kb_name)
        # 使用可配置的相似度阈值，默认为0.3
        similarity_threshold = getattr(config, "SIMILARITY_THRESHOLD", 0.3)
        self.retriever = Retriever(
            self.vector_store,
            top_k=config.TOP_K,
            similarity_threshold=similarity_threshold,
            config=config,
            knowledge_base_name=default_kb_name,  # 指定默认知识库
        )
        
        # 初始化意图识别器
        self.intent_recognizer = IntentRecognizer(config, self.vector_store)
        
        self.prompt_engineer = PromptEngineer()

        # 初始化分词器用于计算token
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # 初始化对话历史
        self.chat_history = []
        
        # 初始化流式响应处理器
        self.streaming_handler = StreamingHandler()

    def index_obsidian_notes(self, knowledge_base_name: str = "default"):
        """
        索引Obsidian笔记到向量库
        """
        logger.info(f"开始索引知识库 {knowledge_base_name} 的笔记...")

        # 获取对应知识库的连接器
        if knowledge_base_name == "default":
            connector = self.obsidian_connector
            vector_store = self.vector_store
        else:
            connector = self.knowledge_base_manager.get_connector(knowledge_base_name)
            vector_store = self.knowledge_base_manager.get_vector_store(knowledge_base_name)
        
        if not connector or not vector_store:
            logger.error(f"无法获取知识库 {knowledge_base_name} 的连接器或向量存储")
            return

        # 获取所有笔记列表
        notes = connector.list_notes()
        logger.info(f"知识库 {knowledge_base_name} 路径: {getattr(connector, 'vault_path', 'Unknown')}")
        logger.info(f"找到 {len(notes)} 个笔记")

        documents = []
        failed_notes = []
        for note in notes:
            try:
                content = connector.get_note_content(note["id"])
                if content.strip():
                    # 分块处理长文档
                    chunks = self._split_document(content, note["title"], note["id"])
                    for i, chunk in enumerate(chunks):
                        doc_id = f"{knowledge_base_name}_{note['id']}_chunk_{i}"
                        documents.append(
                            {
                                "id": doc_id,
                                "content": chunk,
                                "metadata": {
                                    "title": note["title"],
                                    "path": note["id"],
                                    "knowledge_base": knowledge_base_name,  # 标识文档来自哪个知识库
                                    "chunk_index": i,
                                },
                            }
                        )
            except Exception as e:
                logger.error(f"处理笔记 {note['id']} 时出错: {e}")
                failed_notes.append(note['id'])
        
        if failed_notes:
            logger.warning(f"知识库 {knowledge_base_name} 中有 {len(failed_notes)} 个笔记处理失败: {failed_notes}")

        # 批量添加到向量库
        if documents:
            vector_store.add_documents(documents)
            logger.info(f"知识库 {knowledge_base_name} 索引完成，共添加 {len(documents)} 个文档块")
        else:
            logger.warning(f"知识库 {knowledge_base_name} 没有找到可索引的文档内容")

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

        # 1. 进行意图识别
        intent_result = self.intent_recognizer.recognize_intent(user_question, chat_history)
        logger.info(f"意图识别结果: {intent_result.intent_type}, 置信度: {intent_result.confidence}")

        # 2. 检查是否需要澄清
        clarification = self.intent_recognizer.get_clarification_response(intent_result)
        if clarification and intent_result.intent_type != "history_query":
            # 更新对话历史
            self.chat_history.append({
                "query": user_question,
                "response": clarification,
                "intent": "clarification"
            })
            
            # 限制对话历史长度，避免过长
            if len(self.chat_history) > 10:  # 保留最近10轮对话
                self.chat_history = self.chat_history[-10:]
            
            return clarification

        # 3. 根据意图类型决定处理方式
        if intent_result.intent_type == "chitchat":
            # 对于闲聊类意图，直接使用通用模型回答
            prompt = RAG_PROMPT_TEMPLATES["no_document_found"].format(query=user_question)
        elif intent_result.intent_type == "history_query":
            # 对于历史查询意图，使用专门的处理方法
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
            else:
                prompt = RAG_PROMPT_TEMPLATES["no_document_found"].format(query=user_question)
        elif intent_result.intent_type in ["knowledge_query", "ambiguous"]:
            # 对于知识查询或模糊查询（但置信度不够需要澄清的），使用重写后的查询进行检索
            # 注意：即使置信度较低，我们也尝试检索，因为这可能是一个有效的知识查询
            query_to_use = intent_result.rewritten_query if intent_result.rewritten_query else user_question
            
            # 进行文档检索 - 现在支持跨多个知识库检索
            filtered_results, has_relevant_docs = self._retrieve_from_all_knowledge_bases(query_to_use)

            if has_relevant_docs:
                # 如果有相关文档，格式化并使用RAG提示词
                # 无论查询类型是knowledge_query还是ambiguous，只要有相关文档就使用RAG
                context = self.retriever.format_results(filtered_results)
                prompt = self.prompt_engineer.build_rag_prompt(query_to_use, context)
            else:
                # 如果没有相关文档，使用无文档提示词
                logger.info("未找到相关文档，使用通用模型回答")
                prompt = RAG_PROMPT_TEMPLATES["no_document_found"].format(query=query_to_use)
        else:
            # 其他意图类型，使用通用模型回答
            prompt = RAG_PROMPT_TEMPLATES["no_document_found"].format(query=user_question)

        # 4. 调用Ollama模型
        try:
            response = ollama.chat(
                model=self.config.OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options=self.config.get_generation_options()
            )
            answer = response["message"]["content"]
            logger.info("成功获取模型回答")
            
            # 如果是知识查询且有相关文档，进行幻觉检测
            if intent_result.intent_type in ["knowledge_query", "ambiguous"] and has_relevant_docs:
                logger.info("进行幻觉检测...")
                hallucination_result = self.hallucination_detector.detect_hallucinations(
                    response=answer,
                    retrieved_docs=filtered_results,
                    query=query_to_use
                )
                
                # 如果检测到幻觉且置信度较低，提供警告
                if not hallucination_result.is_consistent and hallucination_result.confidence_score < 0.7:
                    logger.warning(f"检测到潜在幻觉，置信度: {hallucination_result.confidence_score}")
                    warning_msg = "\n\n⚠️ 注意：以下信息基于检索到的文档，但请谨慎验证关键信息的准确性。\n"
                    answer += warning_msg
                    
                    # 记录不一致之处
                    if hallucination_result.inconsistencies:
                        logger.debug(f"检测到的不一致之处: {hallucination_result.inconsistencies}")
            
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

    def _retrieve_from_all_knowledge_bases(self, query: str):
        """
        从所有知识库中检索相关文档
        """
        all_filtered_results = []
        has_relevant_docs = False
        
        # 获取所有启用的知识库
        kb_infos = self.knowledge_base_manager.list_knowledge_bases()
        enabled_kbs = [kb for kb in kb_infos if kb.enabled]
        
        logger.info(f"正在从 {len(enabled_kbs)} 个知识库中检索")
        
        for kb_info in enabled_kbs:
            logger.debug(f"正在检索知识库: {kb_info.name}")
            
            # 获取该知识库的向量存储
            vector_store = self.knowledge_base_manager.get_vector_store(kb_info.name)
            if vector_store:
                # 创建临时检索器用于该知识库
                temp_retriever = Retriever(
                    vector_store=vector_store,
                    top_k=self.retriever.top_k,
                    similarity_threshold=self.retriever.similarity_threshold,
                    config=self.config,
                    knowledge_base_name=kb_info.name
                )
                
                # 从特定知识库检索
                kb_results, kb_has_docs = temp_retriever.retrieve_and_filter_by_similarity(
                    query, knowledge_base_filter=kb_info.name
                )
                
                if kb_has_docs:
                    logger.debug(f"知识库 {kb_info.name} 找到 {len(kb_results)} 个相关文档")
                    all_filtered_results.extend(kb_results)
                    has_relevant_docs = True
                else:
                    logger.debug(f"知识库 {kb_info.name} 未找到相关文档")
            else:
                logger.warning(f"无法获取知识库 {kb_info.name} 的向量存储")
        
        logger.info(f"总共从 {len(enabled_kbs)} 个知识库中检索到 {len(all_filtered_results)} 个相关文档")
        return all_filtered_results, has_relevant_docs

    async def query_stream(self, user_question: str, chat_history: List[Dict] = None) -> AsyncGenerator[StreamEvent, None]:
        """
        异步流式处理用户查询
        生成流式事件，包括思考过程、参考文档和最终回答
        """
        logger.info(f"流式处理查询: {user_question}")

        # 如果没有提供对话历史，则使用实例的对话历史
        if chat_history is None:
            chat_history = self.chat_history

        # 1. 进行意图识别
        intent_result = self.intent_recognizer.recognize_intent(user_question, chat_history)
        logger.info(f"意图识别结果: {intent_result.intent_type}, 置信度: {intent_result.confidence}")

        # 2. 检查是否需要澄清
        clarification = self.intent_recognizer.get_clarification_response(intent_result)
        if clarification and intent_result.intent_type != "history_query":
            # 发送澄清信息
            yield StreamEvent(
                event=EventType.TEXT,
                content=clarification,
                cover=False
            )
            
            # 更新对话历史
            self.chat_history.append({
                "query": user_question,
                "response": clarification,
                "intent": "clarification"
            })
            
            # 限制对话历史长度，避免过长
            if len(self.chat_history) > 10:  # 保留最近10轮对话
                self.chat_history = self.chat_history[-10:]
            
            return

        # 3. 根据意图类型决定处理方式
        if intent_result.intent_type == "chitchat":
            # 对于闲聊类意图，直接使用通用模型回答
            prompt = RAG_PROMPT_TEMPLATES["no_document_found"].format(query=user_question)
        elif intent_result.intent_type == "history_query":
            # 对于历史查询意图，使用专门的处理方法
            history_response = self._handle_history_query(user_question, chat_history)
            if history_response:
                # 发送历史查询结果
                yield StreamEvent(
                    event=EventType.TEXT,
                    content=history_response,
                    cover=False
                )
                
                # 更新对话历史
                self.chat_history.append({
                    "query": user_question,
                    "response": history_response,
                    "intent": "history_query"
                })
                
                # 限制对话历史长度，避免过长
                if len(self.chat_history) > 10:  # 保留最近10轮对话
                    self.chat_history = self.chat_history[-10:]
                
                return
            else:
                prompt = RAG_PROMPT_TEMPLATES["no_document_found"].format(query=user_question)
        elif intent_result.intent_type in ["knowledge_query", "ambiguous"]:
            # 对于知识查询或模糊查询（但置信度不够需要澄清的），使用重写后的查询进行检索
            # 注意：即使置信度较低，我们也尝试检索，因为这可能是一个有效的知识查询
            query_to_use = intent_result.rewritten_query if intent_result.rewritten_query else user_question
            
            # 进行文档检索 - 现在支持跨多个知识库检索
            yield StreamEvent(
                event=EventType.THINK,
                content="正在检索相关文档...",
                cover=False
            )
            
            filtered_results, has_relevant_docs = self._retrieve_from_all_knowledge_bases(query_to_use)

            if has_relevant_docs:
                # 如果有相关文档，格式化并使用RAG提示词
                # 无论查询类型是knowledge_query还是ambiguous，只要有相关文档就使用RAG
                context = self.retriever.format_results(filtered_results)
                prompt = self.prompt_engineer.build_rag_prompt(query_to_use, context)
                
                # 发送参考文档信息
                yield StreamEvent(
                    event=EventType.REFERENCE_DOC,
                    content=f"找到 {len(filtered_results)} 个相关文档",
                    documents=filtered_results,
                    cover=False
                )
            else:
                # 如果没有相关文档，使用无文档提示词
                logger.info("未找到相关文档，使用通用模型回答")
                prompt = RAG_PROMPT_TEMPLATES["no_document_found"].format(query=query_to_use)
                
                # 发送无文档信息
                yield StreamEvent(
                    event=EventType.THINK,
                    content="未找到相关文档，使用通用模型回答",
                    cover=False
                )
        else:
            # 其他意图类型，使用通用模型回答
            prompt = RAG_PROMPT_TEMPLATES["no_document_found"].format(query=user_question)

        # 4. 调用Ollama模型进行流式生成
        try:
            # 发送开始生成事件
            yield StreamEvent(
                event=EventType.THINK,
                content="正在生成回答...",
                cover=False
            )
            
            # 调用Ollama模型进行流式生成
            stream = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ollama.chat(
                    model=self.config.OLLAMA_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    options=self.config.get_generation_options(),
                    stream=True
                )
            )
            
            full_response = ""
            hallucination_checked = False
            
            # 流式处理模型响应
            for chunk in stream:
                if chunk.get('message') and chunk['message'].get('content'):
                    content = chunk['message']['content']
                    full_response += content
                    
                    # 发送文本内容
                    yield StreamEvent(
                        event=EventType.TEXT,
                        content=content,
                        cover=False
                    )

            # 如果是知识查询且有相关文档，进行幻觉检测
            if intent_result.intent_type in ["knowledge_query", "ambiguous"] and has_relevant_docs:
                logger.info("进行幻觉检测...")
                hallucination_result = self.hallucination_detector.detect_hallucinations(
                    response=full_response,
                    retrieved_docs=filtered_results,
                    query=query_to_use
                )
                
                # 如果检测到幻觉且置信度较低，提供警告
                if not hallucination_result.is_consistent and hallucination_result.confidence_score < 0.7:
                    logger.warning(f"检测到潜在幻觉，置信度: {hallucination_result.confidence_score}")
                    warning_msg = "\n\n⚠️ 注意：以下信息基于检索到的文档，但请谨慎验证关键信息的准确性。\n"
                    
                    # 发送警告信息
                    yield StreamEvent(
                        event=EventType.TEXT,
                        content=warning_msg,
                        cover=False
                    )
                    
                    full_response += warning_msg
                    
                    # 记录不一致之处
                    if hallucination_result.inconsistencies:
                        logger.debug(f"检测到的不一致之处: {hallucination_result.inconsistencies}")
            
            # 更新对话历史
            self.chat_history.append({
                "query": user_question,
                "response": full_response,
                "intent": intent_result.intent_type
            })
            
            # 限制对话历史长度，避免过长
            if len(self.chat_history) > 10:  # 保留最近10轮对话
                self.chat_history = self.chat_history[-10:]
            
            # 发送完成事件
            yield StreamEvent(
                event=EventType.DONE,
                content="回答生成完成",
                cover=False
            )
            
        except Exception as e:
            logger.error(f"调用Ollama模型失败: {e}")
            yield StreamEvent(
                event=EventType.ERROR,
                content=f"抱歉，处理您的问题时出现错误: {str(e)}",
                cover=False
            )

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
        print("输入 'reindex [knowledge_base_name]' 重新索引指定知识库。")
        print("输入 'status' 查看向量库状态。")
        print("输入 'clear' 清空对话历史。")
        print("输入 'list' 查看所有知识库。")

        while True:
            try:
                user_input = input("\n您的问题: ").strip()

                if user_input.lower() in ["quit", "exit"]:
                    print("再见！")
                    break
                elif user_input.lower() == "reindex":
                    print("正在重新索引笔记...")
                    # 检查是否有知识库名称参数
                    parts = user_input.split()
                    if len(parts) > 1:
                        # 指定特定知识库
                        kb_name = parts[1]
                        if self.knowledge_base_manager.get_knowledge_base(kb_name):
                            self.index_obsidian_notes(kb_name)
                            print(f"知识库 {kb_name} 索引完成！")
                        else:
                            print(f"错误: 知识库 {kb_name} 不存在")
                            print("可用的知识库:")
                            for kb_info in self.knowledge_base_manager.list_knowledge_bases():
                                status = "启用" if kb_info.enabled else "禁用"
                                print(f"  - {kb_info.name} ({status})")
                    else:
                        # 重新索引所有启用的知识库
                        kb_infos = self.knowledge_base_manager.list_knowledge_bases()
                        enabled_kbs = [kb for kb in kb_infos if kb.enabled]
                        if len(enabled_kbs) == 1:
                            # 如果只有一个知识库，只索引当前默认知识库
                            self.index_obsidian_notes(self.retriever.knowledge_base_name)
                            print("索引完成！")
                        else:
                            # 如果有多个知识库，索引所有启用的知识库
                            failed_kbs = []
                            success_kbs = []
                            for kb_info in enabled_kbs:
                                print(f"正在索引知识库: {kb_info.name}")
                                try:
                                    self.index_obsidian_notes(kb_info.name)
                                    success_kbs.append(kb_info.name)
                                except Exception as e:
                                    logger.error(f"索引知识库 {kb_info.name} 时出错: {e}")
                                    failed_kbs.append(kb_info.name)
                            
                            print(f"完成索引！成功: {len(success_kbs)}, 失败: {len(failed_kbs)}")
                            if success_kbs:
                                print(f"成功索引: {', '.join(success_kbs)}")
                            if failed_kbs:
                                print(f"索引失败: {', '.join(failed_kbs)}")
                                print("请单独重试失败的知识库: reindex [knowledge_base_name]")
                    continue
                elif user_input.lower() == "status":
                    stats = self.vector_store.collection.count()
                    print(f"向量库状态: 当前有 {stats} 个文档块")
                    continue
                elif user_input.lower() == "list":
                    kb_infos = self.knowledge_base_manager.list_knowledge_bases()
                    print("可用的知识库:")
                    for kb_info in kb_infos:
                        status = "启用" if kb_info.enabled else "禁用"
                        print(f"  - {kb_info.name} ({status}): {kb_info.path} ({kb_info.indexed_documents_count} 个文档)")
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
            # 获取所有启用的知识库，如果只有一个则索引默认知识库，否则索引所有知识库
            kb_infos = agent.knowledge_base_manager.list_knowledge_bases()
            enabled_kbs = [kb for kb in kb_infos if kb.enabled]
            
            if len(enabled_kbs) > 1:
                # 如果有多个知识库，索引所有启用的知识库
                failed_kbs = []
                success_kbs = []
                for kb_info in enabled_kbs:
                    print(f"正在索引知识库: {kb_info.name}")
                    try:
                        agent.index_obsidian_notes(kb_info.name)
                        success_kbs.append(kb_info.name)
                    except Exception as e:
                        logger.error(f"索引知识库 {kb_info.name} 时出错: {e}")
                        failed_kbs.append(kb_info.name)
                
                print(f"初始索引完成！成功: {len(success_kbs)}, 失败: {len(failed_kbs)}")
                if success_kbs:
                    print(f"成功索引: {', '.join(success_kbs)}")
                if failed_kbs:
                    print(f"索引失败: {', '.join(failed_kbs)}")
            else:
                # 只有一个知识库，索引默认知识库
                agent.index_obsidian_notes(agent.retriever.knowledge_base_name)
            print("初始索引完成！")
    except Exception as e:
        print(f"检查向量库状态时出错: {e}")
        print("正在索引Obsidian笔记...")
        try:
            # 同样处理多个知识库
            kb_infos = agent.knowledge_base_manager.list_knowledge_bases()
            enabled_kbs = [kb for kb in kb_infos if kb.enabled]
            
            if len(enabled_kbs) > 1:
                # 如果有多个知识库，索引所有启用的知识库
                failed_kbs = []
                success_kbs = []
                for kb_info in enabled_kbs:
                    print(f"正在索引知识库: {kb_info.name}")
                    try:
                        agent.index_obsidian_notes(kb_info.name)
                        success_kbs.append(kb_info.name)
                    except Exception as e_inner:
                        logger.error(f"索引知识库 {kb_info.name} 时出错: {e_inner}")
                        failed_kbs.append(kb_info.name)
                
                print(f"初始索引完成！成功: {len(success_kbs)}, 失败: {len(failed_kbs)}")
                if success_kbs:
                    print(f"成功索引: {', '.join(success_kbs)}")
                if failed_kbs:
                    print(f"索引失败: {', '.join(failed_kbs)}")
            else:
                # 只有一个知识库，索引默认知识库
                agent.index_obsidian_notes(agent.retriever.knowledge_base_name)
        except Exception as e2:
            logger.error(f"索引过程出错: {e2}")
        print("索引完成！")

    # 运行命令行界面
    agent.run_cli()


if __name__ == "__main__":
    main()
